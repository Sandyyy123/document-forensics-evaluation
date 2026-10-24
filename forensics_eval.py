"""Evaluation controls, not an image-forgery detector. CPU PyTorch only."""
import math
import torch


def split_audit(records, axes=("source_document",)):
    """Flag group leakage. Different held-out-domain experiments choose their axes."""
    seen_ids = set()
    groups = {}
    for record in records:
        uid = record["id"]
        if not uid or uid in seen_ids:
            raise ValueError("Missing or duplicate record ID")
        seen_ids.add(uid)
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        for axis in axes:
            value = record.get(axis)
            if value is None or value == "":
                raise ValueError("Missing grouping metadata: " + axis)
            groups.setdefault((axis, str(value)), set()).add(record["split"])
    return [{"axis": axis, "group": group, "splits": sorted(splits)}
            for (axis, group), splits in sorted(groups.items()) if len(splits) > 1]


def inputs(probabilities, masks):
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    y = torch.as_tensor(masks, dtype=torch.float64)
    if p.ndim != 3 or p.shape != y.shape or min(p.shape) == 0:
        raise ValueError("Expected equally shaped nonempty N,H,W arrays")
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite inputs")
    if ((p < 0) | (p > 1)).any() or ((y != 0) & (y != 1)).any():
        raise ValueError("Probabilities must be in [0,1]; masks must be binary")
    return p, y.bool()


def from_logits(logits):
    """Accept binary segmentation logits N,1,H,W; never guess channel semantics."""
    x = torch.as_tensor(logits, dtype=torch.float64)
    if x.ndim != 4 or x.shape[1] != 1 or min(x.shape) == 0:
        raise ValueError("Expected binary segmentation logits N,1,H,W")
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite logits")
    return x.sigmoid()[:, 0]


def validation_threshold(probabilities, masks, max_document_fpr=0.0):
    """Lowest observed-score threshold satisfying a validation-only FPR cap.

    Image score = maximum pixel probability: deliberately simple, not calibrated.
    At least one authentic and one manipulated validation image are required.
    A threshold above 1 is permitted to encode rejecting every image.
    """
    if not 0 <= max_document_fpr <= 1:
        raise ValueError("Invalid FPR cap")
    p, y = inputs(probabilities, masks)
    authentic = ~y.flatten(1).any(1)
    if not authentic.any() or authentic.all():
        raise ValueError("Validation requires authentic and manipulated images")
    scores = p.flatten(1).amax(1)
    candidates = sorted(set([0.0, math.nextafter(1.0, math.inf)] + scores.tolist()))
    for threshold in candidates:
        fpr = (scores[authentic] >= threshold).double().mean().item()
        if fpr <= max_document_fpr:
            return threshold
    raise AssertionError("Reject-all threshold should always be feasible")


def evaluate(probabilities, masks, threshold):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid threshold")
    p, y = inputs(probabilities, masks)
    predicted = p >= threshold
    rows = []
    for score, pred, truth in zip(p, predicted, y):
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        positive = bool(truth.any())
        rows.append({"manipulated": positive, "flagged": bool(pred.any()),
                     "score": score.max().item(), "tp_pixels": tp,
                     "fp_pixels": fp, "fn_pixels": fn,
                     "iou": tp / (tp + fp + fn) if positive else None,
                     "dice": 2 * tp / (2 * tp + fp + fn) if positive else None,
                     "authentic_fp_pixel_fraction": fp / truth.numel() if not positive else None})
    positives = [r for r in rows if r["manipulated"]]
    negatives = [r for r in rows if not r["manipulated"]]
    def average(items, key):
        return sum(r[key] for r in items) / len(items) if items else None
    return {"n_images": len(rows), "n_manipulated": len(positives),
            "n_authentic": len(negatives), "frozen_threshold": threshold,
            "sensitivity": average(positives, "flagged"),
            "authentic_document_fpr": average(negatives, "flagged"),
            "mean_iou_manipulated_only": average(positives, "iou"),
            "mean_dice_manipulated_only": average(positives, "dice"),
            "mean_authentic_fp_pixel_fraction": average(negatives, "authentic_fp_pixel_fraction"),
            "per_image": rows}
