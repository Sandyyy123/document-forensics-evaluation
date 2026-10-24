"""Run explicit constructed examples. No client data, detector, training or API calls."""
import json
import torch
from forensics_eval import evaluate, validation_threshold, split_audit


def run():
    torch.set_num_threads(2)
    val = torch.full((4, 4, 4), 0.1, dtype=torch.float64)
    val[:, 0, 0] = torch.tensor([0.3, 0.5, 0.8, 0.9], dtype=torch.float64)
    vm = torch.zeros_like(val); vm[2:, 0, 0] = 1
    threshold = validation_threshold(val, vm)
    test = torch.full((4, 4, 4), 0.1, dtype=torch.float64)
    test[:, 0, 0] = torch.tensor([0.85, 0.7, 0.9, 0.2], dtype=torch.float64)
    truth = torch.zeros_like(test); truth[:2, 0, 0] = 1
    records = [
        {"id": "original", "source_document": "doc-A", "split": "train"},
        {"id": "jpeg-derivative", "source_document": "doc-A", "split": "test"},
        {"id": "independent", "source_document": "doc-B", "split": "validation"}]
    return {"purpose": "Constructed evaluator demonstration; NOT measured model performance",
            "split_leakage_example": split_audit(records),
            "validation": evaluate(val, vm, threshold),
            "test_at_frozen_threshold": evaluate(test, truth, threshold),
            "limitations": ["Only 4 constructed images per split; no inferential performance claim",
                            "No trained detector, real documents, actual JPEG transform or print-scan experiment",
                            "Split audit depends on correct source-document metadata",
                            "Maximum pixel score is a baseline; not a calibrated fraud probability",
                            "Authentic empty masks excluded from IoU/Dice and assessed separately"]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
