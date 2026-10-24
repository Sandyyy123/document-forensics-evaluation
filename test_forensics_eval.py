import math
import unittest
import torch
from forensics_eval import evaluate, validation_threshold, split_audit, from_logits
from main import run


class EvaluationTests(unittest.TestCase):
    def test_group_leakage_is_detected(self):
        self.assertEqual(run()["split_leakage_example"], [{"axis": "source_document", "group": "doc-A", "splits": ["test", "train"]}])

    def test_separate_groups_pass(self):
        self.assertEqual(split_audit([{"id": "a", "source_document": "A", "split": "train"}, {"id": "b", "source_document": "B", "split": "test"}]), [])

    def test_stricter_template_holdout(self):
        records = [{"id": "a", "source_document": "A", "template": "T", "split": "train"}, {"id": "b", "source_document": "B", "template": "T", "split": "test"}]
        self.assertEqual(split_audit(records), [])
        self.assertEqual(split_audit(records, ("source_document", "template"))[0]["axis"], "template")

    def test_missing_metadata_and_duplicate_id_rejected(self):
        for records in [[{"id": "a", "split": "test"}], [{"id": "a", "source_document": "A", "split": "train"}]*2]:
            with self.assertRaises(ValueError): split_audit(records)

    def test_frozen_threshold_exposes_failure(self):
        result = run()
        self.assertEqual(result["validation"]["frozen_threshold"], 0.8)
        self.assertEqual(result["validation"]["authentic_document_fpr"], 0)
        self.assertEqual(result["test_at_frozen_threshold"]["authentic_document_fpr"], 0.5)
        self.assertEqual(result["test_at_frozen_threshold"]["sensitivity"], 0.5)

    def test_empty_masks_do_not_inflate_iou(self):
        result = run()["test_at_frozen_threshold"]
        self.assertEqual(result["mean_iou_manipulated_only"], 0.5)
        self.assertIsNone(result["per_image"][2]["iou"])
        self.assertEqual(result["per_image"][2]["authentic_fp_pixel_fraction"], 1/16)

    def test_overlap_metrics_known_answer(self):
        r = evaluate([[[0.9,0.9],[0.1,0.1]]], [[[1,0],[1,0]]], 0.5)
        self.assertAlmostEqual(r["mean_iou_manipulated_only"], 1/3)
        self.assertEqual(r["mean_dice_manipulated_only"], 0.5)

    def test_absent_class_reported_unknown(self):
        r = evaluate([[[0.1]]], [[[0]]], 0.5)
        self.assertIsNone(r["sensitivity"])
        self.assertIsNone(r["mean_iou_manipulated_only"])

    def test_invalid_values_shapes_masks_rejected(self):
        for p,y in [([[[float("nan")]]],[[[0]]]), ([[[1.2]]],[[[0]]]), ([[[0.2]]],[[[0.5]]]), ([[0.2]],[[0]]),([],[])]:
            with self.assertRaises(ValueError): evaluate(p,y,0.5)

    def test_validation_requires_both_classes(self):
        with self.assertRaises(ValueError): validation_threshold([[[0.2]]],[[[0]]])

    def test_ties_and_reject_all(self):
        t = validation_threshold([[[1.0]],[[1.0]]],[[[0]],[[1]]])
        self.assertGreater(t,1.0)
        self.assertEqual(evaluate([[[1.0]]],[[[0]]],t)["authentic_document_fpr"],0)

    def test_binary_logit_adapter_and_channel_guard(self):
        p = from_logits(torch.zeros(2,1,3,3))
        self.assertEqual(tuple(p.shape),(2,3,3))
        self.assertTrue((p == 0.5).all())
        with self.assertRaises(ValueError): from_logits(torch.zeros(2,2,3,3))

    def test_invalid_threshold_and_cap(self):
        with self.assertRaises(ValueError): evaluate([[[0.1]]],[[[0]]],math.nan)
        with self.assertRaises(ValueError): validation_threshold([[[0.1]]],[[[0]]],-0.1)


if __name__ == "__main__":
    unittest.main()
