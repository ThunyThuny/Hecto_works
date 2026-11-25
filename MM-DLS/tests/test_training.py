"""Loss masking, full-cohort optimization and held-out test policy checks."""

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from nlst.losses import cox_breslow, masked_bce
from nlst.training import evaluate_metrics, masked_ce, prognosis_loss, run_epoch, train


class ToyOSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, ct, clinical):
        risk = self.weight * clinical[:, 0]
        return {"os_risk": risk, "os_logits": risk[:, None].expand(-1, 3) * 0}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unknown_cross_entropy_is_differentiable_zero_with_nan_placeholders(self):
        logits = torch.full((3, 2), float("nan"), requires_grad=True)
        loss = masked_ce(logits, torch.full((3,), float("nan")), torch.zeros(3))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_observed_cross_entropy_does_not_touch_missing_rows(self):
        logits = torch.tensor([[2.0, 0.0], [float("nan"), float("nan")], [0.0, 2.0]], requires_grad=True)
        labels = torch.tensor([0, -100, 1])
        loss = masked_ce(logits, labels, torch.tensor([1, 0, 1]))
        expected = nn.functional.cross_entropy(logits[[0, 2]], labels[[0, 2]])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(2)))
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            masked_ce(torch.zeros(2, 2), torch.tensor([0, 2]), torch.ones(2))
        with self.assertRaises(ValueError):
            masked_ce(torch.zeros(2, 2), torch.zeros(2), torch.tensor([1.0, 0.5]))

    def observed_labels(self):
        return {
            "subtype": torch.tensor([-100, -100, -100]), "subtype_mask": torch.zeros(3),
            "stage": torch.tensor([0, -100, 2]), "stage_mask": torch.tensor([1.0, 0.0, 1.0]),
            "os_time": torch.tensor([10.0, float("nan"), 40.0]),
            "os_event": torch.tensor([1.0, float("nan"), 0.0]), "os_mask": torch.tensor([1.0, 0.0, 1.0]),
            "os_targets": torch.tensor([[1.0, 1.0, 1.0], [float("nan")] * 3, [0.0, 0.0, float("nan")]]),
            "os_target_mask": torch.tensor([[1.0, 1.0, 1.0], [0.0] * 3, [1.0, 1.0, 0.0]]),
        }

    def test_independent_classification_os_and_horizon_masks(self):
        torch.manual_seed(5)
        outputs = {
            "subtype_logits": torch.randn(3, 2, requires_grad=True),
            "stage_logits": torch.randn(3, 3, requires_grad=True),
            "os_risk": torch.randn(3, requires_grad=True),
            "os_logits": torch.randn(3, 3, requires_grad=True),
        }
        labels = self.observed_labels()
        loss, terms = prognosis_loss(outputs, labels, horizon_weight=0.7)
        self.assertEqual(set(terms), {"subtype", "stage", "os_cox", "os_horizon"})
        expected = nn.functional.cross_entropy(outputs["stage_logits"][[0, 2]], torch.tensor([0, 2]))
        expected = expected + cox_breslow(outputs["os_risk"][[0, 2]], torch.tensor([10.0, 40.0]), torch.tensor([1.0, 0.0]))
        expected = expected + 0.7 * masked_bce(outputs["os_logits"], labels["os_targets"], labels["os_target_mask"])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.equal(outputs["subtype_logits"].grad, torch.zeros(3, 2)))
        self.assertTrue(torch.equal(outputs["stage_logits"].grad[1], torch.zeros(3)))
        self.assertEqual(outputs["os_risk"].grad[1].item(), 0.0)
        self.assertTrue(torch.equal(outputs["os_logits"].grad[1], torch.zeros(3)))
        self.assertEqual(outputs["os_logits"].grad[2, 2].item(), 0.0)

    def test_all_unknown_os_loss_ignores_nan_without_fabricating_events(self):
        outputs = {"os_risk": torch.full((2,), float("nan"), requires_grad=True), "os_logits": torch.full((2, 3), float("nan"), requires_grad=True)}
        labels = {
            "os_time": torch.full((2,), float("nan")), "os_event": torch.full((2,), float("nan")), "os_mask": torch.zeros(2),
            "os_targets": torch.full((2, 3), float("nan")), "os_target_mask": torch.zeros(2, 3),
        }
        loss, terms = prognosis_loss(outputs, labels)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(terms["os_cox"].item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(outputs["os_risk"].grad, torch.zeros(2)))
        self.assertTrue(torch.equal(outputs["os_logits"].grad, torch.zeros(2, 3)))

    def batch(self, values, times, events, start):
        count = len(values)
        return {
            "ct": torch.zeros(count, 1, 1, 1, 1), "clinical": torch.tensor(values)[:, None],
            "patient_id": [str(start + i) for i in range(count)],
            "labels": {"os_time": torch.tensor(times), "os_event": torch.tensor(events), "os_mask": torch.ones(count),
                       "os_targets": torch.full((count, 3), float("nan")), "os_target_mask": torch.zeros(count, 3)},
        }

    def test_cohort_epoch_uses_all_risk_sets_and_one_optimizer_step(self):
        values = [-1.0, 0.5, 1.5, 2.0]
        times, events = [1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 0.0]
        batches = [self.batch(values[:2], times[:2], events[:2], 0), self.batch(values[2:], times[2:], events[2:], 2)]
        expected_weight = torch.tensor(0.25, requires_grad=True)
        expected_loss = cox_breslow(expected_weight * torch.tensor(values), torch.tensor(times), torch.tensor(events))
        expected_loss.backward()
        expected_updated = expected_weight.detach() - 0.1 * expected_weight.grad
        model = ToyOSModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = run_epoch(model, batches, torch.device("cpu"), optimizer, risk_set="cohort")
        self.assertEqual(step.call_count, 1)
        self.assertEqual(result["patient_ids"], ["0", "1", "2", "3"])
        self.assertAlmostEqual(result["loss"], expected_loss.item(), places=6)
        torch.testing.assert_close(model.weight.detach(), expected_updated)
        self.assertFalse(result["outputs"]["os_risk"].requires_grad)
        self.assertFalse(np.isclose(expected_loss.item(), cox_breslow(torch.tensor(values[:2]) * 0.25, torch.tensor(times[:2]), torch.tensor(events[:2])).item() / 2))

    def test_missing_class_and_all_unknown_metrics_return_none(self):
        outputs = {
            "subtype_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            "stage_logits": torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "os_risk": torch.tensor([0.3, -0.3]), "os_logits": torch.zeros(2, 3),
        }
        labels = {
            "subtype": torch.tensor([-100, -100]), "subtype_mask": torch.zeros(2),
            "stage": torch.tensor([0, 0]), "stage_mask": torch.ones(2),
            "os_time": torch.tensor([1.0, 2.0]), "os_event": torch.zeros(2), "os_mask": torch.ones(2),
            "os_targets": torch.full((2, 3), float("nan")), "os_target_mask": torch.zeros(2, 3),
        }
        metrics = evaluate_metrics({"outputs": outputs, "labels": labels, "patient_ids": ["a", "b"], "loss": 0.0})
        self.assertEqual(metrics["subtype"]["observed"], 0)
        for key in ("accuracy", "macro_f1", "auc"):
            self.assertIsNone(metrics["subtype"][key])
        self.assertEqual(metrics["stage"]["accuracy"], 1.0)
        self.assertIsNone(metrics["stage"]["auc"])
        self.assertIsNone(metrics["os"]["c_index"])
        for item in metrics["os"]["horizon_auc_observed_cases"].values():
            self.assertEqual(item["observed"], 0)
            self.assertIsNone(item["auc"])
        self.assertNotIn("dfs", metrics)

    def test_test_split_used_only_after_validation_selection_and_best_reload(self):
        class FakePreprocessor:
            def fit(self, rows): return self
            def save(self, path): Path(path).write_text("{}")
            def to_dict(self): return {}

        class FakeModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.backbone = nn.Parameter(torch.tensor(0.0))
                self.head = nn.Parameter(torch.tensor(0.0))
            def load_pretrained(self, path): return {"loaded_parameter_count": 1}
            def backbone_parameters(self): return iter([self.backbone])
            def head_parameters(self): return iter([self.head])
            def config(self): return {}
            def freeze_backbone(self, frozen): pass

        calls = []
        epochs_trained = []
        def fake_epoch(model, loader, device, optimizer=None, *args, **kwargs):
            calls.append((loader.split, optimizer is not None, model.head.item()))
            if optimizer is not None:
                epochs_trained.append(len(epochs_trained) + 1)
                with torch.no_grad(): model.head.fill_(epochs_trained[-1])
                loss = 1.0
            elif len(epochs_trained) == 1:
                loss = 0.2
            else:
                loss = 0.5
            return {"loss": loss}

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder / "manifest.csv"
            manifest.write_text("test source")
            args = SimpleNamespace(
                epochs=2, batch_size=1, patience=5, freeze_epochs=0, seed=1,
                cpu_threads=1, device="cpu", manifest=str(manifest), output_dir=str(folder / "outputs"),
                cache_dir=None, n_slices=1, image_size=8, cached_intensity_mode=None, num_workers=0,
                dropout=0.0, encode_chunk_size=1, pretrained="source.pt", backbone_lr=1e-5,
                head_lr=1e-4, weight_decay=0.0, risk_set="cohort", horizon_weight=0.5, max_batches=None,
            )
            rows = [{"patient_id": name, "split": name} for name in ("train", "validation", "test")]
            with mock.patch("nlst.training.read_manifest", return_value=rows), \
                 mock.patch("nlst.training.ClinicalPreprocessor", FakePreprocessor), \
                 mock.patch("nlst.training.NLSTDataset", side_effect=lambda rows, split, *a, **k: SimpleNamespace(split=split)), \
                 mock.patch("nlst.training.DataLoader", side_effect=lambda dataset, **k: dataset), \
                 mock.patch("nlst.training.NLSTPrognosisModel", FakeModel), \
                 mock.patch("nlst.training.run_epoch", side_effect=fake_epoch), \
                 mock.patch("nlst.training.evaluate_metrics", side_effect=lambda result: {"loss": result["loss"]}), \
                 mock.patch("nlst.training.save_predictions"), \
                 mock.patch("builtins.print"):
                final = train(args)
            self.assertEqual([split for split, _, _ in calls], ["train", "validation", "train", "validation", "validation", "test"])
            self.assertEqual(final["best_epoch"], 1)
            self.assertEqual(final["epochs_run"], 2)
            self.assertEqual(calls[-1], ("test", False, 1.0))


if __name__ == "__main__":
    unittest.main()
