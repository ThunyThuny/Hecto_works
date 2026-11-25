"""Meaningful transfer, task gating and gradient checks; no patient data used."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mm_dls.HierMM_DLS import HierMM_DLS
from nlst.model import NLSTPrognosisModel


class NLSTModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "original.pt"
        self.source = HierMM_DLS(num_subtypes=2, num_tnm=3)
        torch.save(self.source.state_dict(), self.checkpoint)

    def make_model(self, **kwargs):
        return NLSTPrognosisModel(encode_chunk_size=2, dropout=0.0, **kwargs)

    def test_exact_whole_ct_transfer_and_fresh_heads(self):
        model = self.make_model()
        fresh = {
            key: value.clone() for key, value in model.state_dict().items()
            if not key.startswith(model.TRANSFER_PREFIXES)
        }
        report = model.load_pretrained(self.checkpoint)
        self.assertEqual(report["loaded_key_count"], 14)
        self.assertEqual(report["loaded_parameter_count"], 93184)
        self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        self.assertEqual(len(report["transferred_state_sha256"]), 64)
        source = self.source.state_dict()
        for key, value in model.state_dict().items():
            expected = source[key] if key.startswith(model.TRANSFER_PREFIXES) else fresh[key]
            self.assertTrue(torch.equal(value, expected), key)
        self.assertGreater(report["dropped_key_count"], 0)
        self.assertTrue(all(key.startswith(model.TRANSFER_PREFIXES) for key in report["loaded_keys"]))
        self.assertFalse(any("dfs" in key for key in model.state_dict()))

    def test_wrapped_module_prefix_and_config_resume(self):
        state = {"module." + key: value for key, value in self.source.state_dict().items()}
        torch.save({"model_state_dict": state}, self.checkpoint)
        model = self.make_model()
        model.load_pretrained(self.checkpoint)
        restored = NLSTPrognosisModel(**model.config())
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        inputs = torch.rand(2, 3, 1, 8, 8)
        clinical = torch.rand(2, 12)
        for key, value in model(inputs, clinical).items():
            torch.testing.assert_close(value, restored(inputs, clinical)[key])

    def test_incompatible_transfer_fails_before_mutation(self):
        model = self.make_model()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        state = dict(self.source.state_dict())
        state["space_encoder.encoder.0.weight"] = state["space_encoder.encoder.0.weight"][:1]
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "missing compatible"):
            model.load_pretrained(self.checkpoint)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_nonfinite_transfer_rejected(self):
        state = dict(self.source.state_dict())
        state["space_encoder.encoder.0.weight"] = state["space_encoder.encoder.0.weight"].clone()
        state["space_encoder.encoder.0.weight"][0, 0, 0, 0] = float("nan")
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.make_model().load_pretrained(self.checkpoint)

    def test_forward_backward_and_no_disabled_dfs(self):
        model = self.make_model()
        output = model(torch.rand(2, 3, 1, 8, 8), torch.rand(2, 12))
        shapes = {"subtype_logits": (2, 2), "stage_logits": (2, 3), "os_risk": (2,), "os_logits": (2, 3)}
        self.assertEqual(set(output), set(shapes))
        for key, shape in shapes.items():
            self.assertEqual(tuple(output[key].shape), shape)
            self.assertTrue(torch.isfinite(output[key]).all())
        loss = nn.functional.cross_entropy(output["subtype_logits"], torch.tensor([0, 1]))
        loss = loss + nn.functional.cross_entropy(output["stage_logits"], torch.tensor([0, 2]))
        loss = loss + output["os_risk"].square().mean()
        loss = loss + nn.functional.binary_cross_entropy_with_logits(output["os_logits"], torch.zeros(2, 3))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_frozen_warmup_then_backbone_updates(self):
        model = self.make_model()
        model.load_pretrained(self.checkpoint)
        optimizer = torch.optim.SGD([
            {"params": model.backbone_parameters(), "lr": 0.1},
            {"params": model.head_parameters(), "lr": 0.1},
        ])
        backbone = list(model.backbone_parameters())
        head_ids = {id(parameter) for parameter in model.head_parameters()}
        backbone_ids = {id(parameter) for parameter in backbone}
        self.assertFalse(head_ids.intersection(backbone_ids))
        self.assertEqual(head_ids.union(backbone_ids), {id(parameter) for parameter in model.parameters()})
        before = [parameter.detach().clone() for parameter in backbone]
        inputs, clinical = torch.rand(2, 3, 1, 8, 8), torch.rand(2, 12)
        model.freeze_backbone(True)
        model.train()
        self.assertFalse(model.space_fuser.training)
        self.assertFalse(model.space_encoder.training)
        optimizer.zero_grad(set_to_none=True)
        model(inputs, clinical)["os_logits"].square().mean().backward()
        optimizer.step()
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, backbone)))
        self.assertTrue(all(parameter.grad is None for parameter in backbone))
        model.freeze_backbone(False)
        self.assertTrue(model.space_fuser.training)
        optimizer.zero_grad(set_to_none=True)
        model(inputs, clinical)["os_logits"].square().mean().backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, backbone)))

    def test_only_explicitly_enabled_tasks_exist(self):
        model = self.make_model(enabled_tasks=("os",), clinical_dim=0)
        self.assertIsNone(model.subtype_head)
        self.assertIsNone(model.subtype_emb)
        output = model(torch.rand(2, 3, 1, 8, 8))
        self.assertEqual(set(output), {"os_risk", "os_logits"})
        model = self.make_model(enabled_tasks=("subtype", "stage", "os", "dfs"))
        output = model(torch.rand(2, 3, 1, 8, 8), torch.rand(2, 12))
        self.assertEqual(tuple(output["dfs_risk"].shape), (2,))
        self.assertEqual(tuple(output["dfs_logits"].shape), (2, 3))
        model = self.make_model(enabled_tasks=("subtype",), clinical_dim=0)
        self.assertIsNone(model.survival_mlp)
        self.assertEqual(set(model(torch.rand(2, 3, 1, 8, 8))), {"subtype_logits"})

    def test_invalid_shapes_and_tasks_fail_clearly(self):
        for tasks in ((), ("subtype", "subtype"), ("incidence",), "os"):
            with self.assertRaises(ValueError):
                self.make_model(enabled_tasks=tasks)
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "ct must have shape"):
            model(torch.rand(2, 1, 8, 8), torch.rand(2, 12))
        with self.assertRaisesRegex(ValueError, "clinical must have shape"):
            model(torch.rand(2, 3, 1, 8, 8), torch.rand(2, 6))


if __name__ == "__main__":
    unittest.main()
