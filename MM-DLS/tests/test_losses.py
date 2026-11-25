import itertools
import math
import unittest

import torch

from nlst.losses import cox_breslow, masked_bce


class LossTests(unittest.TestCase):
    def test_bce_masks_nan_placeholders_and_gradients(self):
        logits = torch.tensor([0.0, float("nan"), 1.0], requires_grad=True)
        targets = torch.tensor([1.0, float("nan"), 0.0])
        loss = masked_bce(logits, targets, torch.tensor([1, 0, 1]))
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[[0, 2]], targets[[0, 2]])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[1].item(), 0.0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bce_no_observations_is_differentiable_zero(self):
        logits = torch.tensor([float("nan")], requires_grad=True)
        loss = masked_bce(logits, [float("nan")], [0])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_bce_rejects_invalid_observed_targets_or_masks(self):
        for targets, mask in [([float("nan")], [1]), ([2], [1]), ([1], [.5])]:
            with self.assertRaises(ValueError):
                masked_bce(torch.zeros(1), targets, mask)

    def test_breslow_ties_matches_hand_calculation(self):
        risk = torch.log(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64))
        loss = cox_breslow(risk, [2, 2, 3], [1, 1, 0])
        expected = math.log(10) - (math.log(2) + math.log(3)) / 2
        self.assertAlmostEqual(loss.item(), expected, places=12)

    def test_tied_censor_is_included_in_risk_set(self):
        risk = torch.log(torch.tensor([2.0, 3.0], dtype=torch.float64))
        loss = cox_breslow(risk, [2, 2], [1, 0])
        self.assertAlmostEqual(loss.item(), math.log(5 / 2), places=12)

    def test_breslow_is_order_invariant_even_with_tied_censor(self):
        risk = torch.tensor([0.2, -0.1, 0.7, 1.0], dtype=torch.float64)
        time = torch.tensor([2.0, 2.0, 2.0, 4.0], dtype=torch.float64)
        event = torch.tensor([1, 0, 1, 1])
        expected = cox_breslow(risk, time, event)
        for perm in itertools.permutations(range(4)):
            p = torch.tensor(perm)
            torch.testing.assert_close(cox_breslow(risk[p], time[p], event[p]), expected)

    def test_untied_cox_matches_hand_calculation(self):
        risk = torch.log(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        loss = cox_breslow(risk, [1, 2, 3], [1, 0, 1])
        self.assertAlmostEqual(loss.item(), math.log(6) / 2, places=12)

    def test_no_events_and_no_valid_rows_return_differentiable_zero(self):
        for event, mask in [([0, 0], None), ([float("nan"), float("nan")], [0, 0])]:
            risk = torch.tensor([1.0, 2.0], requires_grad=True)
            loss = cox_breslow(risk, [1, 2], event, mask)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            torch.testing.assert_close(risk.grad, torch.zeros_like(risk))

    def test_masked_unknown_endpoints_do_not_enter_risk_sets(self):
        risk = torch.tensor([0.0, 0.5, float("nan")], requires_grad=True)
        loss = cox_breslow(risk, [1, 2, float("nan")], [1, 0, float("nan")], [1, 1, 0])
        expected = torch.logsumexp(risk[:2], dim=0) - risk[0]
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(risk.grad[2].item(), 0)
        self.assertTrue(torch.isfinite(risk.grad).all())

    def test_extreme_risk_is_stable_and_shift_invariant(self):
        risk = torch.tensor([1000.0, 1001.0, 999.0], dtype=torch.float64, requires_grad=True)
        loss = cox_breslow(risk, [1, 2, 2], [1, 1, 0])
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, cox_breslow(risk - 1000, [1, 2, 2], [1, 1, 0]))
        loss.backward()
        self.assertTrue(torch.isfinite(risk.grad).all())

    def test_invalid_observed_values_raise(self):
        for time, event, mask in [([-1], [1], None), ([1], [2], None), ([1], [1], [.5])]:
            with self.assertRaises(ValueError):
                cox_breslow(torch.tensor([0.0]), time, event, mask)


if __name__ == "__main__":
    unittest.main()

