import unittest

import numpy as np
import pandas as pd

from nlst.labels import DAYS_PER_MONTH, derive_participant_labels, horizon_targets


def participant(**overrides):
    row = {
        "pid": "synthetic",
        "conflc": 1,
        "candx_days": 100.0,
        "histology_cat": 3,
        "de_stag_7thed": 110,
        "deathstat": 0,
        "death_days": np.nan,
        "fup_days": 100.0 + 48 * DAYS_PER_MONTH,
    }
    row.update(overrides)
    return row


class LabelTests(unittest.TestCase):
    def test_exact_histology_categories_and_no_morphology_guess(self):
        rows = [participant(pid=str(code), histology_cat=code) for code in range(1, 10)]
        rows.append(participant(pid="missing", histology_cat=np.nan, de_type=8140))
        labels = derive_participant_labels(pd.DataFrame(rows))
        self.assertEqual(labels.loc[labels.subtype_mask.eq(1), "subtype"].tolist(), [1, 0])
        self.assertEqual(labels.loc[labels.pid.eq("missing"), "subtype"].item(), -100)
        self.assertEqual(labels.subtype_mask.sum(), 2)

    def test_exact_stage7_mapping_and_unknowns(self):
        codes = [110, 120, 210, 220, 310, 320, 400, 888, 900, 994, 999]
        labels = derive_participant_labels(pd.DataFrame([
            participant(pid=str(code), de_stag_7thed=code) for code in codes
        ]))
        self.assertEqual(labels.stage.tolist(), [0, 0, 0, 0, 1, 1, 2, -100, -100, -100, -100])

    def test_stage_never_falls_back_to_another_edition(self):
        labels = derive_participant_labels(pd.DataFrame([
            participant(de_stag_7thed=np.nan, de_stag=110)
        ]))
        self.assertEqual(labels.stage_mask.item(), 0)
        with self.assertRaises(ValueError):
            derive_participant_labels(pd.DataFrame([participant()]), stage_edition="6")

    def test_all_tasks_require_confirmed_cancer_and_valid_origin(self):
        rows = [
            participant(pid="notconfirmed", conflc=2),
            participant(pid="missing", candx_days=np.nan),
            participant(pid="negative", candx_days=-5),
        ]
        labels = derive_participant_labels(pd.DataFrame(rows))
        self.assertEqual(labels[["cohort_eligible", "subtype_mask", "stage_mask", "os_mask", "dfs_mask"]].to_numpy().sum(), 0)

    def test_confirmed_death_codes_and_raw_time_units(self):
        rows = [
            participant(pid=str(status), deathstat=status, death_days=100 + 12 * DAYS_PER_MONTH)
            for status in [1, 2, 3]
        ]
        labels = derive_participant_labels(pd.DataFrame(rows))
        np.testing.assert_allclose(labels.os_time, [12, 12, 12])
        self.assertEqual(labels.os_event.tolist(), [1, 1, 1])
        self.assertEqual(labels.os_y1.tolist(), [1, 1, 1])
        self.assertTrue(labels.os_policy.str.contains("no_cutoff_adjustment").all())

    def test_unconfirmed_and_unknown_deaths_are_masked(self):
        rows = [
            participant(pid=str(status), deathstat=status, death_days=300)
            for status in [4, 5, 6, 7, 9, np.nan]
        ]
        labels = derive_participant_labels(pd.DataFrame(rows))
        self.assertEqual(labels.os_mask.sum(), 0)
        self.assertTrue(labels.os_event.isna().all())

    def test_missing_death_date_does_not_become_censored(self):
        labels = derive_participant_labels(pd.DataFrame([participant(deathstat=1)]))
        self.assertEqual(labels.os_mask.item(), 0)
        self.assertTrue(np.isnan(labels.os_event.item()))

    def test_invalid_chronology_is_masked_without_clamping(self):
        labels = derive_participant_labels(pd.DataFrame([
            participant(deathstat=1, death_days=99)
        ]))
        self.assertEqual(labels.os_mask.item(), 0)
        self.assertTrue(np.isnan(labels.os_time.item()))

    def test_same_day_known_event_preserves_zero_time(self):
        labels = derive_participant_labels(pd.DataFrame([
            participant(deathstat=1, death_days=100)
        ]))
        self.assertEqual(labels.os_mask.item(), 1)
        self.assertEqual(labels.os_time.item(), 0)
        self.assertEqual(labels.os_y1.item(), 1)

    def test_diagnosis_and_death_on_randomization_day_are_valid(self):
        labels = derive_participant_labels(pd.DataFrame([
            participant(candx_days=0, deathstat=1, death_days=0)
        ]))
        self.assertEqual(labels.cohort_eligible.item(), 1)
        self.assertEqual(labels.os_mask.item(), 1)
        self.assertEqual(labels.os_time.item(), 0)
        self.assertEqual(labels.os_event.item(), 1)

    def test_contradictory_alive_death_record_is_masked(self):
        labels = derive_participant_labels(pd.DataFrame([
            participant(death_days=200),
            participant(pid="other", contactstatus=2),
        ]))
        self.assertEqual(labels.os_mask.sum(), 0)

    def test_early_censor_never_supplies_horizon_negatives(self):
        labels = derive_participant_labels(pd.DataFrame([
            participant(fup_days=100 + 6 * DAYS_PER_MONTH)
        ]))
        self.assertEqual(labels.os_event.item(), 0)
        self.assertEqual(labels.os_mask.item(), 1)
        self.assertEqual(labels[["os_m1", "os_m3", "os_m5"]].to_numpy().sum(), 0)

    def test_event_evaluable_at_later_horizons(self):
        y, m = horizon_targets([6], [1], [1])
        np.testing.assert_array_equal(y, [[1, 1, 1]])
        np.testing.assert_array_equal(m, [[1, 1, 1]])

    def test_exact_horizon_boundary_and_unknown_values(self):
        y, m = horizon_targets([12, 12, np.nan], [0, 1, np.nan], [1, 1, 0])
        np.testing.assert_array_equal(y, [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        np.testing.assert_array_equal(m, [[1, 0, 0], [1, 1, 1], [0, 0, 0]])

    def test_bad_masks_or_observed_event_codes_raise(self):
        for time, event, mask in [([1], [0], [.5]), ([1], [2], [1]), ([np.nan], [0], [1])]:
            with self.assertRaises(ValueError):
                horizon_targets(time, event, mask)

    def test_dfs_cannot_be_derived_from_progression_implicitly(self):
        labels = derive_participant_labels(pd.DataFrame([
            participant(progressed_ever=1, prog_days_1st=500, last_progfree_days=500)
        ]))
        self.assertEqual(labels.dfs_mask.item(), 0)
        self.assertTrue(np.isnan(labels.dfs_time.item()))
        self.assertTrue(labels.dfs_reason.str.contains("no disease-free-origin").all())

    def test_duplicate_or_missing_ids_raise(self):
        for frame in [
            pd.DataFrame([participant(), participant()]),
            pd.DataFrame([participant(pid=np.nan)]),
        ]:
            with self.assertRaises(ValueError):
                derive_participant_labels(frame)


if __name__ == "__main__":
    unittest.main()


