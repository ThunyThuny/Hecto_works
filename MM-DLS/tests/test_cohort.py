"""Patient selection, chronology, identifier and split leakage regression tests."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nlst.cohort import assign_splits, build_clinical_features, normalize_ids, prepare


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data_root = Path(self.temporary.name) / "Lung_cancer"
        self.task_root = self.data_root / "NLST_Training_Tasks" / "prep"
        (self.task_root / "sources" / "clinical").mkdir(parents=True)
        (self.task_root / "cohort").mkdir()
        self.output_dir = Path(self.temporary.name) / "cohort_result"

    def participant(self, pid="100", **changes):
        row = dict(
            pid=pid, conflc="1", histology_cat="3", candx_days="400",
            scr_days0="0", scr_days1="365", scr_days2="730",
            de_stag_7thed="110", deathstat="0", death_days=".M", fup_days="1500",
            age="60", gender="1", cigsmok="0", pkyr="45", age_quit="50",
            fammother="0", famfather="0", fambrother="0", famsister="0", famchild="0",
        )
        row.update(changes)
        return row

    def series(self, pid, uid, year="0", split="test", alternate_root=False):
        dicom_root = self.data_root / "NLST" / "dicom" if alternate_root else self.task_root / "images" / "dicom"
        (dicom_root / str(pid) / uid).mkdir(parents=True, exist_ok=True)
        return dict(patient_id=str(pid), series_instance_uid=uid, study_year=year, official_series_split=split)

    def write_sources(self, participants, series):
        pd.DataFrame(participants).to_csv(self.task_root / "sources" / "clinical" / "participant_d021326.csv", index=False)
        pd.DataFrame(series).to_csv(self.task_root / "cohort" / "official_patient_timepoint_series_splits.csv", index=False)

    def read_manifest(self):
        return pd.read_csv(self.output_dir / "manifest.csv", dtype={"patient_id": str, "series_instance_uid": str})

    def test_normalize_patient_numeric_and_preserve_dicom_dot_zero(self):
        patients = normalize_ids(pd.Series([" 00100.0 ", "101"]))
        self.assertEqual(patients.tolist(), ["00100", "101"])
        uids = normalize_ids(pd.Series([" 1.2.840.0 ", "1.2.3.10"]), numeric_patient=False)
        self.assertEqual(uids.tolist(), ["1.2.840.0", "1.2.3.10"])
        for values in ([""], [None], ["  "]):
            with self.assertRaises(ValueError):
                normalize_ids(pd.Series(values, dtype="string"))

    def test_nearest_available_ct_before_diagnosis_with_strict_window(self):
        participants = [
            self.participant(),
            self.participant("101", candx_days=".M"),
            self.participant("102", histology_cat="2", candx_days="100", de_stag_7thed=".M", deathstat=".M", fup_days=".M"),
            self.participant("103", histology_cat="9", candx_days="100"),
            self.participant("104", conflc="0", candx_days="100"),
            self.participant("105", candx_days="300"),
        ]
        series = [
            self.series("100", "1.2.100.10", "0"),
            self.series("100", "1.2.100.12", "1"),
            self.series("100", "1.2.100.11", "1"),
            self.series("100", "1.2.100.13", "2"),
        ]
        series += [self.series(str(pid), f"1.2.{pid}.11") for pid in range(101, 106)]
        self.write_sources(participants, series)
        summary = prepare(self.task_root, self.output_dir)
        manifest = self.read_manifest().set_index("patient_id")
        self.assertEqual(set(manifest.index), {"100", "102"})
        self.assertEqual(manifest.loc["100", "series_instance_uid"], "1.2.100.11")
        self.assertEqual(manifest.loc["100", "scan_to_diagnosis_days"], 35)
        self.assertTrue(manifest["scan_to_diagnosis_days"].between(0, 180).all())
        self.assertTrue(manifest["scan_days"].le(pd.to_numeric(manifest["candx_days"])).all())
        self.assertEqual(manifest.loc["102", "stage_mask"], 0)
        self.assertEqual(manifest.loc["102", "stage"], -100)
        self.assertEqual(manifest.loc["102", "os_mask"], 0)
        self.assertTrue(np.isnan(manifest.loc["102", "os_time"]))
        self.assertTrue(np.isnan(manifest.loc["102", "os_event"]))
        self.assertTrue(manifest["dfs_mask"].eq(0).all())
        self.assertTrue(manifest["dfs_time"].isna().all())
        self.assertFalse(any(column.endswith(("_x", "_y")) for column in manifest.columns))
        self.assertEqual(summary["selected_patients"], 2)
        self.assertEqual(summary["split_counts"], {"test": 2})
        self.assertEqual(summary["source_official_split_counts"], {"test": 2})
        self.assertEqual(summary["patient_overlap_between_splits"], 0)
        self.assertEqual(summary["manifest_sha256"], hashlib.sha256((self.output_dir / "manifest.csv").read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.output_dir / "cohort_summary.json").read_text())["selected_patients"], 2)

    def test_dot_zero_series_uid_and_second_default_dicom_root(self):
        self.write_sources(
            [self.participant(candx_days="100")],
            [self.series("100", "1.2.840.0", alternate_root=True)],
        )
        summary = prepare(self.task_root, self.output_dir)
        manifest = self.read_manifest()
        self.assertEqual(summary["existing_series_directories"], 1)
        self.assertEqual(manifest.loc[0, "series_instance_uid"], "1.2.840.0")
        self.assertEqual(Path(manifest.loc[0, "image_path"]), (self.data_root / "NLST/dicom/100/1.2.840.0").resolve())

    def test_short_relative_task_root_resolves_before_parent_derivation(self):
        self.write_sources([self.participant(candx_days="100")], [self.series("100", "1.2.100.10", alternate_root=True)])
        previous = Path.cwd()
        try:
            os.chdir(self.task_root.parent)
            prepare(Path("prep"), self.output_dir)
        finally:
            os.chdir(previous)
        self.assertEqual(len(self.read_manifest()), 1)

    def test_output_directory_never_overwritten(self):
        self.write_sources([self.participant(candx_days="100")], [self.series("100", "1.2.100.10")])
        self.output_dir.mkdir()
        sentinel = self.output_dir / "existing.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare(self.task_root, self.output_dir)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse((self.output_dir / "manifest.csv").exists())

    def test_duplicate_identifiers_rejected_after_patient_normalization(self):
        self.write_sources([self.participant("100"), self.participant("100.0")], [self.series("100", "1.2.100.10")])
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            prepare(self.task_root, self.output_dir)
        self.write_sources([self.participant(candx_days="100")], [self.series("100", "1.2.100.10"), self.series("100", "1.2.100.10")])
        with self.assertRaisesRegex(ValueError, "duplicate series"):
            prepare(self.task_root, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_no_eligible_patient_leaves_no_output(self):
        self.write_sources([self.participant(candx_days=".M")], [self.series("100", "1.2.100.10")])
        with self.assertRaisesRegex(ValueError, "No eligible"):
            prepare(self.task_root, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_numeric_and_sas_missing_clinical_values_not_replaced_by_zero(self):
        frame = pd.DataFrame([
            self.participant(age="60", scan_days=365.25, cigsmok="0", age_quit="50"),
            self.participant(age=".M", scan_days=0, gender=".M", cigsmok=".M", pkyr=".M", age_quit=".M", fammother=".M"),
            self.participant(age="60", scan_days=0, cigsmok="1", age_quit=".M", famfather="1"),
        ])
        features = build_clinical_features(frame)
        self.assertEqual(features.loc[0, "age"], 61)
        self.assertEqual(features.loc[0, "quit_years"], 11)
        self.assertEqual(features.loc[0, "family_history"], 0)
        self.assertTrue(features.loc[1].isna().all())
        self.assertEqual(features.loc[2, "quit_years"], 0)
        self.assertEqual(features.loc[2, "family_history"], 1)

    def test_random_split_minor_class_three_is_stratified_disjoint_reproducible(self):
        frame = pd.DataFrame({"patient_id": [str(i) for i in range(10)], "subtype": [0] * 3 + [1] * 7, "official_series_split": ["test"] * 10})
        split = assign_splits(frame, "random", seed=91)
        repeated = assign_splits(frame, "random", seed=91)
        pd.testing.assert_frame_equal(split, repeated)
        self.assertEqual(set(split["split"]), {"train", "validation", "test"})
        self.assertTrue(split["official_series_split"].eq("test").all())
        self.assertTrue(frame["official_series_split"].eq("test").all())
        self.assertNotIn("split", frame)
        self.assertEqual(split.groupby("patient_id")["split"].nunique().max(), 1)
        for _, group in split.groupby("split"):
            self.assertEqual(set(group["subtype"]), {0, 1})

    def test_official_split_aliases_and_duplicate_patient_fail(self):
        frame = pd.DataFrame({"patient_id": ["1", "2", "3"], "official_series_split": ["training", "val", "test"], "subtype": [0, 1, 0]})
        result = assign_splits(frame)
        self.assertEqual(result["split"].tolist(), ["train", "validation", "test"])
        frame.loc[2, "official_series_split"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown official"):
            assign_splits(frame)
        frame.loc[2, "patient_id"] = "1"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            assign_splits(frame)


    def test_random_split_preserves_subtypes_and_covers_os_events_in_every_split(self):
        frame = pd.DataFrame({
            "patient_id": [str(i) for i in range(20)],
            "subtype": [0] * 14 + [1] * 6,
            "official_series_split": ["test"] * 20,
        })
        baseline = assign_splits(frame, "random", seed=42)
        # Deliberately put five real OS events in the original training split.
        # Validation and test initially have no event, requiring label-only swaps.
        event_ids = baseline.loc[baseline["split"].eq("train") & baseline["subtype"].eq(0), "patient_id"].iloc[:5]
        frame["os_mask"] = 1
        frame["os_event"] = frame["patient_id"].isin(event_ids).astype(float)
        frame["os_time"] = 24.0
        initial_events = baseline.assign(os_event=frame["os_event"]).groupby("split")["os_event"].sum()
        self.assertEqual(initial_events["validation"], 0)
        self.assertEqual(initial_events["test"], 0)
        result = assign_splits(frame, "random", seed=42)
        repeated = assign_splits(frame, "random", seed=42)
        pd.testing.assert_frame_equal(result, repeated)
        expected_counts = baseline.groupby(["split", "subtype"]).size().sort_index()
        actual_counts = result.groupby(["split", "subtype"]).size().sort_index()
        pd.testing.assert_series_equal(actual_counts, expected_counts)
        self.assertEqual(result["split"].value_counts().to_dict(), {"train": 14, "validation": 3, "test": 3})
        known_events = result["os_mask"].eq(1) & result["os_event"].eq(1)
        event_counts = result.loc[known_events].groupby("split").size().reindex(["train", "validation", "test"], fill_value=0)
        self.assertTrue(event_counts.ge(1).all(), event_counts.to_dict())
        self.assertEqual(int(event_counts.sum()), 5)
        self.assertEqual(result["patient_id"].nunique(), 20)
        self.assertEqual(result.groupby("patient_id")["split"].nunique().max(), 1)
        self.assertTrue(result["official_series_split"].eq("test").all())
        self.assertNotIn("split", frame)
        # The splitter changes roles only, never the event labels themselves.
        pd.testing.assert_series_equal(result["os_event"], frame["os_event"])


if __name__ == "__main__":
    unittest.main()
