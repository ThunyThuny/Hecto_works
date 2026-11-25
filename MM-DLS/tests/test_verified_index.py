"""Verified IDC UUID directory indexing without relying on UID folder names."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nlst.cohort import index_verified_directories


class VerifiedDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "Lung_cancer"
        self.task_root = self.data_root / "NLST_Training_Tasks" / "prep"
        self.verifications = self.task_root / "images/download/verifications"
        self.verifications.mkdir(parents=True)
        self.uid = "1.2.840.10008.1.0"
        self.mapping = {self.uid: "00100"}

    def verified_path(self, parent=None, name="000001.dcm"):
        if parent is None:
            # SeriesInstanceUID is not a directory name; nesting exceeds the
            # depth handled by the bounded UID-name directory search.
            parent = (self.task_root / "images/dicom/batch-uuid/nlst-uuid"
                      / "participant-uuid/study-uuid/series-uuid")
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / name
        path.write_bytes(b"synthetic verified-path fixture; no patient data")
        return path

    def write_verification(self, records):
        pd.DataFrame(records, columns=["patient_id", "series_instance_uid", "path"]).to_csv(
            self.verifications / "batch_001_file_sha256.csv", index=False)

    def test_uuid_layout_is_indexed_by_verified_uid_and_patient(self):
        first = self.verified_path()
        second = self.verified_path(name="000002.dcm")
        self.write_verification([
            {"patient_id": "00100.0", "series_instance_uid": self.uid, "path": str(first)},
            {"patient_id": "00100", "series_instance_uid": self.uid, "path": str(second)},
        ])
        found = index_verified_directories(self.task_root, self.mapping)
        self.assertEqual(found, {self.uid: str(first.parent.resolve())})
        self.assertNotEqual(first.parent.name, self.uid)
        self.assertGreater(len(first.parent.relative_to(self.task_root / "images/dicom").parts), 4)

    def test_mismatched_participant_is_rejected(self):
        path = self.verified_path()
        self.write_verification([
            {"patient_id": "00999", "series_instance_uid": self.uid, "path": str(path)},
        ])
        with self.assertRaisesRegex(ValueError, "different participant"):
            index_verified_directories(self.task_root, self.mapping)

    def test_existing_outside_corpus_path_is_rejected(self):
        path = self.verified_path(self.root / "another_corpus/series-uuid")
        self.write_verification([
            {"patient_id": "00100", "series_instance_uid": self.uid, "path": str(path)},
        ])
        with self.assertRaisesRegex(ValueError, "outside"):
            index_verified_directories(self.task_root, self.mapping)

    def test_unknown_uid_is_ignored_before_participant_and_path_checks(self):
        known = self.verified_path()
        outside = self.verified_path(self.root / "another_corpus/unknown-uuid")
        self.write_verification([
            {"patient_id": "00100", "series_instance_uid": self.uid, "path": str(known)},
            {"patient_id": "00999", "series_instance_uid": "9.9.999", "path": str(outside)},
        ])
        self.assertEqual(index_verified_directories(self.task_root, self.mapping),
                         {self.uid: str(known.parent.resolve())})


if __name__ == "__main__":
    unittest.main()
