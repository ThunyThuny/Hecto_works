"""Small synthetic tests: clinical leakage, CT geometry, normalization, masks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from nlst.data import ClinicalPreprocessor, NLSTDataset, sample_slice_indices


def row(path, patient="0001", **changes):
    result = {
        "patient_id": patient, "series_instance_uid": "1.2.3", "split": "train",
        "image_path": str(path), "age": 60, "sex_male": 1, "ever_smoker": 1,
        "pack_years": 40, "quit_years": 10, "family_history": "",
        "subtype": 0, "subtype_mask": 1, "stage": 1, "stage_mask": 1,
        "os_time": 24, "os_event": 0, "os_mask": 1,
        "os_y1": 0, "os_m1": 1, "os_y3": "", "os_m3": 0,
        "os_y5": "", "os_m5": 0,
    }
    result.update(changes)
    return result


class ClinicalTests(unittest.TestCase):
    def test_training_only_fit_and_roundtrip(self):
        rows = [row("x", age=60), row("x", patient="0002", age=80),
                row("x", patient="0003", split="test", age=999, family_history=1)]
        prep = ClinicalPreprocessor().fit(rows)
        self.assertEqual(prep.means[0], 70)
        self.assertEqual(prep.stds[0], 10)
        self.assertEqual(prep.training_rows, 2)
        self.assertTrue(prep.all_missing[-1])
        self.assertEqual(prep.medians[-1], 0)
        transformed = prep.transform({"age": 80, "family_history": 1})
        self.assertEqual(transformed.shape, (12,))
        self.assertEqual(transformed[0], 1)
        self.assertEqual(transformed[5], 0)
        self.assertEqual(transformed[11], 0)
        missing = prep.transform({})
        np.testing.assert_array_equal(missing[6:], np.ones(6))
        self.assertEqual(missing[0], 0)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clinical.json"
            prep.save(target)
            loaded = ClinicalPreprocessor.load(target)
            np.testing.assert_array_equal(loaded.transform({}), missing)

    def test_refuses_fit_without_training_and_excludes_outcomes(self):
        with self.assertRaises(ValueError):
            ClinicalPreprocessor().fit([row("x", split="test")])
        a = ClinicalPreprocessor().fit([row("x")])
        b = ClinicalPreprocessor().fit([row("x", subtype=999, os_time=999999)])
        self.assertEqual(a.to_dict(), b.to_dict())


class PreparedCTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "volume.pt"
        # Intensity identifies the source depth index. Four evenly spaced
        # samples of eleven slices must include the final slice, not first 4.
        volume = torch.arange(11, dtype=torch.float32)[:, None, None].expand(11, 8, 8) / 10
        torch.save({"volume": volume[None], "selected_series_uid": "1.2.3"}, self.path)
        self.rows = [row(self.path)]
        self.prep = ClinicalPreprocessor().fit(self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return NLSTDataset(self.rows, "train", self.prep, n_slices=4, image_size=12, **kwargs)

    def test_explicit_normalization_and_uid_required(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.dataset()[0]
        self.rows[0]["series_instance_uid"] = "9.9.9"
        with self.assertRaisesRegex(ValueError, "SeriesInstanceUID"):
            self.dataset(cached_intensity_mode="unit")[0]

    def test_stack_sampling_batching_and_unknown_dfs_mask(self):
        item = self.dataset(cached_intensity_mode="unit")[0]
        self.assertEqual(tuple(item["ct"].shape), (4, 1, 12, 12))
        np.testing.assert_allclose(item["ct"][:, 0, 0, 0], [0, .3, .7, 1], atol=1e-6)
        self.assertEqual(item["patient_id"], "0001")
        self.assertFalse(item["labels"]["dfs_mask"])
        self.assertEqual(item["labels"]["dfs_target_mask"].tolist(), [False, False, False])
        self.assertEqual(item["labels"]["os_target_mask"].tolist(), [True, False, False])
        batch = next(iter(DataLoader(self.dataset(cached_intensity_mode="unit"), batch_size=1)))
        self.assertEqual(tuple(batch["ct"].shape), (1, 4, 1, 12, 12))
        self.assertEqual(tuple(batch["clinical"].shape), (1, 12))
        for value in item["labels"].values():
            self.assertTrue(torch.isfinite(value).all())

    def test_hu_numpy_and_repeated_slices(self):
        path = self.root / "raw.npy"
        np.save(path, np.asarray([-2000, -300, 1000], dtype=np.float32)[:, None, None])
        self.rows[0]["image_path"] = str(path)
        ds = NLSTDataset(self.rows, "train", self.prep, n_slices=5, image_size=4, cached_intensity_mode="hu")
        np.testing.assert_allclose(ds[0]["ct"][:, 0, 0, 0], [0, 0, .5, 1, 1])
        np.testing.assert_array_equal(sample_slice_indices(3, 5), [0, 0, 1, 2, 2])

    def test_prepared_cache_reuse_and_range_guard(self):
        ds = self.dataset(cached_intensity_mode="unit", cache_dir=self.root / "processed")
        expected = ds[0]["ct"]
        self.assertEqual(len(list((self.root / "processed").glob("*.pt"))), 1)
        with patch("nlst.data._prepared_ct", side_effect=AssertionError("should use cache")):
            torch.testing.assert_close(ds[0]["ct"], expected)
        torch.save(torch.full((2, 5, 5), -1.0), self.path)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.dataset(cached_intensity_mode="unit")[0]

    def test_patient_split_leakage_rejected(self):
        duplicated = self.rows + [dict(self.rows[0], split="test")]
        with self.assertRaisesRegex(ValueError, "multiple"):
            NLSTDataset(duplicated, "train", self.prep)

    def test_missing_identifier_and_unknown_split_rejected(self):
        for bad_id in (None, float("nan"), "<NA>", ""):
            invalid = [dict(self.rows[0], patient_id=bad_id)]
            with self.assertRaisesRegex(ValueError, "patient_id"):
                NLSTDataset(invalid, "train", self.prep)
        with self.assertRaisesRegex(ValueError, "split"):
            NLSTDataset(self.rows + [dict(self.rows[0], split="trian")], "train", self.prep)

    def test_active_label_cannot_hide_a_missing_value(self):
        self.rows[0]["os_time"] = ""
        with self.assertRaisesRegex(ValueError, "os"):
            self.dataset(cached_intensity_mode="unit")


class DicomTests(unittest.TestCase):
    def make_slice(self, path, uid, position, stored, image_type=None, frames=None):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = uid
        ds.Modality = "CT"
        ds.ImageType = image_type or ["ORIGINAL", "PRIMARY", "AXIAL"]
        # Plane normal points along X: Z-only sorting would fail this test.
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
        ds.ImagePositionPatient = [position, 0, 0]
        ds.Rows, ds.Columns = 5, 6
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 2, -1000
        if frames is not None:
            ds.NumberOfFrames = frames
        ds.PixelData = np.full((5, 6), stored, dtype=np.int16).tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(path, write_like_original=False)

    def test_geometry_hu_window_localizers_and_selective_decode(self):
        import pydicom
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            for depth in range(8):
                # Lexical order intentionally opposes physical depth order.
                self.make_slice(raw / f"{7-depth:02}.dcm", "1.2.3", depth, depth * 100)
            self.make_slice(raw / "localizer.dcm", "1.2.3", -9, 5000, ["LOCALIZER"])
            self.make_slice(raw / "other_series.dcm", "1.2.4", -10, 5000)
            self.make_slice(raw / "multiframe.dcm", "1.2.3", -11, 5000, frames=2)
            (raw / "junk.txt").write_text("not a DICOM", encoding="utf-8")
            rows = [row(raw)]
            prep = ClinicalPreprocessor().fit(rows)
            ds = NLSTDataset(rows, "train", prep, n_slices=3, image_size=8)
            real_read = pydicom.dcmread
            decodes = []
            def tracked_read(*args, **kwargs):
                if not kwargs.get("stop_before_pixels", False):
                    decodes.append(args[0])
                return real_read(*args, **kwargs)
            with patch("pydicom.dcmread", side_effect=tracked_read):
                ct = ds[0]["ct"]
            self.assertEqual(len(decodes), 3)
            self.assertEqual(tuple(ct.shape), (3, 1, 8, 8))
            np.testing.assert_allclose(ct[:, 0, 0, 0], [0, 800 / 1400, 1], atol=1e-6)
            with self.assertRaisesRegex(ValueError, "outside"):
                NLSTDataset(rows, "train", prep, cache_dir=raw / "cache")[0]
            self.assertFalse((raw / "cache").exists())
            rows[0]["series_instance_uid"] = "1.2.99"
            with self.assertRaisesRegex(ValueError, "matching"):
                NLSTDataset(rows, "train", prep)[0]


if __name__ == "__main__":
    unittest.main()

