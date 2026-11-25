# NLST confirmed-cancer prognosis fine-tuning

This additive pipeline transfers the whole-CT encoder and attention fuser from `/workspace/MM-DLS/MODELS/best_model.pt`. Clinical fusion and subtype, AJCC7 stage, OS risk/horizon heads are freshly trained. No lesion crop, PET, radiomics or DFS endpoint is fabricated.

## Study and labels

Population: confirmed first lung cancer (`conflc=1`), adenocarcinoma (`histology_cat=3`) or squamous (`2`). New subtype indices0/1=LUAD/LUSC. Stage indices0/1/2=I-II/III/IV from `de_stag_7thed`; unknowns masked. Other histologies are excluded in this two-class experiment.

OS starts at `candx_days`. Confirmed death statuses1/2/3 use `death_days`; alive0 uses valid `fup_days`. Incomplete/contradictory status and invalid chronology are masked. Raw trial offsets are used, without trial-cutoff adjustment or extended2015 mortality mixing. Months=days/30.4375. DFS is disabled, as authorized: progression does not establish a disease-free treatment origin.

Sources: [NCI Participant Dictionary](https://cdas.cancer.gov/files/download/vi4qsrq5mc/participant.dictionary.d021326.pdf), [NCI User Guide](https://cdas.cancer.gov/files/download/2f4hld9231/userguide.d032421_accessible.pdf), [progression definitions](https://cdas.cancer.gov/learn/nlst/progression/).

The nearest available CT at/before first diagnosis is selected within180days by default. This window is a configurable research choice, not an established clinical standard. One CT per patient. Stage, recurrence/death and future outcomes are labels, never clinical input features.

The available source images originated from an earlier Sybil test-source cohort. The user authorized a NEW prognosis research split. Original `official_series_split` is retained separately from new `split`; this is not untouched official Sybil benchmark evaluation. Subtype-stratified splits preserve both classes; if at least3OS deaths exist, label-only same-subtype swaps ensure a death in each split while preserving size/class counts. No model score selects split membership. Event coverage does not guarantee comparable C-index pairs.

## Inputs and transfer

DICOM: physical-position sorting, strict SeriesUID/orientation/size/rescale checks, localizer/scout exclusion, HU[-1000,400] mapped to[0,1],30uniformly sampled slices resized224x224; short stacks repeat samples. Header-verified download manifests locate IDC UUID directories. Cache is outside raw sources and keyed by source sizes/mtimes plus preprocessing.

Clinical: age at scan, sex_male, ever_smoker, pack_years, quit_years, family_history. Train-only median imputation and standardization plus6missing indicators form12inputs. All-missing train features have a documented zero standardized component. No preprocessing is fitted on validation/test.

Transferred keys: ONLY `space_encoder.*` and `space_fuser.*`,14tensors/93,184parameters out of original347,181. Keys, shapes and finite values checked atomically; checkpoint/tensor hashes recorded. Fresh heads do not assume the original class ordering. No claims that unknown original PET/radiomics preprocessing matches NLST.

## Prepare and train

The tested environment is `/workspace/sybil-env/bin/python` (PyTorch2.5.1+cu124). No existing environment was changed. For another environment, install `requirements_nlst.txt` with a suitable PyTorch wheel.

```bash
cd /workspace/MM-DLS
PY=/workspace/sybil-env/bin/python
$PY prepare_nlst_prognosis.py \
  --task-root /workspace/nfs/datasets/Medical/Lung_cancer/NLST_Training_Tasks/nlst_training_prep_20261014_095101 \
  --output-dir /workspace/Result/MM-DLS/NLST/cohort_new \
  --split-mode random --seed 42 --max-scan-to-diagnosis-days 180

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 $PY train_nlst_prognosis.py \
  --manifest /workspace/Result/MM-DLS/NLST/cohort_new/manifest.csv \
  --pretrained /workspace/MM-DLS/MODELS/best_model.pt \
  --output-dir /workspace/Result/MM-DLS/NLST/run_new \
  --cache-dir /workspace/Result/MM-DLS/NLST/cache_ct_30x224 \
  --device cuda:0 --epochs 20 --freeze-epochs 2 --patience 5 \
  --batch-size 2 --n-slices 30 --image-size 224 --risk-set cohort
```

Output directories must be new. Here cuda:0 means physicalGPU5. First2epochs freeze transferred layers; then backboneLR1e-5/headLR1e-4. Default Cox uses the FULL training riskset with Breslow ties and one update per epoch: every batch graph is retained, so memory scales with fullcohort, not batch-size. Larger cohorts require a validated scalable design; `--risk-set batch` is explicitly an approximation.

Missing tasks/horizons are masked; early censoring never becomes negative. Validation loss selects checkpoint. Test is executed after best-checkpoint selection within each run. Undefined metrics are null. Horizon AUC uses observed cases (not IPCW time-dependent AUC); sigmoid scores are not calibrated survival functions. No synthetic external-test output.

Files: best_model.pt, clinical_preprocessor.json, config.json, transfer_report.json, history.json, metrics.json, patient-level predictions_validation.csv/predictions_test.csv. Keep checkpoint and clinical preprocessor together for inference.

## Predict and test

```bash
CUDA_VISIBLE_DEVICES=5 $PY predict_nlst_prognosis.py \
  --checkpoint /workspace/Result/MM-DLS/NLST/run_new/best_model.pt \
  --manifest /workspace/Result/MM-DLS/NLST/cohort_new/manifest.csv \
  --split test --device cuda:0 \
  --output-csv /workspace/Result/MM-DLS/NLST/run_new/predictions_standalone.csv \
  --cache-dir /workspace/Result/MM-DLS/NLST/cache_ct_30x224

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
$PY -m unittest discover -s tests -p 'test_*.py' -v
```

## Completed implementation pilot

20eligible patients:14LUAD/6LUSC, actual CT-to-diagnosis gap6-161days. New14train/3validation/3test split. Five OS deaths:2train/1validation/2test. The original official split was test for all20. v1 diagnosed absent validation deaths; v2 applied the label-only coverage rule, not performance-based tuning. Both outputs are retained.

Main artifacts: `/workspace/Result/MM-DLS/NLST/cohort_20261023_prognosis_v2` and `/workspace/Result/MM-DLS/NLST/finetune_20261023_v2`. Real CT, finite losses/gradients and actual transferred-weight updates verified. Twenty cases and five deaths establish execution only, not clinical effectiveness or reliable generalization. Original scripts/checkpoint and raw data are preserved.
