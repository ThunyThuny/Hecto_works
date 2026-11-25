"""Build a reproducible, patient-level NLST prognosis cohort from local sources."""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from .labels import derive_participant_labels, LABEL_SOURCES

FEATURES = ['age','sex_male','ever_smoker','pack_years','quit_years','family_history']

def normalize_ids(values, numeric_patient=True):
    out = values.astype('string').str.strip()
    if numeric_patient:
        out = out.str.replace(r'\.0$', '', regex=True)
    if out.isna().any() or out.eq('').any():
        raise ValueError('Missing patient/series identifiers are not allowed')
    return out.astype(str)

def numeric(frame, column):
    return pd.to_numeric(frame[column], errors='coerce') if column in frame else pd.Series(np.nan,index=frame.index)

def index_dicom_directories(roots, known_uids):
    found = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        queue = deque([(root,0)])
        while queue:
            folder,depth = queue.popleft()
            if folder.name in known_uids:
                found.setdefault(folder.name,str(folder.resolve()))
                continue
            if depth >= 4:
                continue
            try:
                with os.scandir(folder) as entries:
                    queue.extend((Path(e.path),depth+1) for e in entries if e.is_dir(follow_symlinks=False))
            except OSError:
                continue
    return found

def index_verified_directories(task_root, uid_to_patient):
    """Use header-verified download manifests for IDC's UUID directory layout."""
    found = {}
    verification_root = Path(task_root) / 'images/download/verifications'
    data_root = Path(task_root).parents[1].resolve()
    for source in sorted(verification_root.glob('*_file_sha256.csv')):
        with pd.read_csv(source, usecols=['patient_id','series_instance_uid','path'], dtype=str, chunksize=50000) as reader:
            for chunk in reader:
                chunk=chunk.loc[chunk['series_instance_uid'].isin(uid_to_patient)].copy()
                if chunk.empty:continue
                chunk['directory']=chunk['path'].map(lambda value:str(Path(value).parent))
                records=chunk[['patient_id','series_instance_uid','path','directory']].drop_duplicates(subset=['patient_id','series_instance_uid','directory'])
                for row in records.itertuples(index=False):
                    patient=str(row.patient_id).strip().removesuffix('.0')
                    if patient!=uid_to_patient[row.series_instance_uid]:
                        raise ValueError('Verified DICOM series belongs to a different participant')
                    actual_file=Path(row.path).resolve()
                    try:actual_file.relative_to(data_root)
                    except ValueError:raise ValueError('Verified image file is outside requested Lung_cancer root')
                    if actual_file.is_file():found.setdefault(row.series_instance_uid,str(actual_file.parent))
    return found

def build_clinical_features(frame):
    raw_age = numeric(frame,'age')
    age = raw_age + np.floor(frame['scan_days']/365.25)
    sex = numeric(frame,'gender').map({1:1.0,2:0.0})
    smoking = numeric(frame,'cigsmok')
    ever = smoking.map({0:1.0,1:1.0})
    packs = numeric(frame,'pkyr').where(lambda s:s.between(0,300))
    quit_age = numeric(frame,'age_quit')
    quit_years = pd.Series(np.nan,index=frame.index)
    quit_years.loc[smoking.eq(1)] = 0.0
    former = smoking.eq(0) & quit_age.notna()
    quit_years.loc[former] = age.loc[former] - quit_age.loc[former]
    quit_years = quit_years.where(quit_years.between(0,100))
    relatives = pd.DataFrame({name:numeric(frame,name) for name in ['fammother','famfather','fambrother','famsister','famchild']})
    family = pd.Series(np.nan,index=frame.index)
    family.loc[relatives.eq(1).any(axis=1)] = 1.0
    family.loc[relatives.eq(0).all(axis=1)] = 0.0
    return pd.DataFrame({'age':age.where(age.between(18,100)),'sex_male':sex,'ever_smoker':ever,'pack_years':packs,'quit_years':quit_years,'family_history':family},index=frame.index)

def assign_splits(frame, mode='official', seed=42):
    result = frame.copy()
    if result['patient_id'].duplicated().any():
        raise ValueError('Cohort must contain exactly one selected CT per patient')
    if mode == 'official':
        aliases = {'train':'train','training':'train','dev':'validation','val':'validation','valid':'validation','validation':'validation','test':'test'}
        result['split'] = result['official_series_split'].str.lower().map(aliases)
        if result['split'].isna().any():
            raise ValueError('Unknown official split labels')
    elif mode == 'random':
        if len(result)<10:
            raise ValueError('Need at least 10 patients for a new train/validation/test research split')
        strata = result['subtype']
        if strata.value_counts().min()<3:
            raise ValueError('Need at least three patients in each supported subtype for stratification')
        rng = np.random.default_rng(seed)
        result['split'] = ''
        for label in sorted(strata.unique()):
            indices = rng.permutation(result.index[strata.eq(label)].to_numpy())
            n_val = max(1, int(round(len(indices) * .15)))
            n_test = max(1, int(round(len(indices) * .15)))
            if len(indices) - n_val - n_test < 1:
                raise ValueError('Insufficient subtype cases for three patient-disjoint splits')
            result.loc[indices[:n_val], 'split'] = 'validation'
            result.loc[indices[n_val:n_val+n_test], 'split'] = 'test'
            result.loc[indices[n_val+n_test:], 'split'] = 'train'
    else:
        raise ValueError('split mode must be official or random')
    if mode == 'random' and {'os_mask','os_event'}.issubset(result.columns):
        events = pd.to_numeric(result['os_mask'],errors='coerce').eq(1) & pd.to_numeric(result['os_event'],errors='coerce').eq(1)
        if int(events.sum()) >= 3:
            # Design the split using labels only, before any model is fitted.
            # Swap within subtype to preserve subtype and split-size strata.
            for target in ['validation','test','train']:
                if (events & result['split'].eq(target)).any():continue
                swapped=False
                recipients=rng.permutation(result.index[result['split'].eq(target) & ~events].to_numpy())
                for recipient in recipients:
                    donors=result.index[events & result['subtype'].eq(result.loc[recipient,'subtype']) & ~result['split'].eq(target)]
                    for donor in rng.permutation(donors.to_numpy()):
                        source=result.loc[donor,'split']
                        if int((events & result['split'].eq(source)).sum()) > 1:
                            result.loc[recipient,'split']=source;result.loc[donor,'split']=target
                            swapped=True;break
                    if swapped:break
                if not swapped:raise ValueError('Cannot ensure OS-event coverage while preserving subtype and split sizes')
    if result.groupby('patient_id')['split'].nunique().max()>1:
        raise ValueError('Patient leakage across splits')
    return result

def prepare(task_root, output_dir, split_mode='official', seed=42,max_scan_to_diagnosis_days=180,dicom_roots=None):
    task_root=Path(task_root).resolve();output_dir=Path(output_dir).resolve()
    ppath=task_root/'sources/clinical/participant_d021326.csv'
    spath=task_root/'cohort/official_patient_timepoint_series_splits.csv'
    participant=pd.read_csv(ppath,low_memory=False)
    participant['patient_id']=normalize_ids(participant['pid'])
    if participant['patient_id'].duplicated().any():
        raise ValueError('Participant source has duplicate IDs')
    labels=derive_participant_labels(participant)
    labels['patient_id']=normalize_ids(labels['pid'])
    participant=participant.drop(columns=['pid'])
    series=pd.read_csv(spath,dtype={'patient_id':str,'series_instance_uid':str})
    series['patient_id']=normalize_ids(series['patient_id']);series['series_instance_uid']=normalize_ids(series['series_instance_uid'], numeric_patient=False)
    if series['series_instance_uid'].duplicated().any():
        raise ValueError('Official series manifest contains duplicate series identifiers')
    if dicom_roots is None:
        dicom_roots=[task_root/'images/dicom',task_root.parents[1]/'NLST/dicom']
    image_index=index_dicom_directories(dicom_roots,set(series.series_instance_uid))
    verified_index=index_verified_directories(task_root,dict(zip(series.series_instance_uid,series.patient_id)))
    image_index.update(verified_index)
    summary={'source_participants':len(participant),'official_series_records':len(series),'existing_series_directories':len(image_index),'download_verified_series_directories':len(verified_index),'max_scan_to_diagnosis_days':max_scan_to_diagnosis_days,'seed':seed,'split_mode':split_mode,'split_policy':'subtype-stratified patient split; when >=3 OS events, same-subtype label-only swaps ensure at least one event per split','endpoints':['subtype','stage','os'],'dfs_status':'disabled: no verified disease-free-origin recurrence endpoint','label_sources':LABEL_SOURCES,'study_scope':'new confirmed-cancer prognosis study; source split and new split retained separately','selection_policy':'one nearest available screening CT at/before first diagnosis within configured day window; subtype histology_cat=2 or3 only','clinical_features':FEATURES,'survival_policy':'diagnosis-origin OS, raw trial death_days/fup_days; confirmed deathstat1/2/3, alive0; no extended mortality mixing'}
    series['image_path']=series['series_instance_uid'].map(image_index)
    candidates=series.merge(participant,on='patient_id',how='inner',validate='many_to_one')
    confirmed=numeric(candidates,'conflc').eq(1)
    supported=numeric(candidates,'histology_cat').isin([2,3])
    summary['available_confirmed_patients']=int(candidates.loc[confirmed & candidates.image_path.notna(),'patient_id'].nunique())
    candidates=candidates.loc[confirmed & supported & candidates.image_path.notna()].copy()
    year=pd.to_numeric(candidates['study_year'],errors='coerce')
    candidates['scan_days']=np.nan
    for k in [0,1,2]:
        mask=year.eq(k)
        candidates.loc[mask,'scan_days']=numeric(candidates.loc[mask],f'scr_days{k}')
    candidates['scan_to_diagnosis_days']=numeric(candidates,'candx_days')-candidates['scan_days']
    valid=np.isfinite(candidates.scan_days) & candidates.scan_days.ge(0) & candidates.scan_to_diagnosis_days.ge(0)
    summary['available_supported_before_window_patients']=int(candidates.loc[valid,'patient_id'].nunique())
    if max_scan_to_diagnosis_days is not None:
        if max_scan_to_diagnosis_days<0:raise ValueError('Scan-to-diagnosis window must be nonnegative')
        valid &= candidates.scan_to_diagnosis_days.le(max_scan_to_diagnosis_days)
    selected=candidates.loc[valid].sort_values(['patient_id','scan_to_diagnosis_days','series_instance_uid']).drop_duplicates('patient_id').reset_index(drop=True)
    if len(selected)==0:
        raise ValueError('No eligible confirmed adeno/squamous cases with available pre-diagnosis CT within window; '+json.dumps(summary))
    features=build_clinical_features(selected)
    label_columns=[c for c in labels if c not in {'pid','patient_id'}]
    manifest=selected[['patient_id','series_instance_uid','image_path','official_series_split','study_year','scan_days','candx_days','scan_to_diagnosis_days']].copy()
    for col in FEATURES:manifest[col]=features[col]
    manifest=manifest.merge(labels[['patient_id']+label_columns],on='patient_id',how='left',validate='one_to_one')
    if not manifest['subtype_mask'].eq(1).all():raise ValueError('Selected subtype cohort contains unsupported labels')
    manifest=assign_splits(manifest,split_mode,seed)
    summary.update({'selected_patients':len(manifest),'split_counts':manifest.split.value_counts().to_dict(),'source_official_split_counts':manifest.official_series_split.value_counts().to_dict(),'subtype_counts':manifest.subtype.value_counts().to_dict(),'stage_observed':int(manifest.stage_mask.sum()),'os_observed':int(manifest.os_mask.sum()),'os_events':int(manifest.loc[manifest.os_mask.eq(1),'os_event'].sum()),'os_events_by_split':{k:int(g.loc[g.os_mask.eq(1),'os_event'].sum()) for k,g in manifest.groupby('split')},'clinical_missing_counts':manifest[FEATURES].isna().sum().to_dict(),'gap_days':{'minimum':float(manifest.scan_to_diagnosis_days.min()),'maximum':float(manifest.scan_to_diagnosis_days.max())},'patient_overlap_between_splits':0})
    output_dir.mkdir(parents=True,exist_ok=False)
    manifest.to_csv(output_dir/'manifest.csv',index=False)
    summary['manifest_sha256']=hashlib.sha256((output_dir/'manifest.csv').read_bytes()).hexdigest()
    (output_dir/'cohort_summary.json').write_text(json.dumps(summary,indent=2,default=int),encoding='utf-8')
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--split-mode',choices=['official','random'],default='official')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--max-scan-to-diagnosis-days',type=float,default=180)
    parser.add_argument('--dicom-root',action='append')
    a=parser.parse_args()
    print(json.dumps(prepare(a.task_root,a.output_dir,a.split_mode,a.seed,a.max_scan_to_diagnosis_days,a.dicom_root),indent=2,default=int))

if __name__=='__main__':main()




