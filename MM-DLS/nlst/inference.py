"""Apply a saved NLST prognosis checkpoint without fitting preprocessing."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import ClinicalPreprocessor, NLSTDataset
from .model import NLSTPrognosisModel

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output-csv',required=True)
    p.add_argument('--preprocessor');p.add_argument('--split',choices=['train','validation','test'],default='test')
    p.add_argument('--device');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--cache-dir')
    args=p.parse_args();checkpoint_path=Path(args.checkpoint)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint.get('format_version')!=1:raise ValueError('Unsupported NLST checkpoint format')
    pre=ClinicalPreprocessor.load(args.preprocessor or checkpoint_path.parent/'clinical_preprocessor.json')
    if pre.to_dict()!=checkpoint['clinical_preprocessor']:raise ValueError('Clinical preprocessing differs from saved checkpoint')
    config=checkpoint['training_config']
    dataset=NLSTDataset(args.manifest,args.split,pre,n_slices=config['n_slices'],image_size=config['image_size'],cached_intensity_mode=config.get('cached_intensity_mode'),cache_dir=args.cache_dir)
    if len(dataset)==0:raise ValueError('No patients in requested prediction split')
    model=NLSTPrognosisModel(**checkpoint['model_config']);model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    device=torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'));model.to(device).eval()
    destination=Path(args.output_csv)
    if destination.exists():raise FileExistsError('Refusing to overwrite existing predictions')
    records=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=args.batch_size,num_workers=args.num_workers):
            outputs=model(batch['ct'].to(device),batch['clinical'].to(device))
            for j,pid in enumerate(batch['patient_id']):
                record={'patient_id':pid}
                for task,names in [('subtype',model.SUBTYPE_CLASSES),('stage',model.STAGE_CLASSES)]:
                    if f'{task}_logits' in outputs:
                        prob=torch.softmax(outputs[f'{task}_logits'][j],dim=0).cpu().tolist()
                        for name,value in zip(names,prob):record[f'{task}_prob_{name}']=value
                for task in ['os','dfs']:
                    if f'{task}_risk' in outputs:
                        record[f'{task}_risk']=float(outputs[f'{task}_risk'][j])
                        for year,value in zip([1,3,5],torch.sigmoid(outputs[f'{task}_logits'][j]).cpu().tolist()):record[f'{task}_event_score_{year}y']=value
                records.append(record)
    destination.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(records).to_csv(destination,index=False)
    print(json.dumps({'prediction_rows':len(records),'enabled_tasks':list(model.enabled_tasks),'output_csv':str(destination),'note':'OS risk and fixed-horizon event scores are not calibrated survival probabilities'}))

if __name__=='__main__':main()
