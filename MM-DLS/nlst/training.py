"""Patient-level NLST prognosis fine-tuning with explicit MM-DLS CT transfer."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from lifelines.utils import concordance_index
from .data import ClinicalPreprocessor, NLSTDataset, read_manifest, WINDOW_HU
from .losses import cox_breslow, masked_bce
from .model import NLSTPrognosisModel


def masked_ce(logits, targets, mask):
    if logits.ndim != 2 or not torch.is_floating_point(logits):
        raise ValueError('Classification logits must be floating point [batch,classes]')
    targets=torch.as_tensor(targets,device=logits.device)
    mask=torch.as_tensor(mask,device=logits.device)
    if tuple(targets.shape)!=(logits.shape[0],) or targets.shape!=mask.shape:
        raise ValueError('Classification target/mask must match batch dimension')
    if not torch.isfinite(mask).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Classification mask must be binary')
    observed=mask.bool()
    if not observed.any():return logits[observed].sum()
    selected=targets[observed]
    if not torch.isfinite(selected).all() or not torch.eq(selected,selected.long()).all():
        raise ValueError('Observed classification labels must be finite integers')
    selected=selected.long()
    if (selected<0).any() or (selected>=logits.shape[1]).any():raise ValueError('Observed class outside configured head')
    if not torch.isfinite(logits[observed]).all():raise ValueError('Observed classification logits must be finite')
    return F.cross_entropy(logits[observed],selected)

def prognosis_loss(outputs, labels, horizon_weight=.5):
    terms={}
    if 'subtype_logits' in outputs:terms['subtype']=masked_ce(outputs['subtype_logits'],labels['subtype'],labels['subtype_mask'])
    if 'stage_logits' in outputs:terms['stage']=masked_ce(outputs['stage_logits'],labels['stage'],labels['stage_mask'])
    for endpoint in ['os','dfs']:
        if f'{endpoint}_risk' in outputs:
            terms[f'{endpoint}_cox']=cox_breslow(outputs[f'{endpoint}_risk'],labels[f'{endpoint}_time'],labels[f'{endpoint}_event'],labels[f'{endpoint}_mask'])
            terms[f'{endpoint}_horizon']=horizon_weight*masked_bce(outputs[f'{endpoint}_logits'],labels[f'{endpoint}_targets'],labels[f'{endpoint}_target_mask'])
    if not terms:raise ValueError('No enabled prognosis loss')
    total=sum(terms.values())
    if not torch.isfinite(total):raise FloatingPointError('Non-finite prognosis loss')
    return total,terms


def _cat(records):
    if not records:raise ValueError('Cannot collect an empty patient split')
    return {key:torch.cat([row[key] for row in records],dim=0) for key in records[0]}


def _to_device(labels,device):return {key:value.to(device) for key,value in labels.items()}


def run_epoch(model,loader,device,optimizer=None,risk_set='cohort',horizon_weight=.5,max_batches=None):
    training=optimizer is not None
    model.train(training)
    outputs_all=[];labels_all=[];pids=[];batch_losses=[]
    if training:optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step,batch in enumerate(loader):
            if max_batches is not None and step>=max_batches:break
            outputs=model(batch['ct'].to(device),batch['clinical'].to(device))
            labels=_to_device(batch['labels'],device)
            if training and risk_set=='batch':
                optimizer.zero_grad(set_to_none=True)
                loss,_=prognosis_loss(outputs,labels,horizon_weight)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
                batch_losses.append(float(loss.detach()))
                outputs_all.append({k:v.detach().cpu() for k,v in outputs.items()})
                labels_all.append({k:v.detach().cpu() for k,v in labels.items()})
            else:
                outputs_all.append(outputs if training else {k:v.cpu() for k,v in outputs.items()})
                labels_all.append(labels if training else {k:v.cpu() for k,v in labels.items()})
            pids.extend(batch['patient_id'])
        collected_outputs=_cat(outputs_all);collected_labels=_cat(labels_all)
        loss,terms=prognosis_loss(collected_outputs,collected_labels,horizon_weight)
        if training and risk_set=='cohort':
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
    if training and risk_set=='batch':
        reported_loss=float(np.mean(batch_losses))
    else:reported_loss=float(loss.detach())
    return {'loss':reported_loss,'terms':{k:float(v.detach()) for k,v in terms.items()},'outputs':{k:v.detach().cpu() for k,v in collected_outputs.items()},'labels':{k:v.detach().cpu() for k,v in collected_labels.items()},'patient_ids':pids}


def evaluate_metrics(result):
    o,y=result['outputs'],result['labels'];metrics={'patients':len(result['patient_ids']),'loss':result['loss']}
    for name in ['subtype','stage']:
        if f'{name}_logits' not in o:continue
        mask=y[f'{name}_mask'].bool().numpy();truth=y[name].numpy()[mask]
        scores=torch.softmax(o[f'{name}_logits'],dim=1).numpy()[mask]
        item={'observed':int(mask.sum()),'accuracy':None,'macro_f1':None,'auc':None}
        if len(truth):
            pred=scores.argmax(axis=1);item['accuracy']=float(accuracy_score(truth,pred));item['macro_f1']=float(f1_score(truth,pred,labels=list(range(scores.shape[1])),average='macro',zero_division=0))
            if len(np.unique(truth))==scores.shape[1]:
                item['auc']=float(roc_auc_score(truth,scores[:,1])) if scores.shape[1]==2 else float(roc_auc_score(truth,scores,multi_class='ovr',average='macro',labels=list(range(scores.shape[1]))))
        metrics[name]=item
    for name in ['os','dfs']:
        if f'{name}_risk' not in o:continue
        mask=y[f'{name}_mask'].bool().numpy();tm=y[f'{name}_time'].numpy()[mask];ev=y[f'{name}_event'].numpy()[mask];risk=o[f'{name}_risk'].numpy()[mask]
        item={'observed':int(mask.sum()),'events':int(ev.sum()),'c_index':None,'horizon_auc_observed_cases':{}}
        if len(tm)>1:
            try:item['c_index']=float(concordance_index(tm,-risk,ev))
            except ZeroDivisionError:pass
        scores=torch.sigmoid(o[f'{name}_logits']).numpy()
        for j,year in enumerate([1,3,5]):
            mh=y[f'{name}_target_mask'][:,j].bool().numpy();truth=y[f'{name}_targets'][:,j].numpy()[mh]
            item['horizon_auc_observed_cases'][f'{year}y']={'observed':int(mh.sum()),'auc':float(roc_auc_score(truth,scores[mh,j])) if len(np.unique(truth))==2 else None}
        metrics[name]=item
    return metrics


def save_predictions(result,path):
    o,y=result['outputs'],result['labels'];frame=pd.DataFrame({'patient_id':result['patient_ids']})
    for name in ['subtype','stage']:
        if f'{name}_logits' in o:
            probabilities=torch.softmax(o[f'{name}_logits'],dim=1).numpy()
            for j in range(probabilities.shape[1]):frame[f'{name}_prob_{j}']=probabilities[:,j]
            frame[name]=y[name].numpy();frame[f'{name}_mask']=y[f'{name}_mask'].numpy()
    for name in ['os','dfs']:
        if f'{name}_risk' in o:
            frame[f'{name}_risk']=o[f'{name}_risk'].numpy()
            for field in ['time','event','mask']:frame[f'{name}_{field}']=y[f'{name}_{field}'].numpy()
            probabilities=torch.sigmoid(o[f'{name}_logits']).numpy()
            for j,year in enumerate([1,3,5]):frame[f'{name}_event_score_{year}y']=probabilities[:,j]
    frame.to_csv(path,index=False)


def _write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def train(args):
    if args.epochs<1 or args.batch_size<1 or args.patience<1 or args.freeze_epochs<0:raise ValueError('Invalid training counts')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested CUDA but unavailable')
    rows=read_manifest(args.manifest)
    counts={split:sum(r.get('split')==split for r in rows) for split in ['train','validation','test']}
    if min(counts.values())<1:raise ValueError('Need nonempty patient-disjoint train/validation/test; refusing to train on official test-only data')
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    manifest_hash=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    preprocessor=ClinicalPreprocessor().fit(rows);preprocessor.save(output/'clinical_preprocessor.json')
    cache=Path(args.cache_dir).resolve() if args.cache_dir else output/'ct_cache'
    datasets={split:NLSTDataset(rows,split,preprocessor,n_slices=args.n_slices,image_size=args.image_size,cached_intensity_mode=args.cached_intensity_mode,cache_dir=cache) for split in counts}
    generator=torch.Generator().manual_seed(args.seed)
    loaders={split:DataLoader(dataset,batch_size=args.batch_size,shuffle=(split=='train'),num_workers=args.num_workers,pin_memory=device.type=='cuda',generator=generator if split=='train' else None) for split,dataset in datasets.items()}
    model=NLSTPrognosisModel(enabled_tasks=('subtype','stage','os'),dropout=args.dropout,encode_chunk_size=args.encode_chunk_size)
    transfer=model.load_pretrained(args.pretrained)
    model.to(device)
    optimizer=torch.optim.AdamW([{'params':list(model.backbone_parameters()),'lr':args.backbone_lr,'name':'transferred_CT'},{'params':list(model.head_parameters()),'lr':args.head_lr,'name':'new_fusion_and_tasks'}],weight_decay=args.weight_decay)
    config=vars(args).copy();config.update({'device_resolved':str(device),'visible_cuda_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'manifest_sha256':manifest_hash,'split_counts':counts,'model_config':model.config(),'preprocessing':{'window_hu':list(WINDOW_HU),'sampling':'uniform full stack; repeat when short','clinical_fit':'train only'},'enabled_tasks':['subtype','stage','os'],'dfs':'disabled; no validated disease-free endpoint','risk_set_note':'full training cohort once per epoch' if args.risk_set=='cohort' else 'minibatch risk sets; approximation, not full-cohort Cox likelihood','test_policy':'untouched until best validation checkpoint selected','metrics_note':'horizon AUC over observed labels; not IPCW time-dependent AUC; horizon sigmoid scores not calibrated survival functions'})
    _write_json(output/'config.json',config);_write_json(output/'transfer_report.json',transfer)
    print(json.dumps({'event':'initialized','split_counts':counts,'enabled_tasks':config['enabled_tasks'],'device':str(device),'output_dir':str(output),'transferred_parameter_count':transfer.get('transferred_parameter_count',transfer.get('loaded_parameter_count'))}),flush=True)
    best=math.inf;wait=0;history=[];start=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.freeze_backbone(epoch<=args.freeze_epochs)
        training=run_epoch(model,loaders['train'],device,optimizer,args.risk_set,args.horizon_weight,args.max_batches)
        validation=run_epoch(model,loaders['validation'],device,horizon_weight=args.horizon_weight,max_batches=args.max_batches)
        record={'epoch':epoch,'backbone_frozen':epoch<=args.freeze_epochs,'train':evaluate_metrics(training),'validation':evaluate_metrics(validation),'elapsed_seconds':round(time.monotonic()-start,2)}
        history.append(record);_write_json(output/'history.json',history)
        print(json.dumps(record,allow_nan=False),flush=True)
        if validation['loss']<best:
            best=validation['loss'];wait=0
            checkpoint={'format_version':1,'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'model_config':model.config(),'epoch':epoch,'best_validation_loss':best,'optimizer_state_dict':optimizer.state_dict(),'manifest_sha256':manifest_hash,'transfer_report':transfer,'clinical_preprocessor':preprocessor.to_dict(),'training_config':config}
            torch.save(checkpoint,output/'best_model.pt')
        else:
            wait+=1
            if wait>=args.patience:break
    checkpoint=torch.load(output/'best_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.to(device)
    final={}
    for split in ['validation','test']:
        evaluated=run_epoch(model,loaders[split],device,horizon_weight=args.horizon_weight,max_batches=args.max_batches)
        final[split]=evaluate_metrics(evaluated);save_predictions(evaluated,output/f'predictions_{split}.csv')
    final.update({'best_epoch':checkpoint['epoch'],'epochs_run':len(history),'manifest_sha256':manifest_hash,'enabled_tasks':config['enabled_tasks'],'elapsed_seconds':round(time.monotonic()-start,2),'interpretation':'implementation/pilot experiment; small cohort and unknown original pretraining preprocessing do not establish clinical performance'})
    _write_json(output/'metrics.json',final)
    print(json.dumps({'event':'completed','output_dir':str(output),'metrics':final},allow_nan=False),flush=True)
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--pretrained',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device');p.add_argument('--epochs',type=int,default=20);p.add_argument('--freeze-epochs',type=int,default=2);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--n-slices',type=int,default=30);p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--head-lr',type=float,default=1e-4);p.add_argument('--backbone-lr',type=float,default=1e-5);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--risk-set',choices=['cohort','batch'],default='cohort');p.add_argument('--horizon-weight',type=float,default=.5)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--encode-chunk-size',type=int,default=16)
    p.add_argument('--cache-dir');p.add_argument('--cached-intensity-mode',choices=['unit','hu']);p.add_argument('--max-batches',type=int)
    train(p.parse_args())

if __name__=='__main__':main()

