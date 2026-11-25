"""Object-focused, mask-conditioned one-step SD-Turbo refinement."""
from pathlib import Path
import math
import numpy as np
import torch
from PIL import Image
from img2img_runtime import MaskTurbo, image_tensors, to_image
METHOD='pix2pix_turbo_mask_v5'

class ObjectTurbo(MaskTurbo):
    def __init__(self,base,rank=8,vae_rank=4,source_keep=0.,crop_size=512,context=1.5,device="cuda"):
        super().__init__(base,rank,vae_rank,device=device)
        if not 0<=source_keep<=1 or crop_size%8 or context<1:raise ValueError('invalid conditioning')
        self.source_keep=float(source_keep);self.crop_size=int(crop_size);self.context=float(context)

    def forward(self,source,mask,prompt,deterministic=False,return_raw=False):
        conditioned=source*(1-mask)+source*mask*self.source_keep
        return super().forward(conditioned,mask,prompt,deterministic,return_raw)

    def initialize_v4(self,checkpoint):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state.get('method')!='pix2pix_turbo_mask_v4' or state['base_revision']!=self.revision:raise ValueError('invalid v4 warm start')
        if set(state['adapter'])!=set(self.adapter_state()):raise ValueError('adapter keys mismatch')
        self.load_state_dict(state['adapter'],strict=False)

    def save(self,path,step,extra=None):
        metadata={'method':METHOD,'source_keep':self.source_keep,'crop_size':self.crop_size,'context':self.context}
        if extra:metadata.update(extra)
        super().save(path,step,metadata)

def load(base,checkpoint,device_mode="cpu-offload"):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('method')!=METHOD:raise ValueError('this demo requires the A/v5 checkpoint')
    model=ObjectTurbo(base,state['rank'],state['vae_rank'],state['source_keep'],
        state['crop_size'],state['context'],device='cpu' if device_mode=='cpu-offload' else 'cuda')
    if state['base_revision']!=model.revision or set(state['adapter'])!=set(model.adapter_state()):raise ValueError('checkpoint mismatch')
    model.load_state_dict(state['adapter'],strict=False)
    model.checkpoint_method=state['method'];model.checkpoint_steps=state['steps']
    model.eval()
    if device_mode=='cpu-offload':
        from accelerate import cpu_offload
        for module in [model.text_encoder,model.unet,model.vae]:
            cpu_offload(module,execution_device=torch.device('cuda'),offload_buffers=True)
    return model

def crop_box(mask,context=1.5):
    ys,xs=np.where(np.array(mask.convert('L'))>=128)
    if len(xs)==0:raise ValueError('empty mask')
    side=max(64,int(math.ceil(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)*context)))
    cx=(int(xs.min())+int(xs.max())+1)/2;cy=(int(ys.min())+int(ys.max())+1)/2
    left=int(math.floor(cx-side/2));top=int(math.floor(cy-side/2))
    return left,top,left+side,top+side

def crop_resize(image,box,size=512,is_mask=False):
    arr=np.array(image);l,t,r,b=box;h,w=arr.shape[:2]
    pads=[(max(0,-t),max(0,b-h)),(max(0,-l),max(0,r-w))]
    if arr.ndim==3:pads.append((0,0))
    arr=np.pad(arr,pads,mode='constant' if is_mask else 'edge')
    crop=Image.fromarray(arr).crop((l+max(0,-l),t+max(0,-t),r+max(0,-l),b+max(0,-t)))
    return crop.resize((size,size),Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)

def prepare(source,mask,target=None,size=512,context=1.5):
    box=crop_box(mask,context);result=[crop_resize(source,box,size),crop_resize(mask,box,size,True)]
    if target is not None:result.append(crop_resize(target,box,size))
    return result,box

def restore(pred,source,box):
    l,t,r,b=box;out=source.copy();out.paste(pred.resize((r-l,b-t),Image.Resampling.BICUBIC),(l,t));return out

def infer(model,image_path,mask_path,prompt,seed):
    source=Image.open(image_path).convert('RGB');mask=Image.open(mask_path).convert('L')
    if source.size!=mask.size:raise ValueError('input/mask dimensions differ')
    (src,m),box=prepare(source,mask,size=model.crop_size,context=model.context)
    x,mt=image_tensors(src,m)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=to_image(model(x,mt,prompt,deterministic=True))
    return restore(pred,source,box)
