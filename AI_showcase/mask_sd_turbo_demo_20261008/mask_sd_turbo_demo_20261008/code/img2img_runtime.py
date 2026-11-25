"""Mask-aware pix2pix-turbo adaptation; distinct from the failed latent regression."""
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig
from vendor.img2img_turbo_model import my_vae_encoder_fwd, my_vae_decoder_fwd

METHOD = 'pix2pix_turbo_mask_v4'
UNET_TARGETS = ['to_k','to_q','to_v','to_out.0','conv','conv1','conv2',
                'conv_shortcut','conv_out','proj_in','proj_out','ff.net.2','ff.net.0.proj']
VAE_TARGETS = ['conv1','conv2','conv_in','conv_shortcut','conv','conv_out',
               'to_k','to_q','to_v','to_out.0']

class MaskTurbo(torch.nn.Module):
    def __init__(self, base, rank=8, vae_rank=4, device="cuda"):
        super().__init__()
        pipe = StableDiffusionPipeline.from_pretrained(str(base), variant='fp16',
            torch_dtype=torch.float32, local_files_only=True, safety_checker=None,
            requires_safety_checker=False)
        self.unet, self.vae, self.text_encoder = pipe.unet, pipe.vae, pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.base, self.rank, self.vae_rank = str(base), rank, vae_rank
        self.revision = (Path(base)/'REVISION').read_text().strip()
        # Four image latent channels plus an explicit edit mask. No independent noise.
        old = self.unet.conv_in
        conv = torch.nn.Conv2d(5,old.out_channels,old.kernel_size,old.stride,old.padding)
        with torch.no_grad():
            conv.weight.zero_(); conv.weight[:,:4].copy_(old.weight); conv.bias.copy_(old.bias)
        self.unet.conv_in = conv; self.unet.register_to_config(in_channels=5)
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(self.vae.encoder)
        self.vae.decoder.forward = my_vae_decoder_fwd.__get__(self.vae.decoder)
        for i,(inc,outc) in enumerate([(512,512),(256,512),(128,512),(128,256)],1):
            layer=torch.nn.Conv2d(inc,outc,1,bias=False)
            torch.nn.init.constant_(layer.weight,1e-5)
            setattr(self.vae.decoder,f'skip_conv_{i}',layer)
        self.vae.decoder.ignore_skip=False; self.vae.decoder.gamma=1
        self.requires_grad_(False)
        self.unet.add_adapter(LoraConfig(r=rank,lora_alpha=rank,init_lora_weights='gaussian',target_modules=UNET_TARGETS))
        self.vae.add_adapter(LoraConfig(r=vae_rank,lora_alpha=vae_rank,init_lora_weights='gaussian',target_modules=VAE_TARGETS),adapter_name='vae_skip')
        self.unet.conv_in.requires_grad_(True)
        for i in range(1,5): getattr(self.vae.decoder,f'skip_conv_{i}').requires_grad_(True)
        self.to(device); self.scheduler.set_timesteps(1,device='cuda')
        assert self.scheduler.timesteps.tolist()==[999]
        self.text_encoder.eval()

    def forward(self, source, mask, prompt, deterministic=False, return_raw=False):
        # Follow the reference's image-conditioned one-step DDPM mapping.
        with torch.no_grad():
            tokens=self.tokenizer(prompt,padding='max_length',max_length=self.tokenizer.model_max_length,
                truncation=True,return_tensors='pt').input_ids.to(source.device)
            text=self.text_encoder(tokens)[0]
        dist=self.vae.encode(source).latent_dist
        z=(dist.mode() if deterministic else dist.sample())*self.vae.config.scaling_factor
        m=torch.nn.functional.interpolate(mask,size=z.shape[-2:],mode='nearest')
        t=self.scheduler.timesteps[0]
        epsilon=self.unet(torch.cat([z,m],1),t,encoder_hidden_states=text).sample
        generator=torch.Generator(device=source.device).manual_seed(0) if deterministic else None
        denoised=self.scheduler.step(epsilon,t,z,generator=generator).prev_sample.to(epsilon.dtype)
        # Retain scene detail via skips; do not directly copy the old object into its edit region.
        skip_mask=torch.zeros_like(mask) if getattr(self,'retain_edit_skips',False) else mask
        self.vae.decoder.incoming_skip_acts=[a*(1-torch.nn.functional.interpolate(skip_mask,
            size=a.shape[-2:],mode='nearest')) for a in self.vae.encoder.current_down_blocks]
        decoded=self.vae.decode(denoised/self.vae.config.scaling_factor).sample
        return decoded if return_raw else decoded.clamp(-1,1)

    def train(self,mode=True):
        super().train(mode); self.text_encoder.eval(); return self

    def adapter_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items()
            if 'lora_' in k or k.startswith('unet.conv_in.') or 'skip_conv_' in k}

    def initialize_reference(self,path):
        """Import upstream edge-to-image adapters; absorb its skip LoRA into full skips."""
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['rank_unet']!=self.rank or state['rank_vae']!=self.vae_rank:
            raise ValueError('reference ranks mismatch')
        values=self.adapter_state(); mapped={}
        for key,value in state['state_dict_unet'].items():
            name='unet.'+key
            if key=='conv_in.weight':
                expanded=torch.zeros_like(values[name]);expanded[:,:4]=value;mapped[name]=expanded
            else: mapped[name]=value
        for key,value in state['state_dict_vae'].items():
            if 'skip_conv_' in key: continue
            # Upstream PEFT uses its default alpha=8; this implementation uses alpha=rank.
            mapped['vae.'+key]=value*(8/self.vae_rank) if 'lora_B' in key else value
        for i in range(1,5):
            prefix=f'decoder.skip_conv_{i}'
            a=state['state_dict_vae'][prefix+'.lora_A.vae_skip.weight'][:,:,0,0]
            b=state['state_dict_vae'][prefix+'.lora_B.vae_skip.weight'][:,:,0,0]
            weight=state['state_dict_vae'][prefix+'.base_layer.weight']
            mapped['vae.'+prefix+'.weight']=weight+(b.float()@a.float())[:,:,None,None]*(8/self.vae_rank)
        if set(mapped)!=set(values): raise ValueError('reference adapter mapping mismatch')
        self.load_state_dict(mapped,strict=False)

    def save(self,path,step,extra=None):
        state={'method':METHOD,'rank':self.rank,'vae_rank':self.vae_rank,'base_revision':self.revision,
            'adapter':self.adapter_state(),'steps':step,'reference_commit':'86f54146590ffb4543c8cf85b5a36657da670924'}
        if extra: state.update(extra)
        path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def load(base, checkpoint=None):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True) if checkpoint else None
    if state and state.get('method')!=METHOD: raise ValueError('incompatible checkpoint architecture')
    model=MaskTurbo(base,state['rank'] if state else 8,state['vae_rank'] if state else 4)
    if state:
        if state['base_revision']!=model.revision: raise ValueError('base revision mismatch')
        if set(state['adapter'])!=set(model.adapter_state()): raise ValueError('adapter keys mismatch')
        model.load_state_dict(state['adapter'],strict=False)
    return model.eval()

def image_tensors(image,mask):
    x=torch.from_numpy(np.array(image.convert('RGB')).copy()).permute(2,0,1)[None].float().cuda()/127.5-1
    m=torch.from_numpy((np.array(mask.convert('L'))>=128).copy())[None,None].float().cuda()
    return x,m

def to_image(x):
    return Image.fromarray(((x[0].detach().float().permute(1,2,0).clamp(-1,1)/2+.5)*255).round().byte().cpu().numpy())

def infer(model,image_path,mask_path,prompt,seed):
    image=Image.open(image_path);mask=Image.open(mask_path)
    if image.size!=mask.size or image.width%8 or image.height%8: raise ValueError('invalid dimensions')
    x,m=image_tensors(image,mask)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        return to_image(model(x,m,prompt,deterministic=True))
