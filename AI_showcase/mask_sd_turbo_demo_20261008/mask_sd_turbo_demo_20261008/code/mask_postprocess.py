"""Reusable mask preparation and inward alpha compositing; CPU only.

The edit mask is the allowed modification region, including any intentional
dilation. The original object mask is optional and can force an opaque core.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from scipy.ndimage import distance_transform_edt

def binary_mask(mask):
    return mask.convert('L').point(lambda v:255 if v>=128 else 0)

def dilate_object_mask(object_mask,radius=12):
    if not isinstance(radius,int) or radius<0:raise ValueError('Dilation radius must be a nonnegative integer')
    mask=binary_mask(object_mask)
    return mask.filter(ImageFilter.MaxFilter(2*radius+1)) if radius else mask

def composite(source,generated,edit_mask,object_mask=None,feather_pixels=6):
    if not np.isfinite(feather_pixels) or feather_pixels<0:raise ValueError('Invalid feather width')
    source=source.convert('RGB');generated=generated.convert('RGB');edit_mask=binary_mask(edit_mask)
    if source.size!=generated.size or source.size!=edit_mask.size:raise ValueError('Image and mask dimensions differ')
    edit=np.array(edit_mask)>=128
    if not edit.any():raise ValueError('Empty edit mask')
    obj=None
    if object_mask is not None:
        if object_mask.size!=source.size:raise ValueError('Object mask dimensions differ')
        obj=np.array(binary_mask(object_mask))>=128
        if np.any(obj&~edit):raise ValueError('Object mask is outside allowed edit mask')
    if feather_pixels==0:alpha=edit.astype(float)
    else:
        distance=distance_transform_edt(np.pad(edit,1))[1:-1,1:-1]
        t=np.clip((distance-.5)/feather_pixels,0,1);alpha=t*t*(3-2*t)
    if obj is not None:alpha[obj]=1
    alpha[~edit]=0
    src=np.array(source);raw=np.array(generated)
    result=np.rint(src*(1-alpha[...,None])+raw*alpha[...,None]).astype(np.uint8)
    result[~edit]=src[~edit]
    assert np.array_equal(result[~edit],src[~edit])
    if obj is not None:assert np.array_equal(result[obj],raw[obj])
    return Image.fromarray(result),Image.fromarray(np.rint(alpha*255).astype(np.uint8))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','generated','edit-mask','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--object-mask',type=Path);p.add_argument('--feather-px',type=float,default=6)
    a=p.parse_args()
    alpha_path=a.output.with_name(a.output.stem+'_alpha.png')
    if a.output.suffix.lower()!='.png':raise ValueError('PNG is required for exact outside pixels')
    if a.output.exists() or alpha_path.exists():raise FileExistsError('Output already exists')
    result,alpha=composite(Image.open(a.input),Image.open(a.generated),Image.open(a.edit_mask),Image.open(a.object_mask) if a.object_mask else None,a.feather_px)
    a.output.parent.mkdir(parents=True,exist_ok=True);result.save(a.output);alpha.save(alpha_path)
    print(json.dumps({'output':str(a.output),'alpha':str(alpha_path),'feather_px':a.feather_px,'outside_edit_mask_exact':True}))

if __name__=='__main__':main()
