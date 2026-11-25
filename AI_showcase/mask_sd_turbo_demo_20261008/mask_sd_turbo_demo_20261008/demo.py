#!/usr/bin/env python3
"""오프라인 mask + SD-Turbo + LoRA 객체 교체 데모."""
import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--sample', help='samples/manifest.json의 샘플 ID')
    group.add_argument('--all-samples', action='store_true', help='예제 6개 실행')
    p.add_argument('--input', type=Path, help='사용자 RGB 이미지')
    p.add_argument('--object-mask', type=Path, help='원본 객체 마스크: 흰색=객체')
    p.add_argument('--edit-mask', type=Path, help='선택 사항: 직접 지정한 편집 영역')
    p.add_argument('--prompt', help='교체할 목표 객체를 설명하는 영어 문장')
    p.add_argument('--base', type=Path, default=ROOT/'models/sd-turbo')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'weights/adapter_A.pt')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/demo')
    p.add_argument('--device-mode', choices=['cpu-offload', 'cuda'], default='cpu-offload')
    p.add_argument('--dilate-px', type=int, default=12)
    p.add_argument('--feather-px', type=float, default=6)
    p.add_argument('--check', action='store_true', help='환경 및 파일 검사만 실행')
    a = p.parse_args()
    if a.dilate_px < 0 or not 0 <= a.feather_px < float('inf'):
        p.error('dilate/feather 값은 0 이상의 유한한 수여야 합니다.')
    import numpy as np
    from PIL import Image
    import torch
    from mask_postprocess import binary_mask, dilate_object_mask, composite
    from opt_runtime import load, infer, prepare
    env = {name: importlib.metadata.version(name) for name in
           ['torch', 'diffusers', 'peft', 'transformers', 'accelerate', 'numpy', 'Pillow', 'scipy']}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU가 필요합니다. CPU 전용 환경에서는 code/mask_postprocess.py만 실행할 수 있습니다.')
    if not a.base.is_dir() or not a.checkpoint.is_file():
        raise FileNotFoundError('models/sd-turbo 또는 weights/adapter_A.pt가 없습니다. 모델과 가중치가 포함된 완전한 배포 패키지를 사용하세요.')
    free, total = torch.cuda.mem_get_info()
    env.update(gpu=torch.cuda.get_device_name(0), free_mib=free//2**20, total_mib=total//2**20,
               python=sys.version, device_mode=a.device_mode, base_revision=(a.base/'REVISION').read_text().strip(),
               checkpoint_sha256=digest(a.checkpoint))
    if env['base_revision'] != 'b261bac6fd2cf515557d5d0707481eafa0485ec2':
        raise ValueError('지원하지 않는 SD-Turbo revision입니다.')
    if env['checkpoint_sha256'] != '69d4ddadb02b3241aca43e79444971cd51bfe689881f9a38939c89d013357d79':
        raise ValueError('배포용 A 체크포인트의 SHA256이 일치하지 않습니다.')
    print(json.dumps(env, ensure_ascii=False, indent=2), flush=True)
    if a.check:
        return
    custom = any([a.input, a.object_mask, a.prompt, a.edit_mask])
    if custom:
        if a.sample or a.all_samples or not (a.input and a.object_mask and a.prompt):
            p.error('사용자 입력은 --input --object-mask --prompt를 모두 지정하고 --sample/--all-samples와 분리하세요.')
        items = [dict(id='custom', input=str(a.input.resolve()), object_mask=str(a.object_mask.resolve()),
                      prompt=a.prompt, edit_mask=str(a.edit_mask.resolve()) if a.edit_mask else None)]
    else:
        items = json.loads((ROOT/'samples/manifest.json').read_text())
        if not a.all_samples:
            wanted = a.sample or '048_c9a6b76a3203fae8'
            items = [x for x in items if x['id'] == wanted]
            if not items:
                p.error('샘플 ID를 찾을 수 없습니다. samples/manifest.json을 확인하세요.')
    # Validate every input before loading the model or creating output directories.
    prepared = []
    for item in items:
        source = Image.open(ROOT/item['input']).convert('RGB')
        obj = binary_mask(Image.open(ROOT/item['object_mask']))
        edit = binary_mask(Image.open(ROOT/item['edit_mask'])) if item.get('edit_mask') else dilate_object_mask(obj, a.dilate_px)
        o, e = np.array(obj)>0, np.array(edit)>0
        if source.size != obj.size or source.size != edit.size or not o.any() or np.any(o & ~e):
            raise ValueError(f"잘못된 마스크: {item['id']}. 크기/객체 영역/포함 관계를 확인하세요.")
        prepared.append((item, source, obj, edit))
    a.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = load(a.base, a.checkpoint, a.device_mode)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-start
    results = []
    cards = []
    for item, source, obj, edit in prepared:
        folder = a.output/item['id']
        folder.mkdir()
        source.save(folder/'input.png'); obj.save(folder/'object_mask.png'); edit.save(folder/'edit_mask.png')
        # Inspect the real UNet invocation, including explicit fifth-channel mask.
        _, crop_mask = prepare(source, edit, size=model.crop_size, context=model.context)[0]
        expected = torch.from_numpy((np.array(crop_mask)>=128).copy())[None,None].float().cuda()
        expected = torch.nn.functional.interpolate(expected, size=(64,64), mode='nearest')
        trace = []
        def inspect(module, args, kwargs):
            sample, timestep = args[:2]
            trace.append(dict(shape=list(sample.shape), timestep=int(timestep),
                              mask_channel_exact=bool(torch.equal(sample[:,4:5].float(), expected))))
        hook = model.unet.register_forward_pre_hook(inspect, with_kwargs=True)
        torch.cuda.synchronize(); begin = time.perf_counter()
        try:
            raw = infer(model, folder/'input.png', folder/'edit_mask.png', item['prompt'], seed=0)
        finally:
            hook.remove()
        torch.cuda.synchronize(); inference_seconds = time.perf_counter()-begin
        final, alpha = composite(source, raw, edit, obj, a.feather_px)
        raw.save(folder/'raw.png'); final.save(folder/'output.png'); alpha.save(folder/'alpha.png')
        s, f, r = np.array(source), np.array(final), np.array(raw)
        e, o = np.array(edit)>0, np.array(obj)>0
        outside_changed = int(np.any(s!=f, axis=2)[~e].sum())
        core_exact = bool(np.array_equal(f[o], r[o]))
        verified = len(trace)==1 and trace[0]['shape']==[1,5,64,64] and trace[0]['timestep']==999 and trace[0]['mask_channel_exact'] and outside_changed==0 and core_exact
        if not verified:
            raise AssertionError(f'추론 또는 합성 검증 실패: {item["id"]}')
        reference = None
        if item.get('expected_output') and not item.get('edit_mask') and a.dilate_px==12 and a.feather_px==6:
            ref = np.array(Image.open(ROOT/item['expected_output']).convert('RGB'))
            reference = dict(pixel_exact=bool(np.array_equal(ref,f)), max_abs_rgb=int(np.abs(ref.astype(int)-f.astype(int)).max()),
                             mean_abs_rgb=float(np.abs(ref.astype(int)-f.astype(int)).mean()))
        record = dict(id=item['id'], prompt=item['prompt'], unet_trace=trace,
                      inference_seconds=inference_seconds, outside_changed_pixels=outside_changed,
                      object_core_matches_raw=core_exact, expected_comparison=reference,
                      dilate_px=a.dilate_px if not item.get('edit_mask') else None, feather_px=a.feather_px,
                      input_sha256=digest(folder/'input.png'), output_sha256=digest(folder/'output.png'),
                      verified=verified)
        (folder/'metadata.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        results.append(record)
        cards.append(f'<section><h2>{html.escape(item["id"])}</h2><p>{html.escape(item["prompt"])}</p><div class="row">'+''.join(
            f'<figure><img src="{item["id"]}/{name}.png"><figcaption>{label}</figcaption></figure>' for name,label in
            [('input','입력'),('object_mask','객체 마스크'),('edit_mask','편집 마스크'),('alpha','합성 알파'),('output','교체 결과')])+'</div></section>')
        print(json.dumps(record,ensure_ascii=False), flush=True)
    summary = dict(environment=env, model_load_seconds=load_seconds,
                   max_torch_allocated_mib=torch.cuda.max_memory_allocated()//2**20,
                   all_verified=all(x['verified'] for x in results), samples=results,
                   timing_note='GPU 동기화 포함. 모델 로딩/PNG 저장/후처리는 inference_seconds에서 제외. offload 전송은 포함.')
    (a.output/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (a.output/'browse.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Mask SD-Turbo 데모</title><style>body{font:16px sans-serif;margin:24px;background:#fafafa}.row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;width:240px}img{width:100%;height:200px;object-fit:contain;background:#eee}section{margin-bottom:32px}</style><h1>Mask + SD-Turbo + LoRA 객체 교체</h1><p>실제 단일 UNet 추론 및 마스크 외부 픽셀 보존을 검증했습니다. 예제는 선별된 성공 사례입니다.</p>'+''.join(cards)+'</html>')
    print(f'완료: {a.output.resolve()}/browse.html', flush=True)


if __name__ == '__main__':
    # Avoid hidden network dependencies in the handoff demonstration.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    main()
