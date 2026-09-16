"""bitsandbytes 0.45.5: clear stale UVM storage metadata after checkpoint load.

The notebook embeds this file as sitecustomize.py outside the pinned legalqa
package. Thus old training/retrieval fingerprints stay valid. Only torchrun
workers install the hook. No optimizer values or hyperparameters are reset.
"""
import hashlib
import json
import os
from pathlib import Path


PATCH_ID = 'bnb-0.45.5-restored-paged-state-v1'


def restore_cuda_storage(optimizer, is_tensor):
    """Materialize restored paged tensors on the owning parameter's GPU.

    torch.save/load preserves Python is_paged/page_deviceid attributes, not
    cudaMallocManaged allocations. Ordinary CUDA tensors must not be prefetched
    as UVM. Fresh states still use the original paged optimizer allocator.
    """
    count, size, devices = 0, 0, set()
    if not optimizer.is_paged:
        return {'tensors':count, 'bytes':size, 'devices':[]}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            for name, value in optimizer.state.get(parameter, {}).items():
                if not is_tensor(value) or not getattr(value, 'is_paged', False):
                    continue
                if parameter.device.type != 'cuda':
                    raise ValueError('Paged optimizer resume requires CUDA parameters')
                # detach + copy produces ordinary storage and drops stale tensor
                # attributes; dtype, shape and quantized moment bytes are retained.
                restored = value.detach().to(device=parameter.device, copy=True)
                restored.is_paged = False
                optimizer.state[parameter][name] = restored
                count += 1
                size += restored.numel()*restored.element_size()
                devices.add(str(restored.device))
    return {'tensors':count, 'bytes':size, 'devices':sorted(devices)}


def patch_optimizer_class(cls, is_tensor, report):
    original = cls.load_state_dict
    if getattr(original, '_legalqa_resume_fix', None) == PATCH_ID:
        return

    def load_state_dict(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        note = restore_cuda_storage(self, is_tensor)
        report(note)
        return result

    load_state_dict._legalqa_resume_fix = PATCH_ID
    cls.load_state_dict = load_state_dict


def install():
    from importlib.metadata import version
    installed = version('bitsandbytes')
    if installed != '0.45.5':
        raise RuntimeError(f'{PATCH_ID} requires bitsandbytes==0.45.5, found {installed}')
    import torch
    from bitsandbytes.optim.optimizer import Optimizer8bit

    def report(note):
        note = {'patch':PATCH_ID, 'patch_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'bitsandbytes':installed, 'torch':torch.__version__,
                'cuda':torch.version.cuda, 'rank':int(os.environ['RANK']), **note}
        print('BNB resume storage:', json.dumps(note, sort_keys=True), flush=True)
        folder = os.environ.get('LEGALQA_BNB_RESUME_REPORT_DIR')
        if folder:
            destination = Path(folder)/f"bnb_resume_rank{note['rank']}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_suffix('.json.tmp')
            pending.write_text(json.dumps(note, indent=2)+'\n', encoding='utf-8')
            os.replace(pending, destination)

    patch_optimizer_class(Optimizer8bit, torch.is_tensor, report)
    print(f'BNB resume compatibility installed: {PATCH_ID}; local_rank={os.environ["LOCAL_RANK"]}', flush=True)


if __name__ == 'sitecustomize' and 'LOCAL_RANK' in os.environ:
    # Python otherwise logs and ignores exceptions from sitecustomize. A missing
    # hook must stop the worker before it can hit the native CUDA abort again.
    try:
        install()
    except Exception as error:
        raise SystemExit(f'Cannot install BNB resume compatibility: {error}') from error
