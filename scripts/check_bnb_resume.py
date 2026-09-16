"""Tiny two-GPU optimizer save/load check, before loading the LegalQA model."""
import io
import json
import os


def main():
    import torch
    import bitsandbytes as bnb
    from bitsandbytes.optim.optimizer import Optimizer8bit

    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('BNB resume smoke requires two visible GPU workers')
    if not getattr(Optimizer8bit.load_state_dict, '_legalqa_resume_fix', None):
        raise RuntimeError('BNB resume compatibility hook did not load in this worker')
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    # >100k elements forces bitsandbytes to allocate paged optimizer moments.
    parameter = torch.nn.Parameter(torch.full((131072,), .125, device=device))
    optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=5e-5)
    parameter.grad = torch.full_like(parameter, .01)
    optimizer.step()
    if not getattr(optimizer.state[parameter]['state1'], 'is_paged', False):
        raise RuntimeError('Smoke did not exercise paged optimizer state')

    checkpoint = io.BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    saved_parameter = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, -.02)
    optimizer.step()

    # Trainer loads directly onto args.device. This is the failure-triggering
    # path: deserialized CUDA tensors can retain stale is_paged=True attributes.
    checkpoint.seek(0)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    restored_parameter = torch.nn.Parameter(saved_parameter)
    restored_optimizer = bnb.optim.PagedAdamW8bit([restored_parameter], lr=5e-5)
    restored_optimizer.load_state_dict(state)
    for value in restored_optimizer.state[restored_parameter].values():
        if torch.is_tensor(value):
            if value.device != device or getattr(value, 'is_paged', False):
                raise RuntimeError('Restored state is not ordinary storage on the correct GPU')
    restored_parameter.grad = torch.full_like(restored_parameter, -.02)
    restored_optimizer.step()
    torch.cuda.synchronize(rank)
    torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
    for name, expected in optimizer.state[parameter].items():
        actual = restored_optimizer.state[restored_parameter][name]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        elif actual != expected:
            raise AssertionError(f'Restored optimizer state differs: {name}')
    print('BNB_RESUME_SMOKE_OK', json.dumps({'rank':rank, 'device':str(device),
        'step':restored_optimizer.state[restored_parameter]['step'],
        'parameters_and_optimizer_match':True}), flush=True)


if __name__ == '__main__':
    main()
