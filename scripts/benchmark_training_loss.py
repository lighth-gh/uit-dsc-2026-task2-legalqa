"""Kaggle CUDA A/B of the frozen Qwen lm_head + loss (not a full training step).

Run from the repository root after installing requirements.txt.
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path


def main():
    import torch
    from transformers.loss.loss_utils import ForCausalLMLoss
    from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lengths', type=int, nargs='+', default=[2433, 5067])
    parser.add_argument('--hidden-size', type=int, default=2048)
    parser.add_argument('--vocab-size', type=int, default=151936)
    parser.add_argument('--model-config', help='Optional local generator/config.json to use its exact dimensions')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', default='/kaggle/working/training_loss_benchmark.json')
    args = parser.parse_args()
    if args.model_config:
        config = json.loads(Path(args.model_config).read_text(encoding='utf-8'))
        args.hidden_size, args.vocab_size = config['hidden_size'], config['vocab_size']
    if args.repeats < 1 or any(n < 2 for n in args.lengths):
        parser.error('repeats must be positive; sequence lengths must be at least 2')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this benchmark on Kaggle GPU; CPU timings are not comparable')
    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    # prepare_model_for_kbit_training retains this frozen head in float32.
    weight = torch.randn(args.vocab_size, args.hidden_size, device=args.device) * .02
    records = []
    for length in args.lengths:
        hidden = torch.randn(1, length, args.hidden_size, device=args.device, requires_grad=True)
        labels = torch.randint(args.vocab_size, (1, length), device=args.device)
        labels[:, :min(2048, length//2)] = -100
        for mode in ('transformers', 'liger'):
            logits = loss = None
            hidden.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            times = []
            record = {'mode':mode, 'sequence_tokens':length}
            try:
                for repeat in range(args.repeats+1):
                    hidden.grad = None
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    with torch.autocast('cuda', dtype=torch.float16):
                        if mode == 'transformers':
                            logits = torch.nn.functional.linear(hidden, weight)
                            loss = ForCausalLMLoss(logits, labels, args.vocab_size)
                        else:
                            loss = LigerForCausalLMLoss(hidden, weight, labels, args.hidden_size)
                    (loss * 128).backward()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    if not torch.isfinite(loss) or not torch.isfinite(hidden.grad).all():
                        raise RuntimeError('Non-finite loss/gradient')
                    if repeat:
                        times.append(elapsed)
                    record.update(loss=float(loss.detach()),
                        peak_allocated_bytes=max(record.get('peak_allocated_bytes', 0), torch.cuda.max_memory_allocated()),
                        peak_reserved_bytes=max(record.get('peak_reserved_bytes', 0), torch.cuda.max_memory_reserved()))
                    loss = logits = None
                record.update(status='ok', median_seconds=statistics.median(times))
            except torch.OutOfMemoryError as error:
                record.update(status='oom', error=str(error))
            finally:
                loss = logits = None
                hidden.grad = None
            records.append(record)
            print(json.dumps(record), flush=True)
        del hidden, labels
    report = {'scope':'frozen lm_head + causal loss forward/backward only; excludes transformer/DDP',
              'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'settings':vars(args), 'records':records}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
