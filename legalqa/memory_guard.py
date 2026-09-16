"""Cooperative host-RAM guard, including Kaggle container limits."""
import os
from pathlib import Path


def available_ram_mb(proc=Path('/proc/meminfo'), cgroup=Path('/sys/fs/cgroup')):
    values = {}
    if proc.is_file():
        values = {line.split(':')[0]: int(line.split()[1])*1024
                  for line in proc.read_text().splitlines() if ':' in line}
    available = [values['MemAvailable']] if 'MemAvailable' in values else []
    for limit_file, usage_file in (('memory.max', 'memory.current'),
                                  ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes')):
        try:
            limit = (cgroup/limit_file).read_text().strip()
            if limit != 'max':
                available.append(max(0, int(limit)-int((cgroup/usage_file).read_text())))
        except FileNotFoundError:
            pass
    return min(available)/1024**2 if available else None


def low_memory():
    reserve = float(os.environ.get('LEGALQA_MIN_FREE_RAM_MB', '3072'))
    available = available_ram_mb()
    if available is not None and available < reserve:
        print(f'RAM guard: available={available:.0f} MiB < reserve={reserve:.0f} MiB; pause/save', flush=True)
        return True
    return False
