"""Build the portable CPU notebook from reviewed local sources (no GitHub dependency)."""
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]


def build():
    paths = [ROOT / "legalqa" / n for n in ("__init__.py", "io.py", "metrics.py", "repair.py")]
    paths += [ROOT / "vendor/scoring.py", *sorted((ROOT / "vendor/rouge_score").glob("*.py"))]
    # Preserve attribution alongside the embedded scorer.
    paths += [ROOT / "NOTICE.md"]
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            data = path.read_bytes().replace(b"\r\n", b"\n")
            archive.writestr(info, data)
    payload = buffer.getvalue()
    cells = []

    def cell(kind, ident, source, metadata=None):
        item = {"cell_type": kind, "id": ident, "metadata": metadata or {},
                "source": source.strip().splitlines(keepends=True)}
        if kind == "code": item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", "s4intro", """
# LegalQA Stage 4 — Hậu xử lý CPU và kiểm chứng trước khi xuất submission

**Cách chạy:** Có thể chạy trên Kaggle hoặc máy local:
- **Trên Kaggle:** Chọn Accelerator **None**, Add Input output Stage 3 hoàn tất hoặc dataset chứa diagnostics ZIP.
- **Trên Local:** Tự động phát hiện môi trường local, tìm diagnostics ZIP hoặc file `submission.zip` để hậu xử lý và xuất `submission_repaired.zip`.

Code Stage 4 và scorer BTC được đóng gói ngay trong notebook, không cần push/clone GitHub. Mặc định chạy CPU: kiểm hash/ID/journal, xóa khối lặp nguyên văn liên tiếp, tái lập baseline dev100 rồi chấm bản sửa. Không dùng gold để sửa từng đáp án.

**Quy tắc chọn:** METEOR không giảm, lỗi lặp nặng không tăng và có khối lặp được loại. Nếu không đạt, notebook xuất `submission_original.zip`; nếu đạt, xuất `submission_repaired.zip`. Cả hai ZIP chỉ chứa `submission.json` ở gốc. Bản ứng viên được lưu để review dù bị từ chối.
""")
    cell("code", "s4config", """
from pathlib import Path
import os, sys, json, time, subprocess

SESSION_STARTED = time.monotonic()
IS_KAGGLE = Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
if IS_KAGGLE:
    INPUT = Path('/kaggle/input')
    WORK = Path('/kaggle/working')
else:
    INPUT = Path.cwd()
    WORK = Path.cwd() / 'working'
    WORK.mkdir(parents=True, exist_ok=True)
    print(f'Môi trường Local detected. WORK: {WORK}')

# None: tự tìm diagnostics ZIP, thư mục Stage 3 đã giải nén, hoặc submission.zip
DIAGNOSTICS = None
SUBMISSION = None
OUTPUT = WORK / 'legalqa_main_stage4_v8'
INSTALL_DEPS = IS_KAGGLE     # Trên Kaggle thì cài đặt NLTK data; trên local dùng môi trường có sẵn
AUDIT_ONLY = False           # True: chỉ kiểm tra/sửa ứng viên, không tạo ZIP.
WORK_HOURS = 2.0             # Ngân sách CPU gồm cài đặt + chấm, không phải thời gian dự kiến.
if not 0 < WORK_HOURS <= 9:
    raise ValueError('WORK_HOURS phải nằm trong (0, 9].')
DEADLINE = SESSION_STARTED + WORK_HOURS * 3600

def run_bounded(command, **kwargs):
    remaining = DEADLINE - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Hết ngân sách Stage 4; chưa xác nhận kết quả của phiên này.')
    return subprocess.run(list(map(str, command)), check=True, timeout=remaining, **kwargs)
""")
    cell("markdown", "s4bundle-note", """
## Code đã đóng gói

Cell sau chứa bản sao code và scorer của notebook này, kèm SHA-256. Không cần sửa payload. Muốn thay đổi thuật toán trong repo, chạy `python scripts/build_stage4_notebook.py` để tạo lại notebook.
""")
    cell("code", "s4bundle", "BUNDLE_SHA256 = " + repr(hashlib.sha256(payload).hexdigest())
         + "\nBUNDLE_B64 = " + repr(base64.b64encode(payload).decode()) + "\n",
         {"jupyter": {"source_hidden": True}})
    cell("code", "s4setup", """
import base64, hashlib, io, zipfile
from pathlib import PurePosixPath

payload = base64.b64decode(BUNDLE_B64)
if hashlib.sha256(payload).hexdigest() != BUNDLE_SHA256:
    raise ValueError('Payload code không khớp SHA-256.')
CODE = WORK / ('legalqa_stage4_code_' + BUNDLE_SHA256[:12])
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        part = PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or '\\\\' in name or ':' in name:
            raise ValueError('Đường dẫn không hợp lệ trong code bundle.')
        target = CODE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))

NLTK_ROOT = WORK / 'stage4_nltk_data'
env = dict(os.environ)
env['NLTK_DATA'] = str(NLTK_ROOT) + os.pathsep + env.get('NLTK_DATA', '')
env['PYTHONPATH'] = str(CODE)
env['PYTHONUNBUFFERED'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'

if INSTALL_DEPS and not AUDIT_ONLY:
    run_bounded([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                 'numpy>=1.26,<3', 'nltk==3.9.1', 'absl-py==2.2.2', 'six==1.17.0'])
    run_bounded([sys.executable, '-c',
        'import nltk; nltk.download("wordnet", download_dir=' + repr(str(NLTK_ROOT)) + ', raise_on_error=True); '
        'nltk.download("omw-1.4", download_dir=' + repr(str(NLTK_ROOT)) + ', raise_on_error=True)'], env=env)

if not AUDIT_ONLY:
    try:
        run_bounded([sys.executable, '-c',
                     'from legalqa.metrics import metric_environment; metric_environment(); print("Scorer ready")'],
                    cwd=CODE, env=env)
    except Exception as error:
        if IS_KAGGLE:
            raise
        print(f'Môi trường scorer local chưa đầy đủ ({error}). Chế độ sửa văn bản submission vẫn chạy được.')

print('Code:', CODE)
""")
    cell("markdown", "s4input-note", """
## Nhận diện input (Diagnostics hoặc Submission)

Ưu tiên file `legalqa_main_stage3_v8_diagnostics.zip`. Nếu không có diagnostics, notebook tự động tìm `submission.zip` để hậu xử lý trên máy local.
""")
    cell("code", "s4input", """
TARGET_MODE = 'diagnostics'
if DIAGNOSTICS is None and SUBMISSION is None:
    matches = sorted(INPUT.rglob('legalqa_main_stage3_v8_diagnostics.zip'))
    if not matches:
        matches = sorted(p.parent for p in INPUT.rglob('stage3_manifest.json'))
    if matches:
        DIAGNOSTICS = matches[0]
    else:
        sub_matches = sorted(INPUT.rglob('submission.zip'))
        if sub_matches:
            SUBMISSION = sub_matches[0]
            TARGET_MODE = 'submission'
            print(f'Phát hiện file submission: {SUBMISSION}')
        else:
            raise RuntimeError(f'Cần đúng một diagnostics ZIP hoặc submission.zip trong {INPUT}.')

if TARGET_MODE == 'diagnostics':
    DIAGNOSTICS = Path(DIAGNOSTICS)
    if not DIAGNOSTICS.exists():
        raise FileNotFoundError(DIAGNOSTICS)
    if DIAGNOSTICS.is_dir():
        packed = WORK / 'stage4_input_diagnostics.zip'
        run_bounded([sys.executable, '-c',
            'import sys; from legalqa.repair import diagnostics_zip_from_directory; '
            'diagnostics_zip_from_directory(sys.argv[1], sys.argv[2])', DIAGNOSTICS, packed], cwd=CODE, env=env)
        DIAGNOSTICS = packed
    print('Diagnostics:', DIAGNOSTICS)
else:
    SUBMISSION = Path(SUBMISSION)
    if not SUBMISSION.exists():
        raise FileNotFoundError(SUBMISSION)
    print('Submission to repair:', SUBMISSION)

print('Output:', OUTPUT)
""")
    cell("markdown", "s4run-note", """
## Sửa lặp, dọn đuôi cụt và chọn bản xuất

Chạy module hậu xử lý `legalqa.repair`: loại bỏ vòng lặp nguyên văn, khử lặp khối lớn 2 lần, dọn dẹp đuôi cụt và xuất `submission_repaired.zip`.
""")
    cell("code", "s4run", """
if TARGET_MODE == 'diagnostics':
    command = [sys.executable, '-m', 'legalqa.repair', '--diagnostics', DIAGNOSTICS, '--output', OUTPUT]
else:
    command = [sys.executable, '-m', 'legalqa.repair', '--submission', SUBMISSION, '--output', OUTPUT]
    q_matches = sorted(INPUT.rglob('public-official.json'))
    if q_matches:
        command.extend(['--questions', str(q_matches[0])])

if AUDIT_ONLY:
    command.append('--audit-only')

RUN_SUCCEEDED = False
run_bounded(command, cwd=CODE, env=env)
RUN_SUCCEEDED = True
""")
    cell("code", "s4results", """
if not globals().get('RUN_SUCCEEDED', False):
    raise RuntimeError('Chưa có lần chạy Stage 4 thành công trong phiên này.')
from IPython.display import display, FileLink
manifest = json.loads((OUTPUT / 'repair.manifest.json').read_text(encoding='utf-8'))
if (OUTPUT / 'repair.metrics.json').exists():
    report = json.loads((OUTPUT / 'repair.metrics.json').read_text(encoding='utf-8'))
    print(json.dumps(report, ensure_ascii=False, indent=2))
elif (OUTPUT / 'repair.summary.json').exists():
    summary = json.loads((OUTPUT / 'repair.summary.json').read_text(encoding='utf-8'))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

name = manifest.get('submission_zip')
if name:
    path = OUTPUT / name
    if 'files' in manifest and name in manifest['files']:
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError('Hash ZIP không khớp manifest.')
    print('FILE ĐƯỢC CHỌN ĐỂ NỘP:', path)
    display(FileLink(str(path)))
else:
    print('Audit-only: chưa tạo ZIP.')

for fname in ('repair.audit.json', 'repair.metrics.json', 'repair.summary.json', 'repair.unresolved.json', 'repair.manifest.json'):
    if (OUTPUT / fname).exists():
        display(FileLink(str(OUTPUT / fname)))
print('Hoàn tất Stage 4.')
""")
    cell("markdown", "s4next", """
## Đọc danh sách còn cần xử lý

`repair.unresolved.json` ghi các ID của **bản được chọn** cần kiểm tra tiếp. `regenerate_automatically=false`: đây không phải lệnh tự chạy GPU. Cờ chạm token chỉ yêu cầu kiểm tra đủ ý; không khẳng định đáp án chắc chắn sai.

`repair.candidate_unresolved.json` và `repair.audit.json` giữ kết quả ứng viên trước quyết định toàn tập. Nếu CPU không giải quyết được thiếu ý, bước GPU sau cần generator/tokenizer + selected adapter từ output Kaggle, kiểm hash, xác nhận context phù hợp và thử trên dev trước. Không có nhãn public để cam kết tăng điểm public, và chưa có prediction holdout trong diagnostics để xác nhận độc lập.
""")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "version": "3.11.0"}}, "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "legalqa_main_04_repair_submit.ipynb"
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
