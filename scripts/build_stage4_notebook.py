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

**Cách chạy:** Import notebook vào Kaggle, chọn Accelerator **None**, bật Internet để cài scorer/WordNet. Add Input output Stage 3 hoàn tất hoặc dataset chứa diagnostics ZIP. Nếu Kaggle đã giải nén ZIP thành thư mục, notebook cũng đọc được thư mục có `stage3_manifest.json`.

Code Stage 4 và scorer BTC được đóng gói ngay trong notebook, không cần push/clone GitHub. Mặc định chạy CPU: kiểm hash/ID/journal, xóa khối lặp nguyên văn liên tiếp, tái lập baseline dev100 rồi chấm bản sửa. Không dùng gold để sửa từng đáp án.

**Quy tắc chọn:** METEOR không giảm, lỗi lặp nặng không tăng và có khối lặp được loại. Nếu không đạt, notebook xuất `submission_original.zip`; nếu đạt, xuất `submission_repaired.zip`. Cả hai ZIP chỉ chứa `submission.json` ở gốc. Bản ứng viên được lưu để review dù bị từ chối.

Đây là phần CPU của Stage 4. Các câu còn thiếu ý/lệch trọng tâm nằm trong `repair.unresolved.json`; notebook này không sinh lại bằng GPU. Diagnostics không chứa trọng số adapter. Xác nhận GPU sau cần model/adapter đúng và kiểm tra context trước.
""")
    cell("code", "s4config", """
from pathlib import Path
import os, sys, json, time, subprocess

SESSION_STARTED = time.monotonic()
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
if not INPUT.is_dir() or not WORK.is_dir():
    raise RuntimeError('Notebook này dùng đường dẫn Kaggle. Chạy local bằng python -m legalqa.repair.')

# None: tự tìm đúng một diagnostics ZIP, hoặc một thư mục Stage 3 đã giải nén.
# Nếu có nhiều phiên, điền đường dẫn của phiên COMPLETE muốn xử lý.
DIAGNOSTICS = None
OUTPUT = WORK / 'legalqa_main_stage4_v8'
INSTALL_DEPS = True          # Tắt nếu môi trường đã có scorer dependencies + WordNet.
AUDIT_ONLY = False           # True: chỉ kiểm tra/sửa ứng viên, không chấm và KHÔNG tạo ZIP.
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
    # A failed resource download must stop the run, rather than silently changing METEOR.
    run_bounded([sys.executable, '-c',
        'import nltk; nltk.download("wordnet", download_dir=' + repr(str(NLTK_ROOT)) + ', raise_on_error=True); '
        'nltk.download("omw-1.4", download_dir=' + repr(str(NLTK_ROOT)) + ', raise_on_error=True)'], env=env)
if not AUDIT_ONLY:
    run_bounded([sys.executable, '-c',
                 'from legalqa.metrics import metric_environment; metric_environment(); print("Scorer ready")'],
                cwd=CODE, env=env)
print('Code:', CODE)
""")
    cell("markdown", "s4input-note", """
## Nhận diện diagnostics

Ưu tiên file `legalqa_main_stage3_v8_diagnostics.zip`. Nếu không thấy ZIP, tìm `stage3_manifest.json` trong dataset đã giải nén. Khi có nhiều kết quả, đặt `DIAGNOSTICS` ở cell cấu hình; không tự chọn phiên mới nhất hoặc một file partial.
""")
    cell("code", "s4input", """
if DIAGNOSTICS is None:
    matches = sorted(INPUT.rglob('legalqa_main_stage3_v8_diagnostics.zip'))
    if not matches:
        matches = sorted(p.parent for p in INPUT.rglob('stage3_manifest.json'))
    if len(matches) != 1:
        raise RuntimeError(f'Cần đúng một input Stage 3. Tìm thấy {len(matches)}: {matches}. Đặt DIAGNOSTICS cụ thể.')
    DIAGNOSTICS = matches[0]
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
print('Output:', OUTPUT)
""")
    cell("markdown", "s4run-note", """
## Sửa lặp, chấm dev100 và chọn bản xuất

Giữ nguyên các mục gần giống nhưng khác số liệu/phủ định. Câu chỉ còn dẫn nhập sau xóa lặp được giữ bản gốc và đưa vào danh sách cần xử lý tiếp. Không cắt mọi câu xuống một độ dài cố định; không phục hồi raw output đã bị guard của Stage 3 loại.

Chạy lại cùng input/code/cấu hình được phép. Khi đổi nguồn hoặc chính sách, chọn `OUTPUT` mới; không sửa/xóa identity để ép tái sử dụng kết quả cũ. Trong chế độ audit-only không xuất ZIP. Sau khi sửa lỗi môi trường, có thể chạy lại cell này với cùng identity.
""")
    cell("code", "s4run", """
command = [sys.executable, '-m', 'legalqa.repair', '--diagnostics', DIAGNOSTICS, '--output', OUTPUT]
if AUDIT_ONLY:
    command.append('--audit-only')
# Nếu subprocess lỗi/timeout, cell dừng tại đây; cell xuất kết quả không được xác nhận bằng run cũ.
RUN_SUCCEEDED = False
run_bounded(command, cwd=CODE, env=env)
RUN_SUCCEEDED = True
""")
    cell("code", "s4results", """
if not globals().get('RUN_SUCCEEDED', False):
    raise RuntimeError('Chưa có lần chạy Stage 4 thành công trong phiên này.')
from IPython.display import display, FileLink
report = json.loads((OUTPUT / 'repair.metrics.json').read_text(encoding='utf-8'))
manifest = json.loads((OUTPUT / 'repair.manifest.json').read_text(encoding='utf-8'))
print(json.dumps(report, ensure_ascii=False, indent=2))
name = manifest.get('submission_zip')
if name:
    path = OUTPUT / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][name]:
        raise ValueError('Hash ZIP không khớp manifest.')
    print('FILE ĐƯỢC CHỌN ĐỂ NỘP:', path)
    display(FileLink(str(path)))
else:
    print('Audit-only: chưa tạo ZIP. Chạy chế độ có chấm điểm với OUTPUT mới để chọn bản nộp.')
for name in ('repair.audit.json', 'repair.metrics.json', 'repair.unresolved.json', 'repair.manifest.json'):
    display(FileLink(str(OUTPUT / name)))
print('Các chỉ số ở đây là dev100, chưa phải điểm public của BTC.')
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
