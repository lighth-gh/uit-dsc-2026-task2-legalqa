"""Build the portable Stage 4 CPU/GPU notebook (no GitHub dependency)."""
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]


def build():
    paths = sorted((ROOT / "legalqa").glob("*.py"))
    paths += [ROOT / "assets/approved_models.json"]
    paths += [ROOT / "vendor/scoring.py", *sorted((ROOT / "vendor/rouge_score").glob("*.py"))]
    # Preserve attribution alongside the embedded scorer.
    paths += [ROOT / "NOTICE.md"]
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    cells = []

    def cell(kind, ident, source, metadata=None):
        item = {"cell_type": kind, "id": ident, "metadata": metadata or {},
                "source": source.strip().splitlines(keepends=True)}
        if kind == "code": item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", "s4intro", """
# LegalQA Stage 4 V2 — Lọc CPU và thử sinh lại có chọn lọc trên GPU

**Cách chạy:** Import notebook vào Kaggle, chọn GPU T4/P100, bật Internet để cài dependencies. Add Input diagnostics Stage 3 hoàn tất (hỗ trợ tên có hậu tố như `(5).zip`), output Stage 2/3 có `selected_adapter`, và dataset model gốc có `models.lock.json` cùng các thư mục `generator/`, `embedding/`, `reranker/`. Diagnostics không chứa trọng số. Chỉ CPU: đặt `RUN_GPU=False`, Accelerator None.

Code và scorer BTC được nhúng trong notebook. CPU so sánh bản cũ V1 với V2 (thêm vòng lặp đổi nhãn danh sách), giữ bản có METEOR không thấp hơn. Giữ phần kết luận: thử xóa kết luận trên diagnostics (5) làm giảm METEOR. Không dùng gold làm prompt hoặc chọn đáp án riêng cho từng ID.

GPU dùng đúng adapter/model đã kiểm identity, giữ context/top-k/ngân sách token, thêm repetition penalty 1.08 và no-repeat 12-gram. Chỉ thử câu có cờ lỗi và evidence đủ mạnh theo heuristic. Toàn bộ nhóm dev được thử trước public; cần METEOR toàn dev100 tăng ít nhất 0,001, ít nhất 2 câu thay đổi và lặp nặng không tăng. Nếu không đạt, giữ CPU và không chạy public GPU. Đây là một cấu hình thử nghiệm cố định, chưa được benchmark GPU.

Đầu ra `submission_selected.zip` luôn có đủ 1.000 ID và đúng một `submission.json`. Khi GPU **paused**, ZIP hiện tại là bản CPU; Add Input toàn bộ output vừa lưu và đặt `PREVIOUS_OUTPUT` để tiếp tục. Mục tiêu public 0,59 chưa được bảo đảm bởi điểm dev100.
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
OUTPUT = WORK / 'legalqa_main_stage4_v2'
RUN_GPU = True
MODEL_ROOT = None            # Thư mục chứa models.lock.json và generator/.
ADAPTER_ROOT = None          # Thư mục selected_adapter chứa trọng số + adapter_config.json.
PREVIOUS_OUTPUT = None       # Thư mục legalqa_main_stage4_v2 của phiên paused, từ Add Input.
GPU_MAX_ITEMS = 50           # Tổng câu mới mỗi phiên, cả dev và public; các phiên sau resume.
INSTALL_DEPS = True          # Tắt nếu môi trường đã có scorer dependencies + WordNet.
AUDIT_ONLY = False           # True: chỉ kiểm tra/sửa ứng viên, không chấm và KHÔNG tạo ZIP.
WORK_HOURS = 9.0             # Gồm cài đặt, CPU, GPU và chấm; không cam kết xong trong một phiên.
if not 0 < WORK_HOURS <= 9:
    raise ValueError('WORK_HOURS phải nằm trong (0, 9].')
DEADLINE = SESSION_STARTED + WORK_HOURS * 3600
if RUN_GPU and AUDIT_ONLY:
    raise ValueError('RUN_GPU không dùng cùng AUDIT_ONLY.')
if not isinstance(GPU_MAX_ITEMS, int) or GPU_MAX_ITEMS <= 0:
    raise ValueError('GPU_MAX_ITEMS phải là số nguyên dương.')

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
env['LEGALQA_DEADLINE'] = str(time.time() + max(0, DEADLINE - time.monotonic()))
env['LEGALQA_MAX_ITEMS'] = '0'
if INSTALL_DEPS and not AUDIT_ONLY:
    run_bounded([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                 'numpy>=1.26,<3', 'nltk==3.9.1', 'absl-py==2.2.2', 'six==1.17.0'])
    if RUN_GPU:
        # Retain Kaggle CUDA torch. The model contract matches the Stage 3 environment.
        run_bounded([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                     'transformers==4.51.3', 'accelerate==1.6.0', 'peft==0.15.2',
                     'bitsandbytes==0.45.5', 'huggingface-hub==0.30.2',
                     'safetensors==0.5.3', 'sentencepiece==0.2.0'])
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

Ưu tiên ZIP có tên bắt đầu bằng `legalqa_main_stage3_v8_diagnostics`. Nếu không thấy ZIP, tìm `stage3_manifest.json` trong dataset đã giải nén. Khi có nhiều kết quả, đặt `DIAGNOSTICS` cụ thể; không tự chọn phiên mới nhất.
""")
    cell("code", "s4input", """
if DIAGNOSTICS is None:
    matches = sorted(INPUT.rglob('legalqa_main_stage3_v8_diagnostics*.zip'))
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
    cell("code", "s4gpu-inputs", """
import shutil
if PREVIOUS_OUTPUT is not None and not OUTPUT.exists():
    previous = Path(PREVIOUS_OUTPUT)
    if not (previous / 'identity.json').is_file():
        raise ValueError('PREVIOUS_OUTPUT phải là output Stage 4 V2 có identity.json.')
    shutil.copytree(previous, OUTPUT)
if RUN_GPU:
    if MODEL_ROOT is None:
        choices = sorted(p.parent for p in INPUT.rglob('models.lock.json')
                         if (p.parent / 'generator/config.json').is_file())
        if len(choices) != 1:
            raise RuntimeError(f'Đặt MODEL_ROOT cụ thể; tìm thấy {choices}.')
        MODEL_ROOT = choices[0]
    if ADAPTER_ROOT is None:
        choices = sorted(p.parent for p in INPUT.rglob('selected_adapter/adapter_config.json')
                         if (p.parent / 'adapter_model.safetensors').is_file())
        if len(choices) != 1:
            raise RuntimeError(f'Đặt ADAPTER_ROOT cụ thể; tìm thấy {choices}.')
        ADAPTER_ROOT = choices[0]
    print('Models:', MODEL_ROOT, 'Adapter:', ADAPTER_ROOT)
    print('GPU sẽ kiểm adapter hash và model lock trước khi load weights.')
""")
    cell("markdown", "s4run-note", """
## Sửa lặp, chấm dev100 và chọn bản xuất

Giữ nguyên các mục gần giống nhưng khác số liệu/phủ định. Câu chỉ còn dẫn nhập sau xóa lặp được giữ bản gốc và đưa vào danh sách cần xử lý tiếp. Không cắt mọi câu xuống một độ dài cố định; không phục hồi raw output đã bị guard của Stage 3 loại.

Chạy lại cùng input/code/cấu hình được phép. Khi đổi nguồn hoặc chính sách, chọn `OUTPUT` mới; không sửa/xóa identity để ép tái sử dụng kết quả cũ. Trong chế độ audit-only không xuất ZIP. Sau khi sửa lỗi môi trường, có thể chạy lại cell này với cùng identity.
""")
    cell("code", "s4run", """
command = [sys.executable, '-m', 'legalqa.repair_v2', '--diagnostics', DIAGNOSTICS, '--output', OUTPUT]
if AUDIT_ONLY:
    command.append('--audit-only')
if RUN_GPU:
    command.extend(['--gpu', '--models', MODEL_ROOT, '--adapter', ADAPTER_ROOT, '--max-items', GPU_MAX_ITEMS])
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
print('STATUS:', manifest['status'], 'SELECTED:', manifest.get('selected_variant'))
if manifest['status'] == 'paused':
    print('GPU chưa hoàn tất. ZIP hiện tại là CPU; lưu toàn bộ output để resume phiên sau.')
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
    if (OUTPUT / name).is_file():
        display(FileLink(str(OUTPUT / name)))
if (OUTPUT / 'gpu').is_dir():
    for name in ('candidates.json', 'decision.json', 'status.json'):
        if (OUTPUT / 'gpu' / name).is_file():
            display(FileLink(str(OUTPUT / 'gpu' / name)))
print('Các chỉ số ở đây là dev100, chưa phải điểm public của BTC.')
""")
    cell("markdown", "s4next", """
## Đọc danh sách còn cần xử lý

`repair.unresolved.json` ghi cờ cần xem lại của bản được chọn. Cờ chạm token chỉ yêu cầu kiểm tra đủ ý; không khẳng định đáp án sai. `gpu/candidates.json` ghi danh sách thử và các câu bị bỏ qua vì evidence yếu. Bộ lọc evidence là heuristic, không chứng minh retrieval đúng.

`gpu/*.checkpoint.jsonl` lưu từng lần sinh và lý do từ chối. Không khôi phục raw output bị guard từ chối. Câu không thuộc nhóm thử hoặc bản sinh mới bị lỗi được giữ nguyên từ CPU. Không có nhãn public hoặc prediction holdout để xác nhận độc lập; dev100 đã dùng chọn checkpoint nên có nguy cơ chọn cấu hình quá hợp tập.
""")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "version": "3.11.0"}}, "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "legalqa_main_04_repair_submit.ipynb"
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
