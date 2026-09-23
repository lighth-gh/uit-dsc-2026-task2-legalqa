"""Build the standalone deadline notebook without changing Main 04 v1."""
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def build():
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as z:
        for path in sorted([*(ROOT / 'legalqa').glob('*.py'), ROOT / 'assets/approved_models.json']):
            info = ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    cells = []

    def cell(kind, name, source):
        item = dict(cell_type=kind, id=name, metadata={}, source=source.strip().splitlines(keepends=True))
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', 'intro', '''
# Main 04 ver 2 — sửa fallback/truncated trước hạn nộp

**Chạy ngay private, một cấu hình, một lần sinh/ID. Không dev, không ablation, không train/retrieval lại.**
Import notebook này vào Kaggle, chọn GPU T4/P100 và bật Internet nếu cần cài thư viện.
Chạy **Run All trong phiên interactive**, tải ZIP khi đủ thời gian; không cần chờ Save & Run All.

## Add Input: 4 nguồn bắt buộc
1. **Baseline `submission.zip` hoặc `submission.json` private 1.918 câu** (bản 0.5713).
2. **`legalqa_main_stage2_v8_diagnostics (5).zip`** đúng private, hoặc thư mục có `stage2_manifest.json`.
   ZIP đã chứa câu hỏi và context retrieval; không cần corpus/private-official.json riêng.
3. Dataset **`lighth/ver3-smoke-output`** có thư mục `models/` chứa `models.lock.json`, `generator/`
   và cấu hình model embedding/reranker. Không chạy embedding/reranker.
4. Output Stage 2/3 có **`selected_adapter/adapter_model.safetensors` + `adapter_config.json`** đúng epoch đã chọn.
   ZIP diagnostics không chứa trọng số adapter.

Để `None` nếu mỗi loại chỉ có một nguồn. Nếu notebook in nhiều đường dẫn, chép đúng đường dẫn vào cấu hình.
Không dùng Stage 3 public `(8)` paused 900/1.000 thay cho Stage 2 private.
`PRIVATE_DIAGNOSTICS` không bắt buộc; chỉ điền Stage 3 hoàn chỉnh đúng private nếu có.

## Output cần nộp
**`/kaggle/working/submission.zip`** — chỉ chứa `submission.json`, đủ toàn bộ ID baseline.
ZIP được tạo trước cài thư viện, cập nhật sau mỗi câu; câu chưa chạy/sửa không đạt giữ nguyên baseline.
`main04_v2_deadline/status.json` ghi attempted/changed/remaining; `attempts.jsonl` lưu từng lần sinh.
Hết giờ hoặc lỗi GPU vẫn giữ ZIP gần nhất. Có thể tải ZIP trong tab Files/Output khi cell còn chạy.
Không đảm bảo tăng điểm vì bỏ thử dev theo yêu cầu chạy sát hạn.
''')
    cell('code', 'config', '''
from pathlib import Path
import os, sys, time, subprocess, json
SESSION_STARTED = time.time()
WORK_MINUTES = 22  # Nếu còn ít thời gian: đặt số phút còn lại TRỪ 3 phút để tải/nộp.
DEADLINE = SESSION_STARTED + WORK_MINUTES * 60
INPUT, WORK = Path('/kaggle/input'), Path('/kaggle/working')
assert INPUT.is_dir() and WORK.is_dir(), 'Notebook cần chạy trên Kaggle.'
assert 0 < WORK_MINUTES <= 27
BASELINE_SUBMISSION = None
STAGE2_DIAGNOSTICS = None
MODEL_ROOT = None       # Thư mục models có generator/ + models.lock.json.
ADAPTER_ROOT = None     # Thư mục selected_adapter có adapter_model.safetensors.
PRIVATE_DIAGNOSTICS = None  # Không cần; thiếu audit thì nhận diện bằng heuristic.
INSTALL_DEPS = True
OUTPUT = WORK / 'main04_v2_deadline'
OUTPUT.mkdir(parents=True, exist_ok=True)
''')
    cell('code', 'bundle', 'BUNDLE_SHA256 = ' + repr(hashlib.sha256(payload).hexdigest())
         + '\nBUNDLE_B64 = ' + repr(base64.b64encode(payload).decode()))
    cells[-1]['metadata'] = {'jupyter': {'source_hidden': True}}
    cell('code', 'setup-and-run', '''
import base64, hashlib, io, zipfile
from pathlib import PurePosixPath
from IPython.display import display, FileLink

payload = base64.b64decode(BUNDLE_B64)
assert hashlib.sha256(payload).hexdigest() == BUNDLE_SHA256
CODE = WORK / ('main04_v2_code_' + BUNDLE_SHA256[:12])
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        part = PurePosixPath(name)
        assert not part.is_absolute() and '..' not in part.parts and ':' not in name and chr(92) not in name
        target = CODE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
sys.path.insert(0, str(CODE))
from legalqa.adaptive_inputs import resolve_adaptive_input, read_submission
from legalqa.deadline_repair import export
from legalqa.io import copy_file, digest, read_json, write_json

def publish():
    path = OUTPUT / 'submission.zip'
    if path.is_file():
        copy_file(path, WORK / 'submission.zip')

def unique_folder(explicit, pattern, valid, label):
    matches = [Path(explicit)] if explicit is not None else sorted({p.parent for p in INPUT.rglob(pattern) if valid(p.parent)})
    if len(matches) != 1 or not valid(matches[0]):
        raise ValueError(f'{label}: cần đúng một thư mục hợp lệ. Đặt đường dẫn cụ thể: {matches}')
    return matches[0]

def run_bounded(command, env=None):
    # Keep parent notebook alive to publish the latest ZIP even if child must be killed.
    remaining = DEADLINE - time.time() - 30
    if remaining <= 0:
        raise TimeoutError('Hết thời gian; tải submission.zip hiện tại.')
    process = subprocess.Popen(list(map(str, command)), cwd=CODE, env=env)
    try:
        while process.poll() is None:
            publish()
            if time.time() >= DEADLINE - 30:
                raise TimeoutError('Đã dừng theo deadline; giữ ZIP gần nhất.')
            time.sleep(1)
        if process.returncode:
            raise RuntimeError(f'Process exit={process.returncode}; xem log phía trên.')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        publish()

try:
    BASELINE_SUBMISSION = resolve_adaptive_input(INPUT, 'submission', BASELINE_SUBMISSION)
    baseline = read_submission(BASELINE_SUBMISSION)
    baseline_marker = OUTPUT / 'baseline_identity.json'
    identity = {'baseline': digest(baseline), 'bundle': BUNDLE_SHA256}
    if baseline_marker.exists():
        assert read_json(baseline_marker) == identity, 'Baseline/code đổi: chọn OUTPUT mới.'
    else:
        assert not (OUTPUT / 'attempts.jsonl').exists(), 'Journal thiếu baseline identity.'
        write_json(baseline_marker, identity)
        export(baseline, baseline, OUTPUT)
    publish()
    print('Đã có submission.zip baseline:', len(baseline), 'ID', flush=True)
    STAGE2_DIAGNOSTICS = resolve_adaptive_input(INPUT, 'stage2', STAGE2_DIAGNOSTICS)
    MODEL_ROOT = unique_folder(MODEL_ROOT, 'models.lock.json',
        lambda p: (p / 'models.lock.json').is_file() and (p / 'generator/config.json').is_file(), 'MODEL_ROOT')
    ADAPTER_ROOT = unique_folder(ADAPTER_ROOT, 'adapter_model.safetensors',
        lambda p: (p / 'adapter_model.safetensors').is_file() and (p / 'adapter_config.json').is_file(), 'ADAPTER_ROOT')
    print('Baseline:', BASELINE_SUBMISSION, '\\nStage2:', STAGE2_DIAGNOSTICS,
          '\\nModels:', MODEL_ROOT, '\\nAdapter:', ADAPTER_ROOT, flush=True)
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    if INSTALL_DEPS:
        run_bounded([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            'transformers==4.51.3', 'accelerate==1.6.0', 'peft==0.15.2',
            'bitsandbytes==0.45.5', 'huggingface-hub==0.30.2', 'safetensors==0.5.3', 'sentencepiece==0.2.0'], env)
    command = [sys.executable, '-m', 'legalqa.deadline_repair', '--stage2', STAGE2_DIAGNOSTICS,
        '--submission', BASELINE_SUBMISSION, '--models', MODEL_ROOT, '--adapter', ADAPTER_ROOT,
        '--output', OUTPUT, '--deadline', str(DEADLINE)]
    if PRIVATE_DIAGNOSTICS is not None:
        command += ['--private-diagnostics', PRIVATE_DIAGNOSTICS]
    run_bounded(command, env)
except (Exception, KeyboardInterrupt) as exc:
    print(f'Dừng: {type(exc).__name__}: {exc}', flush=True)
    write_json(OUTPUT / 'notebook_stop.json', {'error': f'{type(exc).__name__}: {exc}'})
finally:
    publish()
    if (OUTPUT / 'status.json').is_file():
        print(json.dumps(read_json(OUTPUT / 'status.json'), ensure_ascii=False, indent=2))
    if (WORK / 'submission.zip').is_file():
        with zipfile.ZipFile(WORK / 'submission.zip') as z:
            assert z.namelist() == ['submission.json'] and z.testzip() is None
            result = json.loads(z.read('submission.json'))
        print('ZIP sẵn sàng:', WORK / 'submission.zip', '—', len(result), 'ID')
        display(FileLink('submission.zip'))
    else:
        print('CHƯA CÓ ZIP: cần sửa đường dẫn BASELINE_SUBMISSION trước.')
''')
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.0'},
        'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True}}, nbformat=4, nbformat_minor=5)
    path = ROOT / 'legalqa_main_04_v2_deadline_submit.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    build()
