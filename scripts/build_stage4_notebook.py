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
# LegalQA Main 04 — P0/P1/P2 ablation và repair

Notebook này đã nhúng sẵn toàn bộ code, scorer và cấu hình thử nghiệm. Chỉ chọn `MODE`; không sửa cell lệnh. Mặc định `p1_dev` tái lập baseline P0 rồi thử độc lập năm cấu hình inference. Các mode khác: `p1_public`, `p2_retrieval`, `p2_generate`, và `repair_v2` để giữ workflow Stage 4 cũ.

**Input bắt buộc:** đúng diagnostics Stage 3 hoàn chỉnh, output Stage 2/3 có `selected_adapter/adapter_model.safetensors`, và dataset Version 3 chứa `models/` cùng `index/`. Chọn GPU T4/P100 và bật Internet. Diagnostics không chứa trọng số adapter.

Mọi mode dùng cùng thư mục `OUTPUT`. Khi cần phiên tiếp theo, Add Input toàn bộ output version trước và đặt `PREVIOUS_OUTPUT` tới thư mục gốc đó. Notebook khóa SHA diagnostics, bundle code, model và adapter; không sửa metadata để ép resume. P1 public chỉ chạy khi đúng variant đã qua điều kiện dev. P2 chỉ tạo/chấm candidate dev100; không tự động nộp public.
""")
    cell("code", "s4config", """
from pathlib import Path
import os, sys, json, time, subprocess

SESSION_STARTED = time.monotonic()
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
if not INPUT.is_dir() or not WORK.is_dir():
    raise RuntimeError('Notebook này dùng đường dẫn Kaggle. Chạy local bằng python -m legalqa.repair.')

# Chọn đúng một mode. Lượt đầu được khuyến nghị: p1_dev.
MODE = 'p1_dev'  # p1_dev | p1_public | p2_retrieval | p2_generate | repair_v2

# None: tự tìm đúng một diagnostics ZIP, hoặc một thư mục Stage 3 đã giải nén.
DIAGNOSTICS = None
EXPECTED_DIAGNOSTICS_SHA256 = 'a19932405fe8ae65713d290c362a1ba2b968ee71d090ea4479dcdeeda018afe7'
OUTPUT = WORK / 'legalqa_main_04_v8_060'
RUN_GPU = True
MODEL_ROOT = Path('/kaggle/input/datasets/lighth/ver3-smoke-output/legalqa_smoke_full_v1/models')
ADAPTER_ROOT = None          # Thư mục selected_adapter chứa trọng số + adapter_config.json.
PREVIOUS_OUTPUT = None       # Output gốc legalqa_main_04_v8_060 của version trước.
P1_WINNER = None             # Bắt buộc với p1_public, ví dụ 'g1_penalty_103'.
P2_SHORTLIST = []            # p2_generate: tối đa 2 tên từ báo cáo p2_retrieval.
P1_VARIANTS = ['g1_penalty_103', 'g1_penalty_105', 'g2_contexts_2',
               'g3_complete_units', 'g4_grounded_prompt']
P2_VARIANTS = ['r1_pool_64', 'r2_intent_query', 'r3_adjacent_articles',
               'r4_lexical_weight_1', 'r5_scope_penalty']
GPU_MAX_ITEMS = 50           # Số câu mới mỗi variant/process trong phiên này.
INSTALL_DEPS = True          # Tắt nếu môi trường đã có scorer dependencies + WordNet.
AUDIT_ONLY = False           # Chỉ áp dụng cho mode repair_v2.
WORK_HOURS = 9.0             # Gồm cài đặt, CPU, GPU và chấm; không cam kết xong trong một phiên.
VALID_MODES = {'p1_dev', 'p1_public', 'p2_retrieval', 'p2_generate', 'repair_v2'}
if MODE not in VALID_MODES:
    raise ValueError(f'MODE không hợp lệ: {MODE}')
if not 0 < WORK_HOURS <= 9:
    raise ValueError('WORK_HOURS phải nằm trong (0, 9].')
DEADLINE = SESSION_STARTED + WORK_HOURS * 3600
if MODE != 'repair_v2' and not RUN_GPU:
    raise ValueError(f'{MODE} cần RUN_GPU=True.')
if MODE != 'repair_v2' and AUDIT_ONLY:
    raise ValueError('AUDIT_ONLY chỉ dùng với repair_v2.')
if RUN_GPU and AUDIT_ONLY:
    raise ValueError('RUN_GPU không dùng cùng AUDIT_ONLY.')
if not isinstance(GPU_MAX_ITEMS, int) or GPU_MAX_ITEMS <= 0:
    raise ValueError('GPU_MAX_ITEMS phải là số nguyên dương.')
if MODE == 'p1_public' and P1_WINNER is None:
    raise ValueError('p1_public yêu cầu P1_WINNER.')
if MODE == 'p2_generate' and not (1 <= len(P2_SHORTLIST) <= 2):
    raise ValueError('p2_generate yêu cầu P2_SHORTLIST có 1 hoặc 2 variant.')

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
    if MODE.startswith('p2'):
        run_bounded([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                     'faiss-cpu==1.10.0', 'ijson==3.4.0.post0'])
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
diagnostics_was_directory = DIAGNOSTICS.is_dir()
if diagnostics_was_directory:
    packed = WORK / 'stage4_input_diagnostics.zip'
    run_bounded([sys.executable, '-c',
        'import sys; from legalqa.repair import diagnostics_zip_from_directory; '
        'diagnostics_zip_from_directory(sys.argv[1], sys.argv[2])', DIAGNOSTICS, packed], cwd=CODE, env=env)
    DIAGNOSTICS = packed
diagnostics_sha256 = hashlib.sha256(DIAGNOSTICS.read_bytes()).hexdigest()
if (EXPECTED_DIAGNOSTICS_SHA256 and not diagnostics_was_directory
        and diagnostics_sha256 != EXPECTED_DIAGNOSTICS_SHA256):
    raise ValueError(f'Sai diagnostics SHA-256: {diagnostics_sha256}')

# P2 dùng trực tiếp questions/references/config trong diagnostics. Không tin đường dẫn ZIP.
EXTRACTED = WORK / 'stage4_diagnostics_extracted'
if not EXTRACTED.exists():
    with zipfile.ZipFile(DIAGNOSTICS) as archive:
        for info in archive.infolist():
            part = PurePosixPath(info.filename)
            if part.is_absolute() or '..' in part.parts or '\\\\' in info.filename or ':' in info.filename:
                raise ValueError(f'Đường dẫn diagnostics không hợp lệ: {info.filename}')
        archive.extractall(EXTRACTED)
print('Diagnostics:', DIAGNOSTICS)
print('Diagnostics SHA-256:', diagnostics_sha256)
print('Output:', OUTPUT)
""")
    cell("code", "s4gpu-inputs", """
import shutil
if PREVIOUS_OUTPUT is not None and not OUTPUT.exists():
    previous = Path(PREVIOUS_OUTPUT)
    if not (previous / 'main04_state.json').is_file():
        raise ValueError('PREVIOUS_OUTPUT phải là output Main 04 mới có main04_state.json.')
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
if MODE.startswith('p2'):
    INDEX_ROOT = MODEL_ROOT.parent / 'index'
    for required in ('index_manifest.json', 'corpus.sqlite', 'dense.faiss'):
        if not (INDEX_ROOT / required).is_file():
            raise FileNotFoundError(INDEX_ROOT / required)
    print('Index:', INDEX_ROOT)
""")
    cell("markdown", "s4run-note", """
## Chạy mode đã chọn

- `p1_dev`: tái lập P0 và thử năm cấu hình inference trên cùng nhóm câu được chọn bằng tín hiệu inference.
- `p1_public`: yêu cầu `P1_WINNER`; tự kiểm `decision.json`, resume tối đa `GPU_MAX_ITEMS` câu và đóng ZIP khi hoàn tất.
- `p2_retrieval`: tạo năm cache retrieval + diagnostic, chưa generation.
- `p2_generate`: yêu cầu `P2_SHORTLIST` tối đa hai variant; resume generation dev100, repair/chấm/paired comparison khi đủ.
- `repair_v2`: workflow Stage 4 V2 cũ.

Không dùng reference trong prompt hoặc chọn candidate theo từng ID. Mọi cache/journal giữ identity riêng. Khi trạng thái `paused`, Save output, Add Input version đó, đặt `PREVIOUS_OUTPUT`, giữ nguyên code/cấu hình và chạy lại.
""")
    cell("code", "s4run", """
sys.path.insert(0, str(CODE))
from legalqa.experiments import INFERENCE_VARIANTS, RETRIEVAL_VARIANTS
from legalqa.io import read_json, write_json
if P1_VARIANTS != list(INFERENCE_VARIANTS) or P2_VARIANTS != list(RETRIEVAL_VARIANTS):
    raise ValueError('Danh sách variant trong notebook khác code bundle.')

RUN_SUCCEEDED = False
OUTPUT.mkdir(parents=True, exist_ok=True)
STATE_PATH = OUTPUT / 'main04_state.json'
state_identity = {'diagnostics_sha256': diagnostics_sha256, 'bundle_sha256': BUNDLE_SHA256}
if STATE_PATH.is_file():
    state = read_json(STATE_PATH)
    if state.get('identity') != state_identity:
        raise ValueError('PREVIOUS_OUTPUT khác diagnostics/code; dùng output mới.')
else:
    state = {'identity': state_identity, 'runs': {}}

def record(status, **details):
    state['runs'][MODE] = {'status': status, **details}
    state['last_mode'] = MODE
    write_json(STATE_PATH, state)

BASELINE = OUTPUT / 'baseline'
EXPECTED_BASELINE_METEOR = 0.6202453105154175

def ensure_baseline():
    manifest = BASELINE / 'repair.manifest.json'
    if not manifest.is_file() or read_json(manifest).get('status') != 'complete':
        run_bounded([sys.executable, '-m', 'legalqa.repair_v2',
                     '--diagnostics', DIAGNOSTICS, '--output', BASELINE], cwd=CODE, env=env)
    metrics = read_json(BASELINE / 'dev.selected.metrics.json')
    if abs(metrics['meteor'] - EXPECTED_BASELINE_METEOR) > 1e-10:
        raise ValueError(f'Không tái lập đúng P0: {metrics["meteor"]}')
    return metrics

CONFIG_ROOT = OUTPUT / 'configs'
def ensure_configs():
    manifest = CONFIG_ROOT / 'manifest.json'
    if not manifest.is_file():
        run_bounded([sys.executable, '-m', 'legalqa.experiments', 'write-configs',
                     '--base', EXTRACTED / 'config.json', '--output', CONFIG_ROOT], cwd=CODE, env=env)
    return manifest

if MODE == 'p1_dev':
    baseline_metrics = ensure_baseline()
    root = OUTPUT / 'p1'
    summary = {}
    for variant in P1_VARIANTS:
        target = root / f'{variant}_dev'
        run_bounded([sys.executable, '-m', 'legalqa.experiments', 'inference',
                     '--diagnostics', DIAGNOSTICS, '--baseline', BASELINE,
                     '--models', MODEL_ROOT, '--adapter', ADAPTER_ROOT,
                     '--variant', variant, '--output', target, '--split', 'dev',
                     '--max-items', GPU_MAX_ITEMS], cwd=CODE, env=env)
        status = read_json(target / 'status.json')
        summary[variant] = status
        decision = target / 'decision.json'
        metrics = target / 'dev.candidate.metrics.json'
        if decision.is_file():
            summary[variant]['decision'] = read_json(decision)
        if metrics.is_file():
            score = read_json(metrics)
            summary[variant]['metrics'] = {key: score[key] for key in ('meteor', 'rougeL')}
        if status.get('status') == 'paused':
            break
    write_json(root / 'p1_summary.json', {'baseline': {k: baseline_metrics[k] for k in ('meteor','rougeL')},
                                          'variants': summary})
    complete = len(summary) == len(P1_VARIANTS) and all(
        row.get('status') == 'complete' for row in summary.values())
    record('complete' if complete else 'paused', summary='p1/p1_summary.json')

elif MODE == 'p1_public':
    ensure_baseline()
    root = OUTPUT / 'p1'
    dev_result = root / f'{P1_WINNER}_dev'
    decision = read_json(dev_result / 'decision.json')
    if not decision.get('passes_screen'):
        raise ValueError(f'{P1_WINNER} không qua điều kiện dev; không chạy public.')
    target = root / f'{P1_WINNER}_public'
    run_bounded([sys.executable, '-m', 'legalqa.experiments', 'inference',
                 '--diagnostics', DIAGNOSTICS, '--baseline', BASELINE,
                 '--models', MODEL_ROOT, '--adapter', ADAPTER_ROOT,
                 '--variant', P1_WINNER, '--dev-result', dev_result,
                 '--output', target, '--split', 'public',
                 '--max-items', GPU_MAX_ITEMS], cwd=CODE, env=env)
    status = read_json(target / 'status.json')
    zip_path = None
    if status.get('status') == 'complete':
        zip_path = OUTPUT / f'submission_{P1_WINNER}.zip'
        run_bounded([sys.executable, '-m', 'legalqa', '--config', EXTRACTED / 'config.json',
                     'package', '--predictions', target / 'public.candidate.json',
                     '--questions', EXTRACTED / 'data/test.questions.json',
                     '--output', zip_path], cwd=CODE, env=env)
    record(status.get('status', 'paused'), winner=P1_WINNER,
           submission_zip=zip_path.name if zip_path else None)

elif MODE == 'p2_retrieval':
    ensure_configs()
    root = OUTPUT / 'p2'
    summary = {}
    for variant in P2_VARIANTS:
        cfg = CONFIG_ROOT / 'retrieval' / f'{variant}.json'
        target = root / variant
        retrieval = target / 'dev100.retrieval.json'
        diagnostic = target / 'retrieval.diagnostic.json'
        run_bounded([sys.executable, '-m', 'legalqa', '--config', cfg, '--models', MODEL_ROOT,
                     'retrieve', '--questions', EXTRACTED / 'data/dev100.questions.json',
                     '--index', INDEX_ROOT, '--output', retrieval], cwd=CODE, env=env)
        if not retrieval.is_file():
            summary[variant] = {'status': 'paused'}
            break
        run_bounded([sys.executable, '-m', 'legalqa', '--config', cfg,
                     'diagnose-retrieval', '--qa', EXTRACTED / 'data/dev100.json',
                     '--retrieval', retrieval, '--index', INDEX_ROOT,
                     '--output', diagnostic], cwd=CODE, env=env)
        report = read_json(diagnostic)
        summary[variant] = {'status': 'complete', 'values': report['values']}
    write_json(root / 'retrieval_summary.json', summary)
    complete = len(summary) == len(P2_VARIANTS) and all(
        row.get('status') == 'complete' for row in summary.values())
    record('complete' if complete else 'paused', summary='p2/retrieval_summary.json')

elif MODE == 'p2_generate':
    ensure_configs()
    baseline_metrics = ensure_baseline()
    unknown = sorted(set(P2_SHORTLIST) - set(P2_VARIANTS))
    if unknown:
        raise ValueError(f'P2_SHORTLIST không hợp lệ: {unknown}')
    root = OUTPUT / 'p2'
    summary = {}
    generation_env = {**env, 'LEGALQA_MAX_ITEMS': str(GPU_MAX_ITEMS)}
    for variant in P2_SHORTLIST:
        cfg = CONFIG_ROOT / 'retrieval' / f'{variant}.json'
        target = root / variant
        retrieval = target / 'dev100.retrieval.json'
        if not retrieval.is_file():
            raise FileNotFoundError(f'Chạy p2_retrieval trước: {retrieval}')
        raw = target / 'dev.raw.json'
        run_bounded([sys.executable, '-m', 'legalqa', '--config', cfg, '--models', MODEL_ROOT,
                     'generate', '--questions', EXTRACTED / 'data/dev100.questions.json',
                     '--retrieval', retrieval, '--adapter', ADAPTER_ROOT,
                     '--output', raw], cwd=CODE, env=generation_env)
        if not raw.is_file():
            partial = raw.with_suffix('.partial.json')
            summary[variant] = {'status': 'paused',
                                'generated': len(read_json(partial)) if partial.is_file() else 0}
            continue
        repaired = target / 'dev.repaired.json'
        run_bounded([sys.executable, '-m', 'legalqa.experiments', 'postprocess',
                     '--predictions', raw, '--audit', raw.with_suffix('.audit.json'),
                     '--output', repaired], cwd=CODE, env=env)
        base_report = target / 'baseline.metrics.json'
        candidate_report = target / 'dev.metrics.json'
        paired = target / 'paired.json'
        run_bounded([sys.executable, '-m', 'legalqa', '--config', cfg, 'evaluate',
                     '--predictions', BASELINE / 'dev.selected.json',
                     '--references', EXTRACTED / 'data/dev100.references.json',
                     '--output', base_report, '--label', 'baseline_repaired'], cwd=CODE, env=env)
        run_bounded([sys.executable, '-m', 'legalqa', '--config', cfg, 'evaluate',
                     '--predictions', repaired,
                     '--references', EXTRACTED / 'data/dev100.references.json',
                     '--output', candidate_report, '--label', variant], cwd=CODE, env=env)
        run_bounded([sys.executable, '-m', 'legalqa', '--config', cfg, 'compare',
                     '--baseline', base_report, '--candidate', candidate_report,
                     '--output', paired], cwd=CODE, env=env)
        scores = read_json(candidate_report)
        summary[variant] = {'status': 'complete', 'meteor': scores['meteor'],
                            'rougeL': scores['rougeL'], 'paired': read_json(paired)}
    write_json(root / 'generation_summary.json',
               {'baseline': {k: baseline_metrics[k] for k in ('meteor','rougeL')},
                'variants': summary})
    complete = len(summary) == len(P2_SHORTLIST) and all(
        row.get('status') == 'complete' for row in summary.values())
    record('complete' if complete else 'paused', summary='p2/generation_summary.json')

else:  # repair_v2 compatibility mode
    target = OUTPUT / 'repair_v2'
    command = [sys.executable, '-m', 'legalqa.repair_v2',
               '--diagnostics', DIAGNOSTICS, '--output', target]
    if AUDIT_ONLY:
        command.append('--audit-only')
    if RUN_GPU:
        command.extend(['--gpu', '--models', MODEL_ROOT, '--adapter', ADAPTER_ROOT,
                        '--max-items', GPU_MAX_ITEMS])
    run_bounded(command, cwd=CODE, env=env)
    manifest = read_json(target / 'repair.manifest.json')
    record(manifest.get('status', 'paused'), output='repair_v2',
           submission_zip=manifest.get('submission_zip'))

RUN_SUCCEEDED = True
""")
    cell("code", "s4results", """
if not globals().get('RUN_SUCCEEDED', False):
    raise RuntimeError('Chưa có lần chạy Stage 4 thành công trong phiên này.')
from IPython.display import display, FileLink
state = read_json(OUTPUT / 'main04_state.json')
current = state['runs'][MODE]
print('MODE:', MODE, '| STATUS:', current['status'])
print(json.dumps(current, ensure_ascii=False, indent=2))

links = [OUTPUT / 'main04_state.json']
if MODE == 'p1_dev':
    links.append(OUTPUT / 'p1/p1_summary.json')
elif MODE == 'p1_public':
    links += [OUTPUT / f'p1/{P1_WINNER}_public/status.json']
    if current.get('submission_zip'):
        links.append(OUTPUT / current['submission_zip'])
elif MODE == 'p2_retrieval':
    links.append(OUTPUT / 'p2/retrieval_summary.json')
elif MODE == 'p2_generate':
    links.append(OUTPUT / 'p2/generation_summary.json')
else:
    links += [OUTPUT / 'repair_v2/repair.metrics.json',
              OUTPUT / 'repair_v2/repair.manifest.json']
    if current.get('submission_zip'):
        links.append(OUTPUT / 'repair_v2' / current['submission_zip'])

for path in links:
    if path.is_file():
        display(FileLink(str(path)))
if current['status'] == 'paused':
    print('Save toàn bộ output, Add Input version này, đặt PREVIOUS_OUTPUT rồi chạy lại cùng MODE/config.')
print('Điểm P1/P2 hiện tại là dev100; chưa phải bằng chứng public >= 0.60.')
""")
    cell("markdown", "s4next", """
## Bước tiếp theo

Sau `p1_dev`, xem `p1/p1_summary.json`; chỉ điền `P1_WINNER` và chuyển sang `p1_public` khi `passes_screen=true`. Sau `p2_retrieval`, gửi `p2/retrieval_summary.json` để chọn tối đa hai tên cho `P2_SHORTLIST`; sau đó dùng `p2_generate`. Nếu một mode paused, không đổi mode/variant giữa chừng.

`answer-token coverage` của P2 chỉ là diagnostic, không phải gold recall. Dev100 đã dùng chọn checkpoint nên ứng viên tốt vẫn phải xác nhận trên dev600 trước khi chạy public1000. Không dùng reference hoặc ngưỡng riêng theo ID public.
""")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "version": "3.11.0"}}, "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "legalqa_main_04_repair_submit.ipynb"
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
