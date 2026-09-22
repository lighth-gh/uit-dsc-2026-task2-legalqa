"""Rebuild the dev100 repetition experiment with its portable helper."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.strip('\n').splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    stage3 = json.loads((ROOT / 'legalqa_main_03_generate_submit.ipynb').read_text(encoding='utf-8'))
    supervisor = next(''.join(c['source']) for c in stage3['cells']
                      if c['cell_type'] == 'code' and 'def bounded_process' in ''.join(c['source']))
    supervisor = supervisor[supervisor.index('class BudgetPause'):supervisor.index('if CODE.exists():')]
    cells = [
        cell('markdown', '''# Dev100 — so sánh repetition_penalty 1.0 / 1.03 / 1.05

Chạy đủ 100 câu cho mỗi mức, cùng QLoRA adapter đã chọn, cùng retrieval, prompt, seed và giới hạn token của Main 2/Main 3. Chỉ thay `repetition_penalty`; không train lại, chia lại dữ liệu hoặc retrieve lại. Đây là tuning trên dev đã dùng chọn adapter, không phải kiểm định độc lập.

**Add Input bắt buộc:**
1. **Output Main 02 hoàn tất** (`legalqa_main_02_select_retrieve.ipynb`), thư mục chứa `stage2_manifest.json` có `status=complete`. Lấy `config.json`, `session.json`, `models.lock.json`, `selection.json`, `selected_adapter/`, `data/dev100.questions.json`, `data/dev100.references.json`, `data/split_manifest.json`, `dev100.retrieval.json`. Gắn toàn bộ output để kiểm tra manifest.
2. **Dataset `lighth/ver3-smoke-output`**: model weights trong `legalqa_smoke_full_v1/models/`, cùng revision Main 2 đã dùng.

**Không cần output Main 01, Main 03, Main 04; không cần dataset train.** Main 2 đã chứa adapter và split dev100 cần dùng. Chọn **GPU T4 x2 + Internet**. Notebook tự pin code đúng commit Main 2; helper thử nghiệm được nhúng riêng ngoài checkout, không sửa hash/cache nguồn.

300 lượt sinh có thể cần nhiều giờ. Ngân sách tối đa 9 giờ/phiên, có checkpoint riêng từng mức. Nếu paused, Save Output; phiên sau gắn thêm output dev100 này, đặt `PREVIOUS_OUTPUT` tới thư mục `legalqa_dev100_repetition_v1`, đồng thời giữ hai input bắt buộc trên. Không đổi cấu hình giữa các phiên.
'''),
        cell('code', '''from pathlib import Path
import json, os, shutil, signal, subprocess, sys, time

SESSION_STARTED = time.monotonic()
if not Path('/kaggle').is_dir():
    raise RuntimeError('Notebook chỉ chạy trên Kaggle.')
WORK = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
VERSION3_ROOT = Path('/kaggle/input/datasets/lighth/ver3-smoke-output/legalqa_smoke_full_v1')
MAIN2_OUTPUT = None  # Tự tìm đúng một stage2_manifest.json; nếu nhiều version, điền ROOT cụ thể.
PREVIOUS_OUTPUT = None  # ROOT output dev100 này từ phiên paused trước, không phải output Main 3.
WORK_HOURS = 9.0
MAX_NEW_QUESTIONS_PER_VARIANT = 0  # 0 = không giới hạn số câu ngoài ngân sách giờ; 4 để smoke rồi resume.
REPO_URL = 'https://github.com/lighth-gh/uit-dsc-2026-task2-legalqa.git'
CODE = WORK / 'legalqa_dev100_code'
RUN_ROOT = WORK / 'legalqa_dev100_repetition_v1'
MODELS = WORK / 'dev100_runtime_models'
HELPER = WORK / 'dev100_repetition.py'
PENALTIES = (1.0, 1.03, 1.05)
if not 0 < WORK_HOURS <= 9:
    raise ValueError('WORK_HOURS phải trong (0, 9].')
if not isinstance(MAX_NEW_QUESTIONS_PER_VARIANT, int) or MAX_NEW_QUESTIONS_PER_VARIANT < 0:
    raise ValueError('MAX_NEW_QUESTIONS_PER_VARIANT phải là số nguyên >= 0.')
WORK_END = SESSION_STARTED + WORK_HOURS * 3600
'''),
        cell('markdown', '## Khóa code và kiểm tra input'),
        cell('code', supervisor + '''
if MAIN2_OUTPUT is None:
    matches = sorted(INPUT.rglob('stage2_manifest.json'))
    if len(matches) != 1:
        raise RuntimeError(f'Cần đúng một output Main 2; tìm thấy {matches}. Điền MAIN2_OUTPUT cụ thể.')
    MAIN2_OUTPUT = matches[0].parent
MAIN2_OUTPUT = Path(MAIN2_OUTPUT)
manifest = json.loads((MAIN2_OUTPUT / 'stage2_manifest.json').read_text(encoding='utf-8'))
if manifest.get('schema') != 2 or manifest.get('stage') != 2 or manifest.get('status') != 'complete':
    raise ValueError('Cần output Main 2 schema 2 đã complete.')
PIN = manifest['code_commit']
if len(PIN) != 40 or any(c not in '0123456789abcdef' for c in PIN):
    raise ValueError('Main 2 thiếu full commit SHA hợp lệ.')
if PREVIOUS_OUTPUT is not None:
    PREVIOUS_OUTPUT = Path(PREVIOUS_OUTPUT)
    if not (PREVIOUS_OUTPUT / 'experiment.identity.json').is_file():
        raise FileNotFoundError('PREVIOUS_OUTPUT phải là ROOT của dev100 repetition experiment.')
if CODE.exists():
    remote = subprocess.check_output(['git', '-C', str(CODE), 'remote', 'get-url', 'origin'], text=True, timeout=30).strip()
    dirty = subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True, timeout=30).strip()
    if remote.rstrip('/') != REPO_URL.rstrip('/') or dirty:
        raise RuntimeError('Checkout không đúng origin hoặc có sửa đổi. Dùng session mới.')
else:
    bounded_process(['git', 'clone', '--no-checkout', '--depth', '1', REPO_URL, CODE], seconds=300)
bounded_process(['git', '-C', CODE, 'fetch', '--depth', '1', 'origin', PIN], seconds=300)
bounded_process(['git', '-C', CODE, 'checkout', '--detach', 'FETCH_HEAD'], seconds=60)
commit = subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True, timeout=30).strip()
if commit != PIN:
    raise RuntimeError('Checkout không đúng commit Main 2.')
print('Pinned Main 2 commit:', commit)
saved_models = VERSION3_ROOT / 'models'
if not (saved_models / 'models.lock.json').is_file():
    raise FileNotFoundError(saved_models / 'models.lock.json')
MODELS.mkdir(exist_ok=True)
for role in ('embedding', 'reranker', 'generator'):
    source = saved_models / role
    if not (source / 'config.json').is_file() or not list(source.glob('*.safetensors')):
        raise FileNotFoundError(f'Thiếu model weights/config Version 3: {source}')
    target = MODELS / role
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f'Model link khác nguồn: {target}')
    else:
        target.symlink_to(source, target_is_directory=True)
shutil.copy2(saved_models / 'models.lock.json', MODELS / 'models.lock.json')
'''),
        cell('markdown', '''## Helper thử nghiệm và môi trường

Helper nằm ngoài code Main 2, chỉ tạo output thí nghiệm mới. Muốn cập nhật helper trong repo, chạy `python scripts/build_dev100_notebook.py` để đồng bộ notebook. METEOR dùng scorer BTC và WordNet; lỗi kiểm tra metric sẽ dừng phiên.
'''),
        cell('code', 'EXPERIMENT_HELPER = ' + repr((ROOT / 'scripts/dev100_repetition.py').read_text(encoding='utf-8')) +
             "\nHELPER.write_text(EXPERIMENT_HELPER, encoding='utf-8')\n"),
        cell('code', '''bounded_process([sys.executable, '-m', 'pip', 'install', '-q', '-r', CODE / 'requirements.txt'], seconds=1200)
bounded_process([sys.executable, '-m', 'nltk.downloader', '-q', 'wordnet', 'omw-1.4'], seconds=300)
bounded_process([sys.executable, 'scripts/check_metrics.py'], cwd=CODE, seconds=300)
prepare = [sys.executable, '-B', HELPER, 'prepare', '--upstream', MAIN2_OUTPUT,
           '--models', MODELS, '--output', RUN_ROOT]
if PREVIOUS_OUTPUT is not None:
    prepare += ['--previous', PREVIOUS_OUTPUT]
bounded_process(prepare, cwd=CODE, seconds=600)
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, timeout=60)
(RUN_ROOT / 'environment.freeze.txt').write_text(freeze, encoding='utf-8')
'''),
        cell('markdown', '''## Sinh 3 × 100 câu, chấm và so sánh

Mỗi mức sinh lại toàn bộ dev100 với cùng adapter, dùng hai GPU khi có; không lấy điểm baseline model gốc hoặc chỉ sinh lại câu lặp. `no_repeat_ngram_size`, prompt, top-k và `max_new_tokens` giữ nguyên Main 2. Log in cấu hình và số câu đã xong.

Khi hết ngân sách/cap, trạng thái `paused` là hợp lệ, chưa có kết luận chọn mức. Checkpoint riêng từng mức giữ câu đã xong. Diagnostics được đóng cả khi paused hoặc worker báo lỗi.
'''),
        cell('code', '''from zipfile import ZIP_DEFLATED, ZipFile

def run_variant(cfg, *args):
    env = {**os.environ, 'PYTHONUNBUFFERED': '1',
           'LEGALQA_MAX_ITEMS': str(MAX_NEW_QUESTIONS_PER_VARIANT),
           'LEGALQA_DEADLINE': str(time.time() + max(0, WORK_END - time.monotonic() - 300))}
    bounded_process([sys.executable, '-B', '-m', 'legalqa', '--config', cfg,
                     '--models', MODELS, *args], cwd=CODE, env=env)

status, failure = 'paused', None
try:
    complete = True
    for penalty in PENALTIES:
        variant = RUN_ROOT / f'rp_{penalty:.2f}'
        prediction = variant / 'predictions.json'
        print(f'DEV100 repetition_penalty={penalty}; all 100 questions; selected Main 2 adapter', flush=True)
        run_variant(variant / 'config.json', 'generate', '--multi-gpu',
                    '--questions', RUN_ROOT / 'data/dev100.questions.json',
                    '--retrieval', RUN_ROOT / 'dev100.retrieval.json',
                    '--adapter', RUN_ROOT / 'selected_adapter', '--output', prediction)
        if not all(path.is_file() for path in (prediction, prediction.with_suffix('.audit.json'),
                                               prediction.with_suffix('.manifest.json'))):
            complete = False
            print(f'Paused {variant.name}; chưa đủ 100 câu. Resume output này.', flush=True)
            break
        run_variant(variant / 'config.json', 'evaluate', '--predictions', prediction,
                    '--references', RUN_ROOT / 'data/dev100.references.json',
                    '--output', variant / 'metrics.json', '--label', variant.name)
    if complete:
        bounded_process([sys.executable, '-B', HELPER, 'summarize', '--output', RUN_ROOT], cwd=CODE, seconds=600)
        status = 'complete'
except BudgetPause as error:
    print('PAUSED:', error, flush=True)
except BaseException as error:
    status, failure = 'failed', error
finally:
    (RUN_ROOT / 'experiment.status.json').write_text(json.dumps(
        {'status': status, 'code_commit': PIN, 'penalties': PENALTIES,
         'error': str(failure) if failure else None}, ensure_ascii=False, indent=2), encoding='utf-8')
    archive_path = RUN_ROOT / 'dev100_repetition_diagnostics.zip'
    temporary = archive_path.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(RUN_ROOT.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv', '.txt'}:
                archive.write(path, arcname=path.relative_to(RUN_ROOT).as_posix())
    os.replace(temporary, archive_path)
    print('STATUS:', status, '| OUTPUT:', RUN_ROOT, '| DIAGNOSTICS:', archive_path, flush=True)
if failure is not None:
    raise failure
'''),
        cell('markdown', '''## Đọc kết quả

- `comparison.csv` / `comparison.json`: METEOR, ROUGE-L và delta so với 1.0; số câu lặp raw/final, tỷ lệ dòng lặp trung bình, số câu chạm token limit, fallback, độ dài và thời gian trung bình/câu (không phải tổng thời gian hai GPU).
- `per_question.csv`: 300 dòng để so sánh từng ID; `raw_quality_reasons` / `final_quality_reasons` và cờ riêng cho `inline_phrase_loop`, `long_numeric_run`, `heading_only`. Xem `rp_*/predictions.audit.json` để đọc `raw_answer` và `rp_*/predictions.json` để đọc đáp án cuối.
- `rp_1.03/paired_comparison.json`, `rp_1.05/paired_comparison.json`: paired bootstrap METEOR/ROUGE-L so với 1.0; không bảo đảm chất lượng trên private.
- Bộ đo v2 bổ sung: cụm 3–24 từ lặp liên tiếp trong một dòng ít nhất 4 lần và tổng ít nhất 24 từ; chuỗi ít nhất 20 số nguyên liên tiếp tăng 1; đáp án chỉ còn phần dẫn/tiêu đề/mục đánh số. Các ngưỡng được lưu tại `measurement_policy` trong `comparison.json`. Có các cột đếm riêng raw/final trong bảng tổng hợp; `heading_only` không cộng vào số câu lặp.
- Chỉ đề xuất mức mới khi **cả METEOR và ROUGE-L không giảm**, số câu raw bị lặp giảm, số câu final bị lặp, số đáp án raw/final chỉ còn tiêu đề và tỷ lệ dòng lặp trung bình không tăng. Trong các mức đạt, ưu tiên ít câu lặp raw/final hơn, rồi METEOR và ROUGE-L cao hơn. Nếu không mức nào đạt, giữ **1.0**, `improvement_found=false`.
- Bộ phát hiện lặp là heuristic, không đánh giá tính đúng pháp lý. Notebook không thay config Main 3 và không tạo submission private.

Helper v2 đổi hash so với bản cũ: không dùng `PREVIOUS_OUTPUT` của helper v1 để resume generation. Nếu đã có đủ ba bộ predictions/audit/metrics, có thể dùng helper mới chạy `summarize --output <bản sao thư mục kết quả>` trong môi trường code đã pin để đo lại, không cần sinh lại đáp án.

Đủ điều kiện hoàn tất khi log có `DEV100 REPETITION COMPARISON COMPLETE` và `STATUS: complete`, đủ 100 câu ở cả ba mức. Nếu `paused`, gắn toàn bộ output để resume; diagnostics ZIP không chứa adapter weights nên không thay thế output đầy đủ khi resume.
'''),
    ]
    old = json.loads((ROOT / 'legalqa_dev100_pipeline.ipynb').read_text(encoding='utf-8'))
    old['cells'] = cells
    (ROOT / 'legalqa_dev100_pipeline.ipynb').write_text(
        json.dumps(old, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    build()
