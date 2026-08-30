# UIT DSC 2026 – LegalQA retrieval/RAG pipeline

Project có hai nhóm chế độ:

1. `extractive`, `knn`, `hybrid`: baseline lexical cũ, không dùng mô hình học sâu.
2. `rag`, `hybrid_rag`: BM25 + `Vietnamese_Embedding_v2` → RRF Top 50 → `Vietnamese_Reranker` Top 3 → `Vi-Qwen2-1.5B-RAG`.

Corpus được chia theo `Điều`/`Phụ lục`, sau đó dùng cửa sổ từ cho đoạn quá dài. BM25 dùng SQLite FTS5; Dense index dùng FAISS `IndexFlatIP` với dot product. Cấu hình RAG không tự ý bỏ qua lỗi Dense/Reranker; muốn fallback BM25 phải truyền rõ `--allow-retrieval-fallback`.

## 1. Dữ liệu đầu vào

Đặt ba tệp BTC cung cấp ở một thư mục bất kỳ:

```text
data/
├── train.json
├── public-official.json
└── selected-contexts.zip
```

Không cần giải nén `selected-contexts.zip`.

Schema đã được pipeline kiểm tra:

- Train: JSON object gồm 7.000 ID, mỗi giá trị có `question` và `answer`.
- Public: JSON object gồm 1.000 ID, `answer` là `null`.
- Corpus: 8.532 tệp `context_*.json`, các trường `id`, `name`, `link`, `passage`.
- Submission: `{ "id": { "answer": "..." } }` với đúng 1.000 ID Public.

## 2. Yêu cầu môi trường

- Python 3.10 trở lên.
- Python phải được build với SQLite FTS5. Có thể kiểm tra bằng lệnh:

```bash
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('create virtual table x using fts5(t)'); print('FTS5 OK')"
```

Các mode lexical chỉ dùng Python standard library. Mode RAG cần `requirements-generator.txt`; `requirements-metrics.txt` chỉ cần khi chạy đúng scoring program của BTC.

## 3. Xây index

Chạy tại thư mục chứa README này:

```bash
python -m legalqa_baseline build-index \
  --contexts data/selected-contexts.zip \
  --train data/train.json \
  --db artifacts/legalqa.sqlite
```

Cấu hình mặc định: chunk 620 từ, overlap 100 từ. Index khá lớn vì corpus giải nén gần 489 MB; không đưa file SQLite vào Git/ZIP. Có thể dựng lại hoàn toàn từ dữ liệu BTC.

Muốn xây lại index:

```bash
python -m legalqa_baseline build-index \
  --contexts data/selected-contexts.zip \
  --train data/train.json \
  --db artifacts/legalqa.sqlite \
  --force
```

## 4. Validation cục bộ

Chạy leave-one-out trên 300 mẫu Train cố định:

```bash
python -m legalqa_baseline validate \
  --train data/train.json \
  --db artifacts/legalqa.sqlite \
  --output artifacts/validation.json \
  --limit 300 \
  --modes extractive,knn,hybrid
```

`validate` mặc định dùng METEOR exact-token gần đúng và ROUGE-L để không phải tải dữ liệu NLTK. Phần ROUGE-L mô phỏng cả tokenizer ASCII-only trong mã chấm BTC. Điểm dùng để so sánh nhanh giữa các cấu hình, không nên ghi vào báo cáo như điểm chính thức.

Để chạy công thức giống file chấm BTC:

```bash
pip install -r requirements-metrics.txt
python score_official.py \
  --reference data/train.json \
  --prediction artifacts/prediction_on_labeled_split.json \
  --download-nltk
```

Lưu ý: `score_official.py` chỉ chấm được khi prediction và reference có cùng tập ID. Chương trình chấm BTC không word-segment tiếng Việt; nó gọi trực tiếp `str.split()`.

Một chi tiết cần biết: bản `rouge_score/tokenize.py` BTC gửi dùng regex `[a-z0-9]`, nên ROUGE-L loại/xé nhiều chữ cái tiếng Việt có dấu. METEOR không gặp lỗi này và cũng là metric chính; vì vậy baseline ưu tiên tối ưu METEOR.

## 5. Sinh submission Public

Tệp `submission.json` đang được giữ lại như artifact của baseline lexical cũ;
không xem đó là kết quả của pipeline RAG 0.2. Submission RAG phải được sinh lại
thành `artifacts/submission_rag.json` sau khi cả BM25 và Dense index đã sẵn sàng.

Khuyến nghị chạy cả ba mode và nộp thử để biết hướng nào hợp leaderboard:

```bash
python -m legalqa_baseline predict \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --output artifacts/submission_extractive.json \
  --mode extractive

python -m legalqa_baseline predict \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --output artifacts/submission_knn.json \
  --mode knn

python -m legalqa_baseline predict \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --output artifacts/submission_hybrid.json \
  --mode hybrid

# Chạy RAG Generator (Vi-Qwen2-1.5B-RAG + BM25 + Dense FAISS + RRF + Vietnamese_Reranker):
python -m legalqa_baseline predict \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --dense-index artifacts/legalqa_dense \
  --output artifacts/submission_rag.json \
  --mode rag \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --rrf-k 60 \
  --rrf-top-k 50 \
  --rerank-top-k 3 \
  --device cuda
```

### Xây dựng Dense FAISS Vector Index:
```bash
python -m legalqa_baseline build-dense-index \
  --contexts data/selected-contexts.zip \
  --dense-index artifacts/legalqa_dense \
  --embedding-model AITeamVN/Vietnamese_Embedding_v2 \
  --device cuda
```

Dense index của bản 0.1 dùng mean-pooling/cosine không còn tương thích. Encoder hiện dùng đúng CLS-pooling và dot product theo model card; nếu CLI báo schema cũ, hãy build lại với `--force`.

Hoặc sử dụng script chạy RAG chuyên biệt:

```bash
pip install -r requirements-generator.txt
python run_rag_inference.py \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --dense-index artifacts/legalqa_dense \
  --output artifacts/submission_rag.json \
  --model-name AITeamVN/Vi-Qwen2-1.5B-RAG \
  --embedding-model AITeamVN/Vietnamese_Embedding_v2 \
  --reranker-model AITeamVN/Vietnamese_Reranker \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --rrf-k 60 \
  --rrf-top-k 50 \
  --rerank-top-k 3 \
  --mode rag \
  --device cuda
```

Tham số đáng thử:

- `--bm25-top-k` & `--dense-top-k`: 50 (lấy 50 ứng viên từ mỗi nhánh).
- `--rrf-k`: 60 (tham số Reciprocal Rank Fusion tiêu chuẩn).
- `--rrf-top-k`: 50 (giữ Hybrid Top 50 để reranker chấm).
- `--rerank-top-k`: 3 (chọn đúng 3 Điều luật chính xác nhất đưa vào LLM).

## 6. Kiểm tra một câu hỏi

```bash
# Kiểm tra chế độ hybrid (extractive/knn)
python -m legalqa_baseline inspect \
  --db artifacts/legalqa.sqlite \
  --mode hybrid \
  --question "Vận chuyển động vật không có giấy chứng nhận kiểm dịch bị phạt thế nào?"

# Kiểm tra chế độ Hybrid RAG Generator
python -m legalqa_baseline inspect \
  --db artifacts/legalqa.sqlite \
  --dense-index artifacts/legalqa_dense \
  --mode rag \
  --question "Vận chuyển động vật không có giấy chứng nhận kiểm dịch bị phạt thế nào?"
```

Kết quả cho biết route (`extractive`/`knn`/`rag`), độ tin cậy và `evidence` (top chunks, BM25 score, Dense score, RRF score, Reranker score) đã dùng để LLM sinh câu trả lời.

## 7. Kiến trúc mô hình (< 4B tham số)

Theo danh sách mô hình BTC đã phê duyệt, hệ thống kết hợp:

```text
BM25 FTS5 (0B) + Vietnamese_Embedding_v2 (~0.6B) [Top 50 + Top 50]
  → Reciprocal Rank Fusion (rrf_k=60) [Top 50]
  → Vietnamese_Reranker (~0.6B) [Top 3]
  → Vi-Qwen2-1.5B-RAG (~2B theo Hugging Face) [Sinh đáp án]
```

Tổng theo số làm tròn trên model card khoảng `3.2B`, vẫn dưới `4B`. Trước khi nộp, cần ghi lại số chính xác bằng `sum(p.numel())` cho từng checkpoint/revision thực tế thay vì dựa vào tên model hoặc số làm tròn.
