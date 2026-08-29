# UIT DSC 2026 – LegalQA baseline 0.1

Baseline này chạy hoàn toàn offline và không dùng mô hình học sâu:

1. Chia 8.532 văn bản theo `Điều`/`Phụ lục`, sau đó dùng cửa sổ từ cho đoạn quá dài.
2. Lập chỉ mục BM25 bằng SQLite FTS5 (`unicode61`, hỗ trợ tiếng Việt có dấu); mỗi truy vấn ưu tiên 8 token hiếm nhất trong corpus để giảm nhiễu và tăng tốc.
3. Truy xuất Top-K chunk và rerank nhẹ bằng độ phủ từ khóa câu hỏi.
4. Sinh câu trả lời bằng cách trích cửa sổ liên quan nhất.
5. Nếu một câu hỏi Public gần như trùng câu hỏi Train, chế độ `hybrid` dùng đáp án của hàng xóm gần nhất.

Đây là **baseline kiểm tra pipeline và định dạng submission**, chưa phải cấu hình cạnh tranh cuối cùng. Toàn bộ phần retrieval/generation không có tham số học, vì vậy không ảnh hưởng giới hạn tổng số tham số dưới 4B.

## 1. Dữ liệu đầu vào

Đặt bốn tệp BTC cung cấp ở một thư mục bất kỳ:

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

Core baseline chỉ dùng Python standard library. Các package trong `requirements-metrics.txt` chỉ cần khi chạy đúng scoring program của BTC.

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

# Chạy RAG Generator (Vi-Qwen2-1.5B-RAG):
python -m legalqa_baseline predict \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --output artifacts/submission_rag.json \
  --mode rag \
  --device cuda
```

Hoặc sử dụng script chạy RAG chuyên biệt:

```bash
pip install -r requirements-generator.txt
python run_rag_inference.py \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --output artifacts/submission_rag.json \
  --model-name AITeamVN/Vi-Qwen2-1.5B-RAG \
  --mode rag \
  --context-top-k 3 \
  --device cuda
```

Tham số đáng thử trước:

- `--max-answer-words`: 320, 420, 520, 620.
- `--top-k`: 8, 12, 20.
- `--knn-threshold`: 0.65–0.85; ngưỡng càng cao càng ít sao chép đáp án Train.
- `--context-top-k`: 3–5 (số đoạn luật liên quan nhất đưa vào prompt LLM).

## 6. Kiểm tra một câu hỏi

```bash
# Kiểm tra chế độ hybrid (extractive/knn)
python -m legalqa_baseline inspect \
  --db artifacts/legalqa.sqlite \
  --mode hybrid \
  --question "Vận chuyển động vật không có giấy chứng nhận kiểm dịch bị phạt thế nào?"

# Kiểm tra chế độ RAG Generator
python -m legalqa_baseline inspect \
  --db artifacts/legalqa.sqlite \
  --mode rag \
  --question "Vận chuyển động vật không có giấy chứng nhận kiểm dịch bị phạt thế nào?"
```

Kết quả cho biết route (`extractive`/`knn`/`rag`), độ tin cậy và `evidence` hoặc các ngữ cảnh pháp lý đã dùng để LLM sinh câu trả lời.

## 7. Giới hạn đã biết

- BM25 không hiểu đồng nghĩa/ngữ nghĩa sâu.
- Trích một chunk có thể thiếu phần mở đầu, kết luận hoặc căn cứ sửa đổi mà đáp án chuyên gia thêm vào.
- KNN chỉ an toàn khi câu hỏi rất gần nhau; dùng ngưỡng thấp sẽ chép nhầm điều luật.
- METEOR/ROUGE-L thưởng độ giống bề mặt, nên câu đúng nghĩa nhưng diễn đạt khác vẫn có điểm thấp.
- 20 văn bản corpus có `passage` rỗng; builder bỏ qua có chủ đích.

## 8. Hướng nâng cấp sau baseline

Theo danh sách mô hình BTC đã phê duyệt, cấu hình hợp lý tiếp theo là:

```text
BM25 Top 100
  → Vietnamese_Embedding_v2 hoặc multilingual-e5-base (dense retrieval)
  → Vietnamese_Reranker / Qwen3-Reranker-0.6B (Top 5)
  → Vi-Qwen2-1.5B-RAG hoặc Qwen2.5-1.5B-Instruct (sinh đáp án)
```

Không ghép tùy ý các model sát 4B: giới hạn của BTC tính **tổng tham số embedding + reranker + generator**, không phải từng model riêng lẻ. Baseline giữ interface retrieval/evidence/generation rõ ràng để thay từng tầng mà không đổi schema submission.

