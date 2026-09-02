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
  --modes extractive,knn,hybrid \
  --official-metrics
```

Đánh giá retrieval theo từng tầng sau khi BM25 và Dense index đã sẵn sàng:

```bash
python -m legalqa_baseline evaluate-retrieval \
  --train data/train.json \
  --db artifacts/legalqa.sqlite \
  --dense-index artifacts/legalqa_dense \
  --output artifacts/retrieval_eval.json \
  --limit 100 \
  --ks 1,3,5 \
  --device cuda
```

Lệnh này báo chi tiết các chỉ số cho từng tầng (BM25, Dense, RRF, Reranker):
- **Recall@1, Recall@3, Recall@5** (và Hit@K)
- **MRR, MRR@1, MRR@3, MRR@5** (Mean Reciprocal Rank)
- **NDCG@1, NDCG@3, NDCG@5** (Normalized Discounted Cumulative Gain)
- **MAP@1, MAP@3, MAP@5** (Mean Average Precision)
- **Gold Chunk Recall@K & Precision@K**

Vì Train không có nhãn `context_id/chunk_id`, evaluator tạo pseudo-gold bằng cách truy xuất theo answer rồi đo độ phủ token và cụm 5-token trong chunk. Báo cáo luôn chứa `pseudo_gold_coverage`; không được trình bày các số này như recall trên nhãn do con người gán.

`validate` và lệnh `score` báo cáo đầy đủ các chỉ số tương đồng câu trả lời:
- **METEOR exact-token** (mô phỏng nhẹ không cần NLTK)
- **ROUGE-L** (mô phỏng tokenizer BTC `[a-z0-9]`)
- **Answer Token-F1, Precision, Recall** (multiset token similarity)
- **Exact Match (EM)**
- **BLEU-1, BLEU-2, BLEU-4** (với brevity penalty và smoothing)
- **Độ dài trung bình và Length Ratio** (tỷ lệ độ dài câu trả lời dự đoán / tham chiếu)

Cờ `--official-metrics` tính thêm `competition_meteor` và `competition_rougeL` bằng đúng hai thư viện trong scoring program BTC; cần cài `requirements-metrics.txt` và NLTK WordNet.

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

Kết quả cuối cùng của pipeline RAG được ghi vào `artifacts/submission.json`
sau khi cả BM25 và Dense index đã sẵn sàng.

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
  --output artifacts/submission.json \
  --mode rag \
  --bm25-top-k 50 \
  --dense-top-k 50 \
  --rrf-k 60 \
  --rrf-top-k 50 \
  --reranker-candidate-k 20 \
  --rerank-top-k 3 \
  --reranker-max-length 1024 \
  --device cuda
```

Nhánh trả lời cuối là hybrid theo dạng câu hỏi: câu hỏi về thủ tục, hồ sơ,
danh sách, biểu mẫu hoặc điều luật trả nguyên văn chunk tốt nhất cùng chunk
trước/sau trong cùng văn bản (`extractive_long`). Các câu còn lại dùng LLM
512 token (`generated_512`); nếu model từ chối, trả quá ngắn, có dấu hiệu bị
cắt hoặc chạm token limit thì tự động quay về raw context liền kề
(`extractive_fallback`). Phần overlap giữa các chunk được loại bỏ nhưng không
có giới hạn ký tự cứng. Với câu hỏi về Mẫu/Phụ lục/Điều, raw answer bắt đầu từ
tiêu đề khớp tốt nhất trong cụm ba chunk và giữ nguyên toàn bộ nội dung phía sau;
nếu không tìm thấy tiêu đề phù hợp thì dùng thứ tự best → previous → next.

Mỗi lần predict còn ghi `<output>.audit.jsonl` ngay sau từng ID. Log chứa route,
độ dài answer/context, token sinh, trạng thái token-limit/từ chối/cắt câu và
`retrieval_trace`. Trace lưu thứ hạng, điểm số, `document_id/context_id`, `chunk_no`,
tiêu đề của BM25 Top-50, Dense Top-50, RRF Top-50, reranker pool Top-20 và
reranker Top-3 để xác định chính xác tài liệu bị loại ở tầng nào.

Retrieval còn mở rộng có kiểm soát các alias có độ tin cậy cao: `điện 8` ↔
`điện VIII` và mức tiền `1,8 triệu` ↔ `1.800.000 đồng`. BM25 ưu tiên exact phrase
như `Quy hoạch điện VIII`/`mức lương cơ sở`, Dense dùng query đã mở rộng, còn
reranker và generator vẫn nhận nguyên câu hỏi gốc. Số Điều, khoản, nghị định và
quyết định không bị chuyển sang số La Mã ngoài ngữ cảnh tên Quy hoạch điện.
Sau RRF, pipeline cộng một boost nhỏ có trần cho các chunk khớp chính xác số
hiệu văn bản, mức tiền, năm, tên quy hoạch, tên biểu mẫu hoặc cụm pháp lý dài.
Audit giữ cả `rrf_score`, `legal_signal_boost`, `boosted_rrf_score` và hạng
trước/sau boost. Pool đưa vào reranker vẫn mặc định là Top-20; chỉ nên tăng lên
30 khi trace thực tế cho thấy tài liệu đúng thường nằm ở hạng RRF 21–30.

### Xây dựng Dense FAISS Vector Index:
```bash
python -m legalqa_baseline build-dense-index \
  --contexts data/selected-contexts.zip \
  --dense-index artifacts/legalqa_dense \
  --embedding-model AITeamVN/Vietnamese_Embedding_v2 \
  --batch-size 8 \
  --resume \
  --checkpoint-chunks 4096 \
  --device cuda
```

Dense schema 5 dùng đúng recipe của model: lấy CLS, chuẩn hóa L2 ở FP32, rồi tìm kiếm bằng dot product; đồng thời dùng corpus hash v2 có framing an toàn và không phụ thuộc thứ tự file. Index schema 4 trở xuống không còn tương thích; notebook tự build lại, còn CLI có thể build lại với `--force`. Muốn khóa tuyệt đối model Hub khi chạy ngoài notebook, truyền thêm `--embedding-revision <commit-sha>` cho cả lệnh build và predict/evaluate.

Khi có nhiều GPU CUDA, dense encoder tự động dùng DataParallel trên tất cả GPU hiện diện. `--batch-size` là batch tổng (không phải batch mỗi GPU); batch 8 là mức an toàn cho 2x Tesla T4. Nếu vẫn hết VRAM, encoder tự động retry với batch nhỏ hơn. T4 dùng FP16; BF16 chỉ được chọn trên GPU Ampere trở lên.

`--resume` lưu Dense embedding thành các part nguyên tử trong thư mục `*.dense-checkpoint`; nếu runtime dừng, chạy lại cùng lệnh để tiếp tục từ part cuối thay vì dựng lại từ đầu. Trong lúc `predict`, CLI giữ cả `submission_*.checkpoint.json` và một `submission_*.json` đọc được sau mỗi chu kỳ checkpoint.

### Chạy lâu trên Kaggle

Notebook `uit-dsc-2026-task2-legalqa.ipynb` ghi toàn bộ submission, checkpoint, BM25/Dense index và snapshot ba model vào `/kaggle/working`. Snapshot được khóa theo commit Hugging Face và bỏ các file ONNX không dùng để giảm dung lượng Output. Sau khi chạy, chọn **Save Version** để Kaggle lưu Output. Lần sau, chọn **Add Data** và thêm Output của version trước; notebook sẽ tự tìm lại model/index/checkpoint và resume. Các model ở đây là pretrained checkpoint được tải về, pipeline không có bước fine-tune nên không cần train lại.

Hoặc sử dụng script chạy RAG chuyên biệt:

```bash
pip install -r requirements-generator.txt
python run_rag_inference.py \
  --input data/public-official.json \
  --db artifacts/legalqa.sqlite \
  --dense-index artifacts/legalqa_dense \
  --output artifacts/submission.json \
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
- `--rrf-top-k`: 50 (giữ thứ hạng Hybrid Top 50 sau fusion để audit/prefilter).
- `--reranker-candidate-k`: 20 (chỉ cross-encode Top 20 sau RRF; Top 50 vẫn được giữ ở tầng fusion).
- `--rerank-top-k`: 3 (chọn đúng 3 Điều luật chính xác nhất đưa vào LLM).
- `--reranker-max-length`: 1024 (giảm mạnh attention cost so với 2304; phù hợp smoke/full inference trên T4).

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

Kết quả cho biết route (`extractive_long`/`generated_512`/`extractive_fallback`), độ tin cậy và `evidence` (top chunks, BM25 score, Dense score, RRF score, Reranker score) đã dùng để tạo câu trả lời.

## 7. Kiến trúc mô hình (< 4B tham số)

Theo danh sách mô hình BTC đã phê duyệt, hệ thống kết hợp:

```text
BM25 FTS5 (0B) + Vietnamese_Embedding_v2 (~0.6B) [Top 50 + Top 50]
  → Reciprocal Rank Fusion (rrf_k=60) [Top 50]
  → Vietnamese_Reranker (~0.6B) [Top 3]
  → Vi-Qwen2-1.5B-RAG (~2B theo Hugging Face) [Sinh đáp án]
```

Tổng theo số làm tròn trên model card khoảng `3.2B`, vẫn dưới `4B`. Trước khi nộp, cần ghi lại số chính xác bằng `sum(p.numel())` cho từng checkpoint/revision thực tế thay vì dựa vào tên model hoặc số làm tròn.
