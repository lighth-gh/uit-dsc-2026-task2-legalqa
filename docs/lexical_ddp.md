# Stage 1: BM25 cache và QLoRA trên hai T4

Log/diagnostics được cung cấp ghi 768 câu lexical, trung bình 29,03795 giây/câu,
tổng 22.301 giây (~6 giờ 12 phút). Training cũ ép `CUDA_VISIBLE_DEVICES=0`.

## Thay đổi

- BM25 nhánh OR cache điểm của từng từ, cộng bằng NumPy float64 và lấy top-k
  bằng partial selection. Giữ token/stopword, trọng số header/text 2.5/1.0,
  top-k, xử lý đồng điểm theo ID và RRF như cũ.
  Điểm BM25 FTS5 là tổng đóng góp từng từ với IDF và chuẩn hóa độ dài toàn corpus:
  [công thức SQLite FTS5](https://www.sqlite.org/fts5.html#the_bm25_function).
- `retrieval.bm25_cache_mb=1024`: tối đa 1 GiB cho các postings được giữ trong
  LRU cache. Mảng điểm truy vấn và postings đang đọc cần thêm RAM. Cache ở RAM,
  không sửa SQLite nguồn; phiên mới phải làm ấm lại. Đặt 0 để dùng SQL cũ.
- Lexical không nạp FAISS vào RAM; vẫn kiểm tra hash và identity của index.
  Log mỗi 10 câu có thời gian riêng cho BM25, phrase và precise.
- Stage 1 gọi `torch.distributed.run --standalone --nproc_per_node=2`.
  Mỗi worker nạp model NF4 lên `cuda:LOCAL_RANK`; Trainer dùng DDP, FP16,
  gradient checkpointing. Chỉ rank 0 ghi audit, report, adapter và bản sao epoch.
  Mọi rank đồng bộ quyết định dừng để lưu/resume checkpoint.
- `training.gradient_accumulation=8` là số lần tích lũy quy đổi về một GPU:
  chia cho world size khi tạo Trainer. Batch hiệu dụng vẫn là
  `1 × 4 × 2 = 8`; không đổi epoch, số QA hay độ dài tối đa.

## Chạy notebook mới

1. Đưa bản sửa lên repo mà notebook clone rồi mở
   `legalqa_main_01_qlora_train.ipynb`, chọn **GPU T4 x2**.
2. Bắt đầu experiment mới, không gắn output Stage 1 cũ trong Input. Để
   `PREVIOUS_OUTPUT=None`, `LEGACY_INPUT_ROOT=None`. Lưu ý `None` vẫn tự dò
   output trong Input: nếu gắn output cũ, notebook sẽ khóa về commit cũ.
   Retrieval cache/checkpoint cũ không cùng fingerprint với code/config mới.
3. Chạy các cell. Cell đầu kiểm tra có hai T4. Khi train, log phải có cả
   `rank=0/2 ... device=cuda:0` và `rank=1/2 ... device=cuda:1`, accumulation=4,
   effective_batch=8. Thiếu worker/GPU sẽ báo lỗi, không âm thầm train một GPU.
4. Sau khi train hoặc pause hợp lệ, kiểm tra
   `sft/distributed_training.json`: world_size=2, hai worker có cùng global_step
   > 0, device khác nhau và peak_allocated_bytes > 0. Code còn kiểm tra model
   thực sự được bọc DDP. File này được đưa vào diagnostics.
5. Chỉ resume bằng output của chính experiment mới này; notebook giữ nguyên
   cơ chế khóa commit và phục hồi optimizer/RNG của checkpoint.

## Benchmark trên index đầy đủ

Sau khi Stage 1 đã chuẩn bị dữ liệu, chạy cell sau trong notebook. Các biến
CODE, VERSION3_ROOT và RUN_ROOT do notebook định nghĩa; kiểm tra file trước khi chạy.

```python
database = VERSION3_ROOT / 'index' / 'corpus.sqlite'
questions = RUN_ROOT / 'data' / 'train.sft.questions.json'
assert database.is_file() and questions.is_file()
bounded_process([
    sys.executable, CODE / 'scripts' / 'benchmark_lexical.py',
    '--database', database, '--questions', questions,
    '--config', RUN_ROOT / 'config.json', '--limit', '100',
    '--output', WORK / 'lexical_benchmark.json',
], cwd=CODE)
```

Benchmark mặc định so bản đã cache BM25 nhưng phrase/precise còn dùng SQL
(`--baseline bm25-cache`) với bản tối ưu cả ba nhánh. Thêm `--baseline sql` để
so với bản chưa cache BM25. Báo cáo có tổng thời gian/speedup từng nhánh và kiểm tra
thứ hạng giống nhau cho từng câu. Tính cả thời gian làm ấm cache;
câu đầu có thể chậm hơn. Benchmark chạy độc lập, không tạo retrieval cache cho train.

Kiểm tra cục bộ đã dùng 100 câu trong diagnostics, dựng 27.343 đoạn bằng cửa sổ
180 từ, stride 150 từ từ các parent đã retrieve. Nhánh BM25: cũ trung bình
0,05469 giây, mới 0,02932 giây (~1,87x); 50 câu sau ~3,20x, top-100 trùng
100/100 câu. Đây là fixture thu nhỏ trong RAM, **không phải** phép đo trên
index Kaggle 407.107 chunks, không suy ra trực tiếp thời gian mới từ 29 giây/câu.
Chưa chạy QLoRA thực tế trên hai T4 trong môi trường sửa code này.

## Cập nhật riêng phrase và precise theo log 270/5.600 câu

28 mốc log riêng biệt (đã bỏ dòng lặp) có phrase trung bình 5,638 giây và precise
3,786 giây. Đây là mẫu các câu được ghi log, không phải trung bình cả 270 câu.

- `fast_phrase_precise=true` bật cách mới; false quay lại hai nhánh SQL cũ.
  Hàm BM25 rộng và cấu hình QLoRA không đổi trong cập nhật này.
- Precise lấy giao toàn bộ ID trong các postings từng từ đã được BM25 cache,
  cộng điểm theo thứ tự term ban đầu. Giữ các mức AND 12/8/5 term, thứ tự
  strict rồi relaxed, dedup và top-k. Nếu thiếu postings thì dùng SQL cũ.
- Phrase giữ tối đa 24 cụm 4/3 từ và trọng số BM25 2.5/1.0. Chia thành hai nhóm
  OR liên tiếp, chạy song song qua hai kết nối SQLite chỉ đọc rồi cộng điểm theo ID.
  Không giới hạn ứng viên vào top-k của BM25 rộng. `phrase_workers=1` chạy tuần tự.
- `phrase_cache_mb=64` giữ tối đa 64 MiB postings của các nhóm phrase, giới hạn
  thêm 4.096 key. Cache riêng này không loại bỏ hay thay thứ tự cache BM25 từng từ.
  SQLite dùng page cache tối đa khoảng 64 MiB ở kết nối chính và 32 MiB mỗi worker.
  Mảng trung gian của truy vấn cần thêm RAM; kết nối worker đóng sau mỗi truy vấn.
  Với SQLite trong RAM hoặc chỉ một nhóm cache miss, dùng kết nối hiện có.

Benchmark cuối trên **100 câu, 27.343 đoạn, SQLite file 57.704.448 byte**, tính cả
làm ấm cache, cho kết quả:

| Nhánh | Tổng cũ / 100 câu | Tổng mới / 100 câu | Nhanh hơn |
|---|---:|---:|---:|
| bm25_phrases_query | 3,718 giây | 3,579 giây | 1,04x |
| bm25_precise_query | 0,876 giây | 0,233 giây | 3,76x |

Thứ hạng cả ba nhánh trùng 100/100 câu. Số liệu từng câu nằm trong
[phrase_precise_benchmark.json](phrase_precise_benchmark.json). Phrase mới chỉ
cải thiện nhẹ trên fixture; chưa có bằng chứng giảm mạnh 5–8 giây/câu trên index
Kaggle đầy đủ. Chạy benchmark ở trên để đo trên máy/corpus thực tế.

Phiên Kaggle đang chạy không tự nhận code mới. Code/config đổi fingerprint, nên
dùng một experiment mới để áp dụng; cơ chế resume hiện tại vẫn khóa commit cũ.

## Kiểm tra mã

```text
python -B -m unittest discover -s tests -q
```

Để smoke test GPU trước full run, dùng một bản config riêng trong `/kaggle/working`,
đặt `training.max_examples=8`, giữ world_size=2 và các cấu hình còn lại. Dùng CLI
`prepare-sft`, `retrieve --mode lexical`, rồi `torchrun --standalone
--nproc_per_node=2 --module legalqa --config <smoke-config> --models <runtime-models>
fit --train <smoke-train> --retrieval <smoke-cache> --output <thư-mục-smoke-mới>`.
Chỉ chạy sau khi xác nhận các đường dẫn input có thật; không dùng thư mục full run
cho smoke test. Hoàn thành hai epoch trên 8 QA sẽ có 2 optimizer steps cho mỗi rank.
