# Tối ưu riêng thời gian đọc của bm25_phrases_query

Log mới có 57 mốc lexical riêng biệt: phrase trung bình 7,261 giây, median
6,114 giây và tối đa 19,436 giây. Đây là mẫu các câu được log, không phải
trung bình toàn bộ tập train. Precise đã xuống khoảng 0,006 giây nên giữ nguyên.

## Phạm vi và điều kiện bảo toàn chất lượng

Bản sửa chỉ thay cách mở và đọc SQLite của các worker phrase:

- Tái sử dụng hai kết nối chỉ đọc và thread pool trong một lượt retrieve.
  Bản trước tạo rồi đóng cả hai kết nối ở từng câu.
- Bật `phrase_mmap_mb=1024` cho mỗi kết nối worker. Memory mapping cho phép
  SQLite đọc các trang qua bộ nhớ đệm của hệ điều hành và giảm sao chép dữ liệu;
  hệ thống không hỗ trợ sẽ dùng cách đọc thông thường.
  [Giải thích của SQLite](https://www.sqlite.org/mmap.html).
- Đóng pool/kết nối khi hoàn thành, pause hoặc truy vấn phát sinh lỗi. Nếu một
  worker lỗi, đợi worker còn lại kết thúc trước khi cho phép tái sử dụng kết nối.

Hàm chọn phrase, 24 cụm 4/3 từ, top-k, phân nhóm, câu SQL, trọng số BM25,
phép cộng điểm, cách xử lý đồng điểm và RRF giữ nguyên. BM25 rộng, precise,
QLoRA, generation và định dạng submission không đổi. Không tạo hay sửa index.

Benchmark đối chiếu **toàn bộ ID và điểm BM25 của từng nhóm ứng viên**, bằng
so sánh bằng nhau tuyệt đối, rồi đối chiếu thứ hạng đầu ra. Vì phép truy xuất
không thay đổi, tối ưu này không đặt ra đánh đổi recall hoặc độ dài câu trả lời.
Chưa chạy lại generation/METEOR/ROUGE trên Kaggle; số metric thực tế vẫn cần
đánh giá trên pipeline đầy đủ, không coi benchmark tốc độ là phép đo metric.

## Kết quả kiểm tra

Mỗi benchmark dùng 100 câu từ diagnostics đã cung cấp, tính cả khởi tạo,
luân phiên thứ tự chạy cũ/mới để giảm lợi thế từ OS cache.

| Bộ kiểm tra | Tổng phrase cũ | Tổng phrase mới | Giảm thời gian |
|---|---:|---:|---:|
| 27.343 đoạn, SQLite ~55 MiB | 1,095 giây | 0,860 giây | 21,5% |
| 407.107 đoạn, SQLite ~739 MiB | 19,090 giây | 16,818 giây | 11,9% |

Ở cả hai bộ: **100/100 câu trùng thứ hạng và toàn bộ điểm số**.
File [benchmark nhỏ](phrase_readers_benchmark.json) và
[benchmark quy mô lớn](phrase_readers_scale_benchmark.json) chứa từng phép đo.
Bộ nhỏ dùng cửa sổ 180 từ, stride 150 từ các parent trong diagnostics; bộ lớn
nhân lặp những đoạn đó đến 407.107 dòng. Bộ lớn kiểm tra quy mô I/O, **không phải
corpus Kaggle gốc**, nên không suy trực tiếp các tỷ lệ trên ra log 7–19 giây/câu.

## Kiểm tra trên corpus Kaggle thật

Sau khi đưa code mới lên repo notebook clone, dùng một phiên mới và các file
đã có thật. Cell sau sử dụng các biến do notebook định nghĩa:

```python
database = VERSION3_ROOT / 'index' / 'corpus.sqlite'
questions = RUN_ROOT / 'data' / 'train.sft.questions.json'
assert database.is_file() and questions.is_file()
bounded_process([
    sys.executable, CODE / 'scripts' / 'benchmark_phrases.py',
    '--database', database, '--questions', questions,
    '--config', RUN_ROOT / 'config.json', '--limit', '100',
    '--output', WORK / 'phrase_io_benchmark.json',
], cwd=CODE)
```

Chỉ benchmark phrase; baseline là bản ngay trước thay đổi này, vẫn cache và chạy
hai nhóm phrase như cũ. Kiểm tra `identical_rankings=true` và
`identical_full_scores=true`; script trả lỗi nếu không khớp. `mmap_bytes` cho biết
dung lượng thực tế SQLite cho phép map, có thể nhỏ hơn cấu hình hoặc bằng 0.

`phrase_reuse_readers=false` quay về cách tạo/đóng kết nối từng câu để đối chiếu.
`phrase_mmap_mb=0` giữ tái sử dụng kết nối nhưng tắt memory mapping. Page cache
worker vẫn 32 MiB/kết nối. Mmap giới hạn vùng địa chỉ file được map, không phải
cấp phát hai mảng RAM 1 GiB; mức RAM resident phụ thuộc số trang thực sự truy cập.

Phiên đang chạy tiếp tục dùng commit đã khóa. Code/config mới đổi fingerprint;
không sửa manifest cũ để ép resume bằng code mới.

```text
python -B -m unittest discover -s tests -q
```
