# Dùng lại retrieval 5.600 câu trong lượt train có bản sửa VRAM

Có thể tạo dataset từ output cũ hoặc gắn output notebook trực tiếp vào Input.
Chỉ cần hai file cùng thư mục:

- `train.sft.lexical.retrieval.json`
- `stage1_manifest.json`

Diagnostics `legalqa_main_stage1_v8_diagnostics (3).zip` đã chứa đủ cả hai.
`RETRIEVAL_INPUT` cũng nhận trực tiếp đường dẫn tới ZIP này, không cần giải nén
toàn bộ checkpoint/log/data. Dataset train và dataset Version 3 chứa index/model
vẫn cần được gắn như trước.

Trong cell cấu hình của notebook `legalqa_main_01_qlora_train.ipynb` bản mới:

```python
# Dùng đường dẫn thật hiển thị ở Input của bạn.
RETRIEVAL_INPUT = Path('/kaggle/input/datasets/<owner>/<dataset>')
PREVIOUS_OUTPUT = None
UPSTREAM_OUTPUT = None
LEGACY_INPUT_ROOT = None
REPO_REVISION = None  # main chứa bản sửa đã push; hoặc SHA mới cụ thể
```

Nếu hai file nằm trong thư mục con, trỏ tới thư mục con đó. Nếu dataset giữ
nguyên ZIP, trỏ tới file ZIP. Chọn GPU T4 x2. Dùng phiên mới để tránh code/output
đã checkout của lần chạy trước.

Notebook bỏ tự dò resume khi có `RETRIEVAL_INPUT`, nên dataset cũ không ép
checkout lại commit gây OOM. Sau khi chuẩn bị split train, pipeline nhập cache,
in `Imported lexical retrieval: 5600 records; BM25 skipped` rồi gọi QLoRA `fit`.
Không gọi BM25/dense/reranker cho các câu train này.

Đây là tái sử dụng retrieval và bắt đầu QLoRA với code mới. Không phải resume
optimizer tại bước 130. Những phiên tiếp theo của **lượt mới** dùng
`RETRIEVAL_INPUT=None` và `PREVIOUS_OUTPUT` trỏ tới output mới như bình thường.

## Điều kiện kiểm tra

Importer kiểm tra SHA-256/size của cache theo manifest, đủ chính xác tập ID,
hash và nội dung câu hỏi, model revisions, index, retrieval config, lexical
pool và mode. Snapshot train thất bại vẫn dùng được nếu cache retrieval đã hoàn tất.

Code cũ chỉ được chấp nhận cho fingerprint đã đối chiếu từ commit
`a574bdb960afb23bff55edb773ab999e40fdc77a`, và chỉ khi sáu module phụ thuộc
retrieval vẫn có nội dung như commit đó. Không bỏ kiểm tra code chung cho
mọi cache. Import giữ toàn bộ records, lưu original identity và checksum nguồn
trong `import_provenance`, rồi ghi identity tương thích vào bản sao ở working.
Không sửa file nguồn, không sao chép checkpoint optimizer hoặc session cũ.

## Kết quả kiểm chứng

- Import trực tiếp diagnostics `(3).zip`: 5.600 bản ghi.
- Hash records trước/sau cùng là
  `e1c09ead17777f9730e76f9a60892ae3e2c3c31a55cb3ce31468276b182399df`.
- Import + đọc lại + đối chiếu local: 8,80 giây; không phải dự báo tốc độ Kaggle.
- Bộ test: 104 tests, 103 pass, 1 CUDA skip. Có test Stage 1 chỉ gọi `fit`,
  không gọi `retrieve`; test từ chối cache thiếu/sai ID, khác code/config/index/model,
  nội dung bị sửa, và test tránh tự pin commit cũ từ dataset.
