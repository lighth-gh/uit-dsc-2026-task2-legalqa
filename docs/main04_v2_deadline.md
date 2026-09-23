# Main 04 ver 2: chạy sát hạn

Import **`legalqa_main_04_v2_deadline_submit.ipynb`** vào Kaggle, bật GPU T4/P100.
Bật Internet nếu cần cài dependencies. Chạy **Run All trong phiên interactive** để tải kết quả ngay.
Không cần Save & Run All rồi đợi version hoàn tất.

## Input cần Add Input

| Input | Nội dung/đường dẫn cấu hình |
|---|---|
| Baseline private 0.5713 | `submission.zip` chỉ chứa `submission.json`, hoặc JSON đã giải nén; đặt `BASELINE_SUBMISSION` |
| Diagnostics Stage 2 private | `legalqa_main_stage2_v8_diagnostics (5).zip` hoặc thư mục có `stage2_manifest.json`; đặt `STAGE2_DIAGNOSTICS` |
| Model Version 3 | Dataset `lighth/ver3-smoke-output`; `MODEL_ROOT` là thư mục `models/` có `models.lock.json`, `generator/` và config embedding/reranker |
| Adapter đã chọn | Output Stage 2/3 có `selected_adapter/adapter_model.safetensors` và `adapter_config.json`; đặt `ADAPTER_ROOT` tới thư mục này |

Mặc định các biến là `None`: tự tìm nếu duy nhất một nguồn. Khi có nhiều nguồn, notebook in danh sách để điền đường dẫn thật trong `/kaggle/input/...`.
Không cần corpus/index hoặc `private-official.json` riêng: câu hỏi và context đã có trong Stage 2.
Diagnostics không chứa adapter weights. Không dùng Stage 3 public `(8)` paused 900/1.000.
`PRIVATE_DIAGNOSTICS=None` là đủ; chỉ điền nếu có Stage 3 hoàn chỉnh đúng private để nhận diện theo audit.
Không có audit thì dùng heuristic fallback tái dựng/câu dang dở/độ dài gần trần; đây là nhận diện nghi vấn.

## Cấu hình và cách chạy

- `WORK_MINUTES=22` tính từ lúc chạy cell cấu hình, gồm cài đặt, đọc dữ liệu, load model, generation.
  **Nếu còn dưới 25 phút, sửa thành số phút còn lại trừ 3 phút tải/nộp.**
- Một hướng duy nhất: greedy, penalty 1.0, không cấm n-gram, 4.096 input + tối đa 3.072 output token,
  giữ tối đa 4 context theo thứ tự cũ; không giảm top-k để tăng tốc. Mỗi ID chỉ sinh một lần, không retry.
- Ưu tiên fallback, rồi truncated có audit/câu dang dở, cuối cùng nghi vấn gần trần.
- Dùng một GPU `cuda:0`, tối đa khoảng 150 giây sinh mỗi câu; dừng không EOS thì giữ baseline.
  Trần 3.072 cho thêm chỗ hoàn thành so với bản gốc 1.536, nhưng câu dài sẽ tốn thời gian hơn.
- Không thử dev, không chấm điểm, không quét biến thể, không retrain/retrieval/rerank.
  Vẫn kiểm tra ID/hash input, adapter/model, schema và căn cứ đầu ra trước khi thay đáp án.

## Output để nộp

**Tải `/kaggle/working/submission.zip`** trong Files/Output hoặc link cuối cell.
ZIP chỉ chứa `submission.json` với schema `{ID: {"answer": "..."}}`, đủ 1.918 ID khi dùng baseline này.

ZIP baseline được tạo trước cài dependencies. Mỗi câu hoàn thành sẽ ghi checkpoint và cập nhật ZIP
bằng thay file atomic; bản ở gốc working được đồng bộ mỗi giây. Hết giờ/lỗi thì giữ kết quả đã hoàn thành.
Câu chưa làm, bị cắt do thời gian, fallback lại hoặc không qua guard giữ nguyên baseline.
Có thể tải ZIP lúc cell đang chạy, không cần hoàn thành hàng đợi. Nếu kernel chết cứng,
ZIP gần nhất vẫn có trên đĩa nhưng cell đang sinh có thể chưa được lưu.

`/kaggle/working/main04_v2_deadline/` chứa bản ZIP checkpoint, `candidates.json`, `attempts.jsonl`,
`status.json` (attempted/changed/remaining), và `error.json`/`notebook_stop.json` nếu lỗi.
Chạy lại cùng notebook/input trong cùng phiên sẽ bỏ qua ID đã checkpoint. Không tự resume output bản main 4 cũ.

Chưa chạy GPU thật tại local, không có bảo đảm tăng điểm. Bỏ dev là quyết định theo yêu cầu sát hạn.
File `submission.zip` đang có trong repository là baseline người dùng, không bị ghi đè khi tạo notebook.

Kiểm tra local:

```powershell
python -m unittest discover -s tests -p test_deadline_repair.py -v
python scripts/build_stage4_v2_notebook.py
```
