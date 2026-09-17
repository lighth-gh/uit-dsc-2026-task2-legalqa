# Main 04 V2 trên diagnostics (5)

Mục tiêu của bản này là cải thiện submission bằng sửa lặp và thử sinh lại có chọn lọc, có đối chứng với Main 04 V1. Điểm public người dùng đã báo: 0,5580 trước sửa, 0,5599 sau V1. Chưa có bằng chứng đạt public 0,59.

## Chạy Kaggle

1. Import lại `legalqa_main_04_repair_submit.ipynb` đã cập nhật; chọn GPU T4 hoặc P100, bật Internet.
2. Add Input diagnostics Stage 3 hoàn tất của lần chạy này, output Stage 2/3 có `selected_adapter/adapter_model.safetensors` và `adapter_config.json`, dataset model gốc có `models.lock.json` và các thư mục model. Diagnostics ZIP không chứa adapter weights.
3. Mặc định `RUN_GPU=True`, `GPU_MAX_ITEMS=50`, `WORK_HOURS=9.0`. Nếu phát hiện nhiều input, điền `DIAGNOSTICS`, `MODEL_ROOT`, `ADAPTER_ROOT`; không đổi model hoặc dùng adapter của phiên khác. `MODEL_ROOT` là thư mục cha của `generator/`, không phải `generator/`.
4. Save & Run All. Xem `STATUS` và `selected_variant` ở cell kết quả. Khi `paused`, Save output, gắn toàn bộ output vào phiên tiếp theo rồi đặt `PREVIOUS_OUTPUT` đến thư mục `legalqa_main_stage4_v2` đó. Giữ nguyên code, input và cấu hình. Resume dùng journal, không sinh lại các câu đã ghi xong.
5. Lấy `submission_selected.zip`. Nếu `paused`, ZIP này là bản CPU đầy đủ; nếu GPU bị từ chối trên dev, cũng giữ CPU. Chỉ khi toàn bộ nhóm public hoàn tất và được kiểm tra mới chọn `gpu_v1`.

Muốn chạy CPU trước: `RUN_GPU=False`, Accelerator None. Sau đó có thể bật GPU và tiếp tục cùng output/cùng code; trạng thái GPU có identity riêng. Khi đổi code sau một phiên đã lưu, bắt đầu output mới để không trộn kết quả.

## Quy tắc và chi phí

- V1 được chạy/chấm lại làm đối chứng; V2 chỉ thêm xóa vòng lặp từ 6 mục liên tiếp có cùng nội dung nhưng khác nhãn `a)`, `b)` hoặc `1)`, `2)`. Không xóa phần kết luận, không sửa câu khác số liệu, phủ định hoặc điều kiện.
- GPU dùng recipe cố định: greedy, `repetition_penalty=1.08`, `no_repeat_ngram_size=12`. Giữ nguyên `max_input_tokens`, `max_new_tokens`, số context và thứ tự context của Stage 3. Không huấn luyện lại, không truy xuất lại corpus. Chống lặp có thể làm model bỏ một cụm pháp lý cần lặp; vì vậy phải đo tác động trên dev trước public.
- Danh sách thử dựa trên cờ lặp, câu dẫn dang dở, chạm token hoặc fallback, kèm kiểm tra evidence mạnh. Đây là heuristic, không chứng minh đầy đủ căn cứ. Câu evidence yếu được bỏ qua và ghi rõ để xem lại retrieval.
- Sinh toàn bộ nhóm dev được chọn trước; giữ nguyên các dev khác để chấm toàn bộ dev100. Cần METEOR tăng ít nhất 0,001 so với CPU, ít nhất 2 đáp án dev thay đổi và cờ lặp nặng không tăng. Không dùng gold trong prompt, không chọn bản sửa theo điểm riêng từng câu.
- Public chỉ chạy sau khi dev đạt điều kiện. Bản sinh chạm token, fallback/refusal, thiếu evidence, xung đột thực thể/số hiệu hoặc vẫn dang dở/lặp bị từ chối; giữ đáp án CPU của câu đó. Quy tắc này giống nhau ở dev/public.
- `GPU_MAX_ITEMS` là số câu **mới mỗi phiên**, cộng cả dev và public; không phải giới hạn tổng nhóm thử. `WORK_HOURS` bao gồm setup, CPU, GPU và chấm. Chưa có benchmark GPU cho recipe này, có thể cần nhiều phiên. Quá hạn cứng có thể phải chạy lại câu chưa ghi journal.

## Đầu ra

`repair.metrics.json` ghi điểm gốc, V1, V2 và quyết định GPU nếu có. `repair.audit.json` giữ thay đổi CPU; `repair.unresolved.json` ghi cờ của bản được chọn. `gpu/candidates.json` chứa ID được thử/bỏ qua; `gpu/decision.json` có chênh lệch dev từng câu; `gpu/*.checkpoint.jsonl` chứa generation, kết quả được giữ và lý do từ chối. `conservative/` giữ đầy đủ đối chứng V1.

Code kiểm CRC/hash/ID/journal từ diagnostics, kiểm adapter hash và model lock của Stage 3. Hash file generator/tokenizer được ghi vào identity GPU để từ chối resume nếu file đổi. ZIP đầu ra chỉ có một `submission.json`, schema/ID được kiểm lại; không ghi đè `submission.zip` gốc.

## Giới hạn kiểm chứng

CPU được chạy trực tiếp trên diagnostics người dùng gửi. GPU orchestration/guards/resume được kiểm bằng tests giả lập; chưa chạy model GPU thực trên máy local. Dev100 đã dùng chọn checkpoint nên không phải đánh giá độc lập. Không thể suy ra mức tăng public từ dev hoặc cam kết 0,59.

Kết quả CPU trên diagnostics (5): V1 METEOR 0,62024531 / ROUGE-L 0,59381939; ứng viên V2 METEOR 0,62014189 / ROUGE-L 0,59570201. V2 bị từ chối, giữ V1 (37 câu public đã sửa). GPU sẽ dùng V1 làm đối chứng cho dữ liệu này; chưa có submission mới được chứng minh tốt hơn bản public 0,5599.

Danh sách đã tính trước, chưa sinh GPU: 15 câu dev đủ điều kiện, 9 câu dev bị bỏ qua vì evidence yếu; 128 câu public đủ điều kiện, 97 câu public bị bỏ qua vì evidence yếu. Với giới hạn 50 câu mới/phiên, nếu dev đạt và tất cả 128 câu public được chạy thì cần ít nhất 3 phiên; hạn thời gian có thể làm số phiên nhiều hơn.

Một phương án xóa kết luận lặp đã bị loại: METEOR dev100 giảm từ 0,62025 xuống 0,59447 dù ROUGE-L tăng. Quy tắc này không có trong bản phát hành. Không chọn sửa theo ID hoặc gold để che các câu giảm điểm.
