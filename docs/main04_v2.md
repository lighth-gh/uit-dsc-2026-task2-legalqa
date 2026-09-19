# Main 04 P0/P1/P2 trên diagnostics (5)

Notebook hiện điều phối sẵn baseline P0, inference ablation P1, retrieval ablation P2 và workflow repair V2 cũ. Điểm public người dùng đã báo: 0,5580 trước sửa, 0,5599 sau V1. Chưa có bằng chứng đạt public 0,60.

## Chạy Kaggle

Chỉ cần tải từ máy lên Kaggle một file: `legalqa_main_04_repair_submit.ipynb`. Tạo notebook bằng **New Notebook → File → Import Notebook**, sau đó chọn file này. Trong **Settings**, chọn accelerator **GPU T4 x2** nếu có (T4/P100 một GPU vẫn dùng được) và bật Internet để cài các gói còn thiếu.

Trong **Add Input**, gắn đủ ba nguồn sau. Không tải ZIP/model vào `/kaggle/working` bằng tay:

| Input cần gắn | Chọn ở Kaggle | File đánh dấu notebook sẽ kiểm |
| --- | --- | --- |
| Model và index Version 3 | **Add Input → Datasets**, tìm `lighth/ver3-smoke-output` | `models/models.lock.json`, `models/generator/`, `index/` |
| Checkpoint đã chọn của đúng lượt train | **Add Input → Your Work / Notebook Output**, chọn output Stage 2; nếu output Stage 3 của bạn có nguyên `selected_adapter/` thì có thể dùng nó | `selected_adapter/adapter_model.safetensors`, `selected_adapter/adapter_config.json` |
| Diagnostics hoàn chỉnh | **Add Input → Your Work / Notebook Output**, chọn version Stage 3 đã `complete` | `legalqa_main_stage3_v8_diagnostics*.zip` hoặc thư mục giải nén có `stage3_manifest.json` |

Diagnostics ZIP không chứa adapter weights, vì vậy chỉ gắn ZIP là chưa đủ. Không gắn nhiều version Stage 2/3 cùng lúc nếu không cần; notebook sẽ dừng thay vì đoán khi tìm thấy nhiều adapter hoặc diagnostics.

Sau khi Add Input:

1. Mở cell cấu hình đầu tiên. Lượt đầu giữ `MODE='p1_dev'`, `RUN_GPU=True`, `GPU_MAX_ITEMS=50`, `WORK_HOURS=9.0`, `PREVIOUS_OUTPUT=None`.
2. `MODEL_ROOT` mặc định trỏ tới `lighth/ver3-smoke-output`. Để `ADAPTER_ROOT=None` và `DIAGNOSTICS=None` nếu mỗi loại chỉ có đúng một kết quả. Nếu notebook báo nhiều kết quả, chép đúng đường dẫn được in trong lỗi vào biến tương ứng. `MODEL_ROOT` là thư mục `models`, tức thư mục cha của `generator/`.
3. Bấm **Save Version → Save & Run All**. Output chung được ghi ở `/kaggle/working/legalqa_main_04_v8_060_focused_loop/`; trạng thái điều phối nằm trong `main04_state.json`.
4. Nếu cuối log là `paused`, lưu version có output. Ở phiên sau, Add Input output của chính version đó qua **Your Work / Notebook Output**, đặt `PREVIOUS_OUTPUT` tới thư mục `.../legalqa_main_04_v8_060_focused_loop`, giữ nguyên `MODE` và danh sách variant, rồi Save & Run All lại.
5. Khi một mode báo `complete`, tải báo cáo/ZIP trong tab **Output**, hoặc Save Version để dùng toàn bộ thư mục output làm Input cho mode kế tiếp.

## Các mode có sẵn

- `p1_dev`: chỉ regenerate câu có vòng lặp và so sánh penalty 1.00/1.03/1.05; ghi `p1/p1_summary.json` cùng `p1/penalty_comparison.json`.
- `p1_public`: điền `P1_WINNER`; notebook từ chối nếu variant không có `passes_screen=true`, rồi resume và đóng `submission_<variant>.zip` khi đủ 1.000 ID.
- `p2_retrieval`: chạy năm variant retrieval, ghi `p2/retrieval_summary.json`.
- `p2_generate`: điền tối đa hai tên vào `P2_SHORTLIST`; generation dev100 có journal, sau đó repair, evaluate và paired compare.
- `repair_v2`: giữ workflow recipe 2 trước đây dưới `legalqa_main_04_v8_060_focused_loop/repair_v2`.

Các mode `p1_dev`, `p1_public`, `p2_retrieval` và `p2_generate` bắt buộc `RUN_GPU=True`. Chỉ `repair_v2` có thể đặt `RUN_GPU=False` để chạy lại sửa CPU cũ. Khi đổi code sau một phiên đã lưu, bắt đầu output mới để không trộn kết quả.

## Quy tắc và chi phí

- V1 được chạy/chấm lại làm đối chứng; V2 chỉ thêm xóa vòng lặp từ 6 mục liên tiếp có cùng nội dung nhưng khác nhãn `a)`, `b)` hoặc `1)`, `2)`. Không xóa phần kết luận, không sửa câu khác số liệu, phủ định hoặc điều kiện.
- GPU recipe 2 dùng greedy, `repetition_penalty=1.0`, `no_repeat_ngram_size=0`, cộng chỉ dẫn cho phép chép nguyên văn nguồn, giữ căn cứ/số liệu và chỉ tránh lặp nguyên câu/đoạn từ ba lần liên tiếp. Prompt chống hallucination cũ vẫn giữ nguyên. Giữ `max_input_tokens`, `max_new_tokens`, top-k và thứ tự context; chỉ dẫn mới chiếm một phần ngân sách input nên cửa sổ context thực tế có thể ngắn hơn. Không huấn luyện lại, không truy xuất lại corpus.
- P1 chỉ lấy ID có vòng lặp từ prediction gốc: block/câu dài lặp lại hoặc danh sách đánh số tăng dần có cùng nội dung. Câu khác, kể cả token-limit/fallback nhưng không lặp, không bị sinh lại. Evidence yếu được bỏ qua và ghi rõ.
- Ba lượt P1 dùng cùng prompt tập trung và cùng context policy: tối đa hai parent thuộc cùng văn bản với nguồn top-1, giữ biên khoản/điểm hoàn chỉnh; chỉ `repetition_penalty` thay đổi giữa 1.00/1.03/1.05.
- Sinh toàn bộ nhóm dev được chọn trước; giữ nguyên các dev khác để chấm toàn bộ dev100. Cần METEOR tăng ít nhất 0,001 so với CPU, ít nhất 2 đáp án dev thay đổi và cờ lặp nặng không tăng. Không dùng gold trong prompt, không chọn bản sửa theo điểm riêng từng câu.
- Public chỉ chạy sau khi dev đạt điều kiện. Bản sinh chạm token, fallback/refusal, thiếu evidence, xung đột thực thể/số hiệu hoặc vẫn dang dở/lặp bị từ chối; giữ đáp án CPU của câu đó. Quy tắc này giống nhau ở dev/public.
- `GPU_MAX_ITEMS` là số câu **mới mỗi phiên**, cộng cả dev và public; không phải giới hạn tổng nhóm thử. `WORK_HOURS` bao gồm setup, CPU, GPU và chấm. Recipe 1 đã hoàn tất thử dev trên Kaggle trong khoảng 17 phút cho toàn phiên, nhưng không đạt điều kiện; chưa đo runtime recipe 2. Có thể cần nhiều phiên. Quá hạn cứng có thể phải chạy lại câu chưa ghi journal.

## Đầu ra

`repair.metrics.json` ghi điểm gốc, V1, V2 và quyết định GPU nếu có. `repair.audit.json` giữ thay đổi CPU; `repair.unresolved.json` ghi cờ của bản được chọn. `gpu/candidates.json` chứa ID được thử/bỏ qua; `gpu/decision.json` có chênh lệch dev từng câu; `gpu/*.checkpoint.jsonl` chứa generation, kết quả được giữ và lý do từ chối. `conservative/` giữ đầy đủ đối chứng V1.

Code kiểm CRC/hash/ID/journal từ diagnostics, kiểm adapter hash và model lock của Stage 3. Hash file generator/tokenizer được ghi vào identity GPU để từ chối resume nếu file đổi. ZIP đầu ra chỉ có một `submission.json`, schema/ID được kiểm lại; không ghi đè `submission.zip` gốc.

## Giới hạn kiểm chứng

CPU được chạy trực tiếp trên diagnostics người dùng gửi. Người dùng đã chạy GPU recipe 1 trên Kaggle và cung cấp log/journal; kết quả METEOR 0,61828838 thấp hơn CPU 0,62024531, nên không chạy public GPU. Recipe 2 đã kiểm orchestration/guards/resume bằng tests giả lập, chưa chạy model GPU thực. Dev100 đã dùng chọn checkpoint nên không phải đánh giá độc lập. Không thể suy ra mức tăng public từ dev hoặc cam kết 0,59.

Kết quả CPU trên diagnostics (5): V1 METEOR 0,62024531 / ROUGE-L 0,59381939; ứng viên V2 METEOR 0,62014189 / ROUGE-L 0,59570201. V2 bị từ chối, giữ V1 (37 câu public đã sửa). GPU sẽ dùng V1 làm đối chứng cho dữ liệu này; chưa có submission mới được chứng minh tốt hơn bản public 0,5599.

Danh sách thử: 15 câu dev đủ điều kiện, 9 câu dev bị bỏ qua vì evidence yếu; 128 câu public đủ điều kiện, 97 câu public bị bỏ qua vì evidence yếu. Với giới hạn 50 câu mới/phiên, nếu dev đạt và tất cả 128 câu public được chạy thì cần ít nhất 3 phiên; hạn thời gian có thể làm số phiên nhiều hơn.

Một phương án xóa kết luận lặp đã bị loại: METEOR dev100 giảm từ 0,62025 xuống 0,59447 dù ROUGE-L tăng. Quy tắc này không có trong bản phát hành. Không chọn sửa theo ID hoặc gold để che các câu giảm điểm.

Xem [chẩn đoán journal recipe 1](main04_gpu_journal_review.md) để biết lý do bỏ no-repeat 12-gram. Đây là sửa cơ chế có thể cản sao chép nguồn, chưa phải bằng chứng recipe 2 sẽ tăng điểm.
