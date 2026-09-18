# Chẩn đoán Main 04 từ dev.checkpoint.jsonl

Nguồn: journal 15 câu người dùng tải từ Kaggle và `legalqa-main-04.log`, đối chiếu diagnostics Stage 3 (5), đáp án CPU V1 và references dev100. Journal không kèm file metadata identity; kiểm tra được 15 ID duy nhất, cấu trúc bản ghi và các đáp án bị từ chối khớp CPU, không kiểm độc lập được model/adapter chỉ từ file này.

## Kết quả thực tế của GPU recipe 1

- 10/15 câu có `unsupported_document_numbers`, chuyển `source_fallback` và bị Stage 4 từ chối. Không phải cả 11 câu bị loại đều chạm token.
- Chỉ ID `33507` chạm trần 1.536 token, route `generated_truncated`; bản raw kéo dài một danh sách rất nhiều nhánh. Giữ đáp án CPU là đúng theo chính sách hiện tại.
- 4 câu được guard nhận. So với CPU, METEOR từng câu thay đổi: `22579` −0,04803; `8593` +0,14030; `129715` −0,13479; `53207` −0,15317. Toàn dev100 giảm 0,00195693, từ 0,62024531 xuống 0,61828838. ROUGE-L tăng lên 0,59689525 nhưng tiêu chí chính là METEOR, nên từ chối cả recipe và không chạy public.
- Điểm tăng ở `8593` không chứng minh đáp án đúng: raw vẫn khác reference về năm, điều và điều kiện triệu tập đại hội. Guard số hiệu chưa kiểm chứng mọi năm, tỷ lệ, điều/khoản hay ý nghĩa.

## Dấu hiệu từ output

`108871` đổi `QĐ-TTg` thành `QĐ-TTGTg` và biến dạng thuật ngữ. `158411` có cả đoạn tiếng Trung trong câu trả lời tiếng Việt. `129715` đổi “phương thức đa cấp” thành các cụm khác và giảm độ bao phủ; `53207` mất nhóm đối tượng, xuất hiện dữ kiện khác reference. Một số số hiệu chỉ sai khoảng trắng/dấu nối, nhưng nhiều đáp án còn sai nội dung nên không nên chỉ nới regex hoặc tự sửa số hiệu rồi nhận lại raw.

## Cơ chế cần sửa

Trong Transformers 4.51.3, cả repetition penalty và bộ chặn n-gram mặc định của decoder-only model đều xét các token trong prompt. No-repeat đặt xác suất token bị cấm về âm vô cùng. Vì vậy cấu hình `no_repeat_ngram_size=12` có thể chặn model chép lại chính cụm từ có trong context. [Mã nguồn chính thức Transformers](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/generation/logits_process.py#L800).

Cơ chế này là sự thật từ implementation; việc nó góp phần gây các lỗi cụ thể trong journal là suy luận phù hợp với dữ liệu, chưa phải kết quả đối chứng loại bỏ từng tham số. Không quy mọi lỗi cho một tham số, vì retrieval và năng lực model cũng có giới hạn.

## Bản sửa để thử tiếp

Recipe 2 tắt no-repeat n-gram, đưa repetition penalty về 1.0; bổ sung chỉ dẫn giữ nguyên thuật ngữ/căn cứ/số liệu, chỉ tránh lặp nguyên câu/đoạn nhiều lần. Không lấy gold làm prompt, không chọn sửa theo ID, không phục hồi raw bị guard loại. Giữ mọi guard, ngân sách token và quy tắc chấm toàn dev100 trước public.

Thêm chỉ dẫn chiếm một phần ngân sách input 4.096 token của run này, nên phần context được pack có thể ngắn hơn. Cần đọc audit input/output token và điểm dev ở lượt thử mới. Output recipe 2 là thư mục mới; không dùng journal recipe 1 để giả lập đã sinh bằng cấu hình mới.

Chưa có GPU output recipe 2. Điểm public 0,5599 do người dùng báo vẫn là mốc hiện tại; chưa chứng minh đạt 0,59.
