# Phạm vi kiểm định bộ code

Kiểm định local bao phủ phần chạy CPU và các contract của pipeline. V7 đã chạy thành công 100 câu trên Kaggle và đạt METEOR 0,45299; quality V8 sửa các fallback sai còn lại và thêm QLoRA bắt buộc trong main run. V8 vẫn phải chạy lại smoke30/dev100 trên Kaggle trước khi sinh submission.

| Kiểm định | Trạng thái |
| --- | --- |
| Đối chiếu ba model với Excel được gửi | Đã thực hiện |
| Tính tổng tham số từ cấu hình kiến trúc đầy đủ | Đã thực hiện; 3.801.281.793 kể cả LoRA mặc định |
| So sánh vendor scorer với ZIP gốc | Kiểm tra byte/hash khi đóng gói |
| 33 unit tests, gồm precise/phrase/lexical BM25, legal boosts, local evidence/entity guard, QLoRA/NF4 guard, prompt budget và prepare → submission ZIP | 33/33 qua trên CPU |
| Biên dịch cú pháp toàn bộ code và code cells notebook | Kiểm tra khi đóng gói |
| Khôi phục checkpoint bị ngắt và chặn cache sai cấu hình | Đã chạy CPU |
| Mã chấm BTC chạy thật với NLTK/WordNet | Đã chạy trong Kaggle V5–V7; notebook V8 tiếp tục tự kiểm chứng |
| Đếm tham số bằng model PyTorch trên meta device | Kaggle V5 đạt 3.801.281.793 tham số kể cả LoRA |
| Load đúng ba checkpoint thật từ Dataset Version 3 | Đã chạy trong Kaggle V5 |
| Generation trên Kaggle | V7 hoàn tất 100/100; QLoRA V8 chưa chạy |
| Mốc so sánh dev100 | V5: METEOR 0,44316; V6: 0,42585; V7: 0,45299 trên cùng IDs/reference hash |
| QLoRA main run | Bắt buộc 2 epoch trên tối đa 768 QA; prompt SFT tối đa 2.048 token; từng checkpoint phải được chấm trên dev100 |
| Mục tiêu lựa chọn | Ưu tiên METEOR; ROUGE-L chỉ phá hòa; yêu cầu METEOR ≥ 0,65 |

Các kiểm định quan trọng bao gồm không lẫn câu hỏi/đáp án trùng qua split; không lấy số trang web hoặc ngày `01/08` làm số luật; BM25 cụm từ/chính xác; legal boosts không dùng reference; câu pháp lý chứa “không đủ thông tin” không bị coi là refusal; cửa sổ child phủ nội dung và không vượt parent; context đầu được thêm ngân sách nhưng các context sau vẫn có mức tối thiểu; cứu output bị cắt; mask prompt và giữ nguyên target SFT; ID/schema JSON; cache sai cấu hình bị từ chối; journal tiếp tục được sau dòng ghi dở; ba notebook dùng chung config và Dataset paths.

Trước main run dài, chạy quality V8 trên smoke30 rồi cùng dev100 đã dùng cho V5–V7 để so sánh đúng IDs. Kiểm tra riêng các ID 4927, 11029, 12165, 14843, 15927, 51037, 111153, 136787 và 143605. Main run phải tạo `training_result.json`, checkpoint epoch 1/2 và báo cáo METEOR của từng checkpoint; không coi mục tiêu 0,65 là đã đạt trước khi báo cáo xác nhận. Giữ riêng holdout theo hướng dẫn.
