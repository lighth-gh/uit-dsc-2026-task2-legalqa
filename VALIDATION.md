# Phạm vi kiểm định bộ code

Kiểm định local bao phủ phần chạy CPU và các contract của pipeline. V6 đã chạy thành công 100 câu trên Kaggle nhưng regression ở nhóm refusal/fallback; quality V7 phải chạy lại smoke30/dev100 trước khi dùng full dev.

| Kiểm định | Trạng thái |
| --- | --- |
| Đối chiếu ba model với Excel được gửi | Đã thực hiện |
| Tính tổng tham số từ cấu hình kiến trúc đầy đủ | Đã thực hiện; 3.801.281.793 kể cả LoRA mặc định |
| So sánh vendor scorer với ZIP gốc | Kiểm tra byte/hash khi đóng gói |
| 24 unit tests, gồm precise/phrase BM25, legal boosts, refusal/date guard, prompt budget và prepare → submission ZIP | 24/24 qua trên CPU |
| Biên dịch cú pháp toàn bộ code và code cells notebook | Kiểm tra khi đóng gói |
| Khôi phục checkpoint bị ngắt và chặn cache sai cấu hình | Đã chạy CPU |
| Mã chấm BTC chạy thật với NLTK/WordNet | Đã chạy trong Kaggle V5/V6; notebook V7 tiếp tục tự kiểm chứng |
| Đếm tham số bằng model PyTorch trên meta device | Kaggle V5 đạt 3.801.281.793 tham số kể cả LoRA |
| Load đúng ba checkpoint thật từ Dataset Version 3 | Đã chạy trong Kaggle V5 |
| Generation trên Kaggle | V5 hoàn tất 100/100; QLoRA chưa chạy |
| Mốc so sánh quality V7 | V5: METEOR 0,44316 / ROUGE-L 0,49798; V6: 0,42585 / 0,48897 trên cùng dev100 và reference hash |

Các kiểm định quan trọng bao gồm không lẫn câu hỏi/đáp án trùng qua split; không lấy số trang web hoặc ngày `01/08` làm số luật; BM25 cụm từ/chính xác; legal boosts không dùng reference; câu pháp lý chứa “không đủ thông tin” không bị coi là refusal; cửa sổ child phủ nội dung và không vượt parent; context đầu được thêm ngân sách nhưng các context sau vẫn có mức tối thiểu; cứu output bị cắt; mask prompt và giữ nguyên target SFT; ID/schema JSON; cache sai cấu hình bị từ chối; journal tiếp tục được sau dòng ghi dở; ba notebook dùng chung config và Dataset paths.

Trước lần train/full-dev dài, chạy quality V7 trên smoke30 rồi cùng dev100 đã dùng cho V5/V6 để so sánh đúng IDs. Kiểm tra riêng các ID 14843, 15927, 51037, 128323 và 144875; chỉ chuyển sang full dev nếu METEOR phục hồi mà không tạo regression lớn. Giữ riêng holdout theo hướng dẫn.
