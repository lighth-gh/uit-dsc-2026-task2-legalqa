# Phạm vi kiểm định bộ code

Kiểm định tại môi trường tạo file chỉ bao phủ phần chạy CPU và các contract của pipeline. Đây không phải chứng nhận điểm thi hoặc xác nhận đã chạy thành công trên T4.

| Kiểm định | Trạng thái |
| --- | --- |
| Đối chiếu ba model với Excel được gửi | Đã thực hiện |
| Tính tổng tham số từ cấu hình kiến trúc đầy đủ | Đã thực hiện; 3.801.281.793 kể cả LoRA mặc định |
| So sánh vendor scorer với ZIP gốc | Kiểm tra byte/hash khi đóng gói |
| 18 unit tests, gồm đường đi prepare → schema → submission ZIP | 18/18 qua trên CPU |
| Biên dịch cú pháp toàn bộ code và code cells notebook | Kiểm tra khi đóng gói |
| Khôi phục checkpoint bị ngắt và chặn cache sai cấu hình | Đã chạy CPU |
| Mã chấm BTC chạy thật với NLTK/WordNet | Chưa chạy ở môi trường tạo file; notebook có bước kiểm chứng metric |
| Đếm tham số bằng model PyTorch trên meta device | Có lệnh tự động; chưa chạy tại đây vì không có PyTorch |
| Download và load đúng ba checkpoint thật | Chưa chạy tại đây |
| Huấn luyện QLoRA và generation trên Kaggle T4 | Chưa chạy tại đây |
| METEOR/ROUGE-L của pipeline mới, tốc độ và VRAM | Chưa có kết quả thực nghiệm |

Các kiểm định quan trọng bao gồm không lẫn câu hỏi/đáp án trùng qua split; không lấy số trang web làm số luật; cửa sổ child phủ nội dung và không vượt parent; mở rộng quanh hit ở cuối văn bản; mask prompt và giữ nguyên target SFT; ID và schema JSON; snapshot khác cấu hình bị từ chối; journal tiếp tục được sau dòng ghi dở.

Trước lần train dài, chạy cell smoke trên 30 câu với trọng số thật và đọc vài answer cùng audit. Smoke phát hiện vấn đề vận hành; số liệu 30 câu không đủ để kết luận chất lượng. Sau smoke, đo trên dev cố định và giữ riêng holdout theo hướng dẫn.
