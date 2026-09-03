# Các lỗi cần sửa

Danh sách này ghi lại các lỗi và hướng xử lý do người dùng cung cấp. Các mục chưa
được tự động triển khai cho đến khi có chỉ dẫn tiếp theo.

| Ưu tiên | Tầng lỗi | Bằng chứng từ audit | Hướng sửa |
|---|---|---|---|
| P0 | Reranker guardrail | ID `80189`: đoạn đúng raw rank 1 nhưng guardrail đẩy đoạn sai lên final rank 1 | Giảm bonus heading chung; tăng trọng số exact form/long phrase; thêm kiểm tra chênh lệch raw score. |
| P0 | Chọn chunk/extractive | `31969`, `123257` trả lời chỉ có tiêu đề; nhiều câu ghép 3 chunk thành 800–1.600 từ | Phát hiện heading-only, chọn chunk kề theo relevance, giới hạn độ dài và dừng ở ranh giới điều/mục. |
| P0 | Fallback sai chiến lược | 239 fallback gồm 169 token-limit và 70 refusal nhưng đang xử lý gần giống nhau | Token-limit thì retry với budget lớn hơn; refusal thì re-retrieve/KNN, không dump văn bản thô. |
| P0 | Không có validation | Notebook đang để `RUN_RETRIEVAL_EVAL=False`, `RUN_VALIDATION=False` | Bật validation cố định trước mọi full run. |
| P1 | Low-confidence retrieval | 202/1.000 câu có raw reranker score `<2`; 46/70 refusal nằm trong nhóm này | Dùng `<2` làm tín hiệu re-query hoặc KNN; không dùng làm ngưỡng trả lời extractive trực tiếp. |
| P1 | KNN bị bỏ qua | Pipeline chạy `mode=rag`; một số câu trùng/gần trùng train có thể lấy đáp án chuẩn | Chuyển sang `hybrid_rag` có guard, threshold khoảng `0.90`, kiểm tra intent/entity. |
| P1 | Output bẩn | 310 câu có Markdown, 37 câu lộ slug, 16 câu biến page ID thành “số luật” giả | Chuẩn hóa output và chỉ cho phép citation lấy từ metadata pháp luật. |
| P2 | Query mơ hồ | `ZONE`, “giáo dục của giáo viên”, một số câu dùng thuật ngữ dễ lệch nghĩa | Query normalization/alias có kiểm soát, không mở rộng đại trà. |

## Ghi chú trạng thái

- Mục P0 “Không có validation” đã được xử lý trong working tree hiện tại: notebook
  bật retrieval evaluation và validation, dùng split cố định 100 → 300.
- Mục P0 “Reranker guardrail” đã được triển khai ở mức code và kiểm thử hồi quy:
  heading bonus bị giới hạn, exact form/Điều/văn bản/long phrase được tăng trọng số,
  raw exact leader có cơ chế chống heuristic lật hạng khi score sát nhau hoặc cùng âm,
  còn exact-focus của `34235` được giữ nguyên. Cổng runtime trên model/index thật vẫn là
  5 expected chunks phải đạt final Top 1.
- Các mục còn lại đang chờ chỉ dẫn triển khai.
