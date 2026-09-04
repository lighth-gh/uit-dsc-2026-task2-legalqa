# Các lỗi cần sửa

Tài liệu này phản ánh kết quả smoke 30 từ run `e24a482a461f-226bece3139d`.
Không hạ ngưỡng gate hoặc đổi tên route chỉ để làm báo cáo PASS; mỗi mục chỉ được
đánh dấu hoàn tất sau khi có test và một lần chạy kiểm chứng phù hợp.

## Kết quả hiện tại

- Smoke automatic gate: **FAIL**.
- Thời gian: median `11,947 giây/câu` — đạt ngưỡng `< 15,5 giây`.
- Token-limit cuối: `5/30 = 16,67%` — không đạt ngưỡng `< 5%`.
- Refusal cuối: `8/30 = 26,67%` — không đạt `no_refusal` và ngưỡng `< 2%`.
- Extractive fallback: `4/30 = 13,33%` — không đạt ngưỡng `< 10%`.
- Schema, ID, output cleaning, heading-only, possibly-cut và độ dài đều đạt.
- Public retrieval, validation 100 và validation 300 chưa chạy trong run này.

## Danh sách lỗi theo thứ tự xử lý

| Ưu tiên | Trạng thái | Tầng lỗi | Bằng chứng | Hướng sửa nhỏ nhất |
|---|---|---|---|---|
| P0 | IMPLEMENTED / SMOKE PENDING | Token-limit recovery | Sáu câu hit giới hạn ban đầu nhưng `generated_retry_768_ids=[]`; năm câu vẫn lỗi: `80189`, `63093`, `55463`, `67397`, `42039` | Retry đã giới hạn 360 từ; exception giữ partial để audit; token failure dùng focused extractive khi toàn cửa sổ có evidence quyết định, kể cả raw score thấp. Chờ smoke model thật. |
| P0 | IMPLEMENTED / SMOKE PENDING | Refusal recovery | `18645`, `129215`, `6905` kết thúc ở `recovery_exhausted` | Yes/no chỉ fallback bằng điều khoản khớp mạnh, không tự tạo kết luận Có/Không; câu đếm bước dùng chuỗi chunk kề có exact evidence. Chờ smoke model thật. |
| P0 | PARTIAL / KAGGLE REQUIRED | Retrieval verification | `RUN_PUBLIC_RETRIEVAL=False`, nên expected Top-1 và nguyên nhân anchor yếu chưa được kiểm chứng trên model/index thật | Đã đối chiếu corpus và thêm exact-priority hẹp cho `42039`, alias + priority hẹp cho `55463`; vẫn phải chạy retrieval-only 12 ID trên Kaggle trước khi chỉnh ranking rộng. |
| P1 | PARTIAL / SMOKE PENDING | Fallback rate | `34235`, `117399`, `108017`, `138443` dùng `extractive_fallback`, tỷ lệ 13,33% | Bộ nhận diện structured extractive đã thêm “biện pháp”, “quy định” và “bao nhiêu bước”. Không nới ngưỡng; chờ đo route thật. |
| P1 | DONE LOCAL | Integration regressions | Unit test pass nhưng hành vi model thật vẫn fail retry/refusal | Đã thêm full-route regression cho token-limit lặp lại, partial output, yes/no strong/weak evidence, scalar token-limit và `129215`. |
| P1 | BLOCKED / USER CHANGE | Notebook contract | Full discovery còn 5 lỗi đọc file sau commit `5824bb3` (`chore: remove test notebooks`) xóa các notebook mà `tests/test_notebook_contract.py` vẫn kiểm tra | Không tự phục hồi hoặc làm yếu contract. Cần chọn khôi phục notebook smoke/release-gate hoặc chuyển contract sang một entrypoint Python được track. |
| P1 | BLOCKED | Validation | Validation 100/300 bị SKIPPED vì smoke chưa PASS | Chỉ chạy validation 100 sau smoke PASS; chạy validation 300 và so metric sau validation 100 PASS. |
| P2 | PARTIAL / SMOKE PENDING | Tail latency | Các câu retry lỗi mất khoảng 63–106 giây nhưng vẫn không có đáp án dùng được | Structured/step-count có thể bypass generation; retry còn 360 từ. Giữ checkpoint và chờ đo lại p50/p90/median. |
| P2 | PENDING | Manual review | 26 ID chưa được duyệt thủ công | Duyệt relevance, tính đầy đủ và căn cứ sau khi smoke tự động PASS. |

## Phân nhóm ID cần kiểm chứng

### Token-limit

- Chưa cứu được: `80189`, `63093`, `55463`, `67397`, `42039`.
- Đã chuyển được sang extractive fallback nhưng retry vẫn thất bại: `138443`.

### Refusal

- Refusal recovery thất bại độc lập: `18645`, `129215`, `6905`.
- Năm refusal còn lại chính là các câu token-limit chưa cứu được, không phải một
  nhóm nguyên nhân riêng.

### Extractive fallback

- Refusal được cứu: `34235`, `117399`, `108017`.
- Token-limit được cứu: `138443`.

## Đối chiếu corpus và quyết định sửa

- `55463`: corpus xác nhận `context_58283`, Thông tư `17/2021/TT-BCA`, có đúng
  mục “Trình tự báo cáo và cơ quan tiếp nhận báo cáo”. Đã thêm alias chỉ kích
  hoạt khi câu hỏi đồng thời chứa “báo cáo” và đầy đủ khái niệm “phương tiện
  phòng cháy chữa cháy”; câu hỏi PCCC khác không bị mở rộng.
- `42039`: corpus xác nhận `context_235672`, Điều 42 có tiêu đề trùng nguyên văn
  “Sử dụng Quỹ bảo hiểm tai nạn lao động, bệnh nghề nghiệp”. Đã thêm cụm
  exact-priority; không thêm alias suy diễn.
- `6905`: corpus có điều khoản trực tiếp về người nhận thừa kế quyền sử dụng đất
  tiếp tục thực hiện nghĩa vụ trả nợ tiền sử dụng đất. Grounded-clause fallback
  có thể dùng điều khoản này nhưng không tự tạo kết luận ngoài văn bản.
- `18645`: chưa tìm thấy quy định chung trong Luật Nghĩa vụ quân sự cấm hình xăm;
  các quy định tìm thấy thuộc phạm vi tuyển Công an hoặc tuyển sinh quân sự.
  Giữ `recovery_exhausted` khi retrieval yếu, không hard-code câu trả lời “Không”.

## Kế hoạch triển khai và gate dừng

1. **PARTIAL — Chẩn đoán retrieval 12 ID**: lưu Top-50 RRF, Top-20 reranker, Top-3,
   raw score, guardrail bonus, chunk liền kề và lý do safe-extractive bị từ chối.
2. **DONE LOCAL — Sửa token-limit recovery** và thêm test.
3. **DONE LOCAL — Sửa refusal recovery** theo hai nhóm yes/no và count/process,
   rồi thêm test.
4. **BLOCKED KAGGLE — Chạy targeted 12 ID**, sau đó chạy lại smoke 30.
5. **PENDING — Validation 100** chỉ chạy khi smoke đạt: `0 refusal`, tối đa `1` final
   token-limit, tối đa `2` extractive fallback, không output bẩn/cắt và median
   `< 15,5 giây`.
6. **PENDING — Validation 300/full 1.000**: chỉ sang validation 300 khi
   validation 100 PASS; chỉ chạy full 1.000 khi
   validation 300 không giảm METEOR và ROUGE-L so với baseline tương thích.

## Các lỗi cũ đã xử lý, không mở lại nếu không có regression

- Output Markdown/URL/slug/số văn bản giả: smoke mới không còn vi phạm.
- Heading-only và ghép câu quá dài: smoke mới không còn vi phạm.
- Reranker guardrail cho `80189`: đã có code và unit test; vẫn cần retrieval gate
  trên index/model thật để xác nhận runtime.
- Pipeline đã dùng `hybrid_rag`, Dense và reranker đều active trong smoke mới.

## Nhật ký hoàn tất

- Đã cập nhật `GenerationTokenLimitReached` để giữ partial answer phục vụ kiểm tra,
  nhưng pipeline không tự dùng partial chưa được xác minh.
- Đã thêm focused token-limit fallback cho evidence quyết định raw score thấp; câu
  yếu vẫn giữ `recovery_exhausted` thay vì dump context.
- Đã thêm grounded clause fallback cho yes/no sau khi KNN/focused/alternate
  generation đều từ chối; không tự suy diễn tiền tố Có/Không.
- Đã thêm route extractive hẹp cho câu hỏi đếm bước khi chunk kề chứa exact evidence.
- Đã thêm exact retrieval priority cho Điều 42 (`42039`) và controlled alias cho
  trình tự/cơ quan nhận báo cáo phương tiện PCCC (`55463`).
- Test mục tiêu: `79/79 PASS`.
- Toàn bộ test code (không gồm notebook contract): `185/185 PASS`.
- Full discovery: `191` test, còn `5` lỗi contract vì commit hiện tại `5824bb3`
  đã xóa notebook nhưng test contract vẫn tham chiếu hai notebook smoke; đây
  không phải regression từ patch pipeline.
- Còn bắt buộc: retrieval-only 12 ID và smoke 30 trên Kaggle với cache thật.
