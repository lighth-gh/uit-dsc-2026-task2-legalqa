# Các lỗi cần sửa

Tài liệu này phản ánh smoke 30 cũ `e24a482a461f-226bece3139d`, lần targeted
12 câu tại commit `9a0b19f8062f`, và lần targeted mới tại commit
`3879a7c1ac6e` (submission/checkpoint/audit/log).
Không hạ ngưỡng gate hoặc đổi tên route chỉ để làm báo cáo PASS; mỗi mục chỉ được
đánh dấu hoàn tất sau khi có test và một lần chạy kiểm chứng phù hợp.

## Kết quả targeted 12 tại `3879a7c1ac6e` (trước patch hiện tại)

- Unit gate trong notebook: **120/120 PASS**.
- Retrieval known-target gate: **FAIL**, chỉ còn `129215` sai Top-1.
- Token-limit cuối: **0/12**; năm lỗi token-limit cũ đã được sửa.
- Refusal/invalid cuối: **2/12** — `129215`, `6905`.
- Route: `6 extractive_long`, `4 extractive_fallback`, `2 recovery_exhausted`.
- Tổng thời gian predict: `116,82 giây`; chưa dùng tập targeted thiên lệch này để
  ngoại suy runtime 1.000 câu.
- `138443` đã lấy đúng phần dự phòng lây nhiễm SARS-CoV-2; `18645` đã trả lời có
  điều kiện thay vì refusal.
- Submission có đủ 12 ID; `checkpoint.predictions` khớp submission và audit có
  12 ID duy nhất.

## Kết quả targeted 12 tại `9a0b19f8062f`

- Unit gate trong notebook: **114/114 PASS**; retrieval known-target gate: **PASS**.
- Kết luận generation: **NOT_FIXED_YET**.
- Token-limit cuối: `5/12` — `80189`, `63093`, `55463`, `67397`, `42039`.
- Refusal cuối: `1/12` — `129215`; Top-1 đang nhầm bảng giá thú y `context_178654`.
- Extractive fallback: `5/12` — `18645`, `6905`, `34235`, `117399`, `108017`.
- Tổng thời gian generation `362,5285 giây`; tổng pipeline `416,1166 giây`.
- `138443` qua gate hình thức nhưng duyệt tay phát hiện sai trọng tâm: câu hỏi phòng lây
  nhiễm COVID-19, câu trả lời cũ lại lấy bảng dự phòng bạo hành/quấy rối.

## Kết quả smoke 30 gần nhất

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
| P0 | VERIFIED TARGETED | Safe extractive trả rỗng | Targeted `3879a7c` không còn final token-limit ở cả 5 ID cũ | Giữ regression hiện có; không mở lại nếu smoke 30 không phát hiện regression. |
| P0 | FIXED LOCAL / TARGETED RERUN REQUIRED | Retrieval `129215` | BM25, dense và RRF đều xếp `8035/11` hạng 1, nhưng guardrail cho 1 từ chung `PRRS` cùng bonus `4.0` như 3 cụm exact nên reranker lật sang bảng giá `178654/14` | Giữ `PRRS` làm query alias nhưng bỏ khỏi exact-priority; bonus nhiều cụm exact tăng có giới hạn `4..6`. Mô phỏng lại 20 candidate thật đưa `8035/11` lên Top-1 (`-0,2266` so với `-2,3296`). |
| P0 | VERIFIED TARGETED | Relevance `138443` | Targeted `3879a7c` lấy đúng `44451/11` và nội dung dự phòng lây nhiễm SARS-CoV-2 | Giữ semantic expected target trong notebook. |
| P0 | FIXED LOCAL / TARGETED RERUN REQUIRED | Grounded fallback `6905` | Top-1 đúng `184038/33`, nhưng original-query coverage chỉ `0,40`; alias coverage `0,80` không được dùng nên hai lần generation refusal kết thúc bằng “Không đủ thông tin” | Chỉ cho alias-aware fallback khi alias có ít nhất 4 term, coverage `>=0,75` và candidate có exact evidence; trả nguyên câu điều kiện hoàn chỉnh, không tự thêm Có/Không. |
| P1 | FIXED LOCAL / SMOKE PENDING | Fallback quality | `18645`, `34235`, `117399`, `108017` dùng `extractive_fallback` nhưng đều hợp lệ; `6905` đã được sửa local | Không coi fallback là lỗi chỉ vì route; duyệt relevance và đo rate trên smoke 30. |
| P1 | DONE LOCAL | Integration regressions | Lỗi chỉ xuất hiện trên corpus thật dù unit cũ PASS | Đã thêm regression cho legal-item boundary, inline article citation, next numeric heading, alias PRRS/COVID, ranking noise và route low-score có evidence. |
| P1 | BLOCKED / USER CHANGE | Notebook contract | Full discovery còn 5 lỗi đọc file do commit `5824bb3` (`chore: remove test notebooks`) đã xóa các notebook mà `tests/test_notebook_contract.py` vẫn kiểm tra | Không tự phục hồi hoặc làm yếu contract. Cần chọn khôi phục notebook smoke/release-gate hoặc chuyển contract sang một entrypoint Python được track. |
| P1 | BLOCKED | Validation | Validation 100/300 bị SKIPPED vì smoke chưa PASS | Chỉ chạy validation 100 sau smoke PASS; chạy validation 300 và so metric sau validation 100 PASS. |
| P2 | FIXED LOCAL / MEASURE PENDING | Tail latency | 5 token-limit tiêu tốn `325,29 giây` generation; riêng `80189` mất `103,67 giây` | Bốn câu structured raw-score tốt và `67397` controlled exact evidence bypass generation; `129215`, `138443` cũng có route extractive hẹp. Đo lại targeted và smoke 30. |
| P2 | PENDING | Manual review | 26 ID chưa được duyệt thủ công | Duyệt relevance, tính đầy đủ và căn cứ sau khi smoke tự động PASS. |

## Phân nhóm ID cần kiểm chứng

### Token-limit

- Runtime cũ chưa cứu được: `80189`, `63093`, `55463`, `67397`, `42039`.
- Local reconstruction sau patch: cả 5 đều có extractive không rỗng, không bị đánh dấu cắt;
  còn bắt buộc xác nhận bằng targeted model thật.

### Refusal

- Refusal cuối trong targeted: `129215`; local patch đã tìm đúng Phụ lục D và trả lời 10 bước.
- `18645`, `6905` đã thoát refusal bằng grounded clause nhưng còn rủi ro chất lượng thủ công.

### Extractive fallback

- Refusal được cứu: `18645`, `6905`, `34235`, `117399`, `108017`.
- `138443` hiện đi `extractive_long`, nhưng output tại `9a0b19f` sai section và đã có patch retrieval.

## Đối chiếu corpus và quyết định sửa

- `55463`: corpus xác nhận `context_58283`, Thông tư `17/2021/TT-BCA`, có đúng
  mục “Trình tự báo cáo và cơ quan tiếp nhận báo cáo”. Đã thêm alias chỉ kích
  hoạt khi câu hỏi đồng thời chứa “báo cáo” và đầy đủ khái niệm “phương tiện
  phòng cháy chữa cháy”; câu hỏi PCCC khác không bị mở rộng.
- `42039`: corpus xác nhận `context_235672`, Điều 42 có tiêu đề trùng nguyên văn
  “Sử dụng Quỹ bảo hiểm tai nạn lao động, bệnh nghề nghiệp”. Đã thêm cụm
  exact-priority; không thêm alias suy diễn.
- `129215`: corpus xác nhận `context_8035`, “PHỤ LỤC D (Tham khảo) Phương pháp
  ELISA phát hiện kháng thể PRRS”; chuỗi nhãn đi đến `Bước 10` (nguồn OCR thiếu nhãn
  Bước 2 nhưng các bước còn lại liên tục). Trả lời deterministic chỉ khi cửa sổ chứa
  Bước 1 và gần đủ chuỗi đến bước cuối.
- `138443`: `context_44451` có đúng mục “1. DỰ PHÒNG LÂY NHIỄM SARS-COV-2” tại
  chunk 11 và các checklist đúng chủ đề tại chunk 28/34/35; chunk 19 là mục rủi ro
  bạo hành/quấy rối và không được dùng làm Top-1 nữa.
- `6905`: corpus có điều khoản trực tiếp về người nhận thừa kế quyền sử dụng đất
  tiếp tục thực hiện nghĩa vụ trả nợ tiền sử dụng đất. Grounded-clause fallback
  được focus bằng hai cụm “ghi nợ nghĩa vụ tài chính” và “phải thực hiện xong
  nghĩa vụ tài chính trước khi thực hiện các quyền”; không tự tạo kết luận ngoài văn bản.
- `18645`: chưa tìm thấy quy định chung trong Luật Nghĩa vụ quân sự cấm hình xăm;
  các quy định tìm thấy thuộc phạm vi tuyển Công an hoặc tuyển sinh quân sự.
  Sau refusal, chỉ dùng câu trả lời phạm vi hẹp khi extract đầy đủ tiêu chuẩn tuyển quân
  có “Không gọi nhập ngũ…”, “Bộ Quốc phòng” và không chứa quy định về xăm; không
  hard-code câu trả lời tuyệt đối “Không”.

## Kế hoạch triển khai và gate dừng

1. **DONE — Chẩn đoán targeted 12 tại `9a0b19f`** bằng đủ Top-50/RRF/reranker/audit.
2. **DONE TARGETED — Sửa safe extractive và `138443`** tại `3879a7c`.
3. **DONE LOCAL — Sửa post-rerank `129215` và grounded fallback `6905`**, thêm
   regression và đối chiếu lại trên candidate/chunk thật.
4. **NEXT — Commit/push rồi chạy lại notebook targeted 12**. Bắt buộc:
   `0` final token-limit, `0` refusal, Top-1 `129215=8035/11`,
   `138443=44451/{11,28,34,35}`, `6905=184038/33`, `18645=297709/4`,
   không answer rỗng/heading/cắt.
5. **PENDING — Chạy smoke 30** chỉ sau khi targeted 12 PASS.
6. **PENDING — Validation 100** chỉ chạy khi smoke đạt: `0 refusal`, tối đa `1` final
   token-limit, tối đa `2` extractive fallback, không output bẩn/cắt và median
   `< 15,5 giây`.
7. **PENDING — Validation 300/full 1.000**: chỉ sang validation 300 khi
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
- Đã cập nhật notebook `legalqa-targeted-fixes-smoke.ipynb`: chạy bốn regression suite
  (`120` test, dùng discovery tương thích Kaggle/Python 3.12), bổ sung expected Top-1
  cho `129215` và `138443`,
  retrieval-only và generation đúng 12 ID lỗi; kết luận chỉ mở gate smoke 30,
  không tự cho phép chạy full 1.000.
- Đã build BM25 tạm từ toàn bộ corpus thật (`8.532` văn bản, `246.856` chunks):
  `129215` lên Top-1 `8035/11`, `6905` lên Top-1 `184038/33`, `67397` lên
  Top-1 `261171/6`; `138443` chỉ còn các chunk đúng chủ đề của `44451` trong Top-5.
- Test liên quan trực tiếp sau patch: `148/148 PASS` (baseline, storage, routing,
  generator, dense RAG); bốn suite trong notebook: `120/120 PASS`.
- Full discovery hiện tại: `199` test, `194` pass và còn `5` lỗi contract vì
  commit `5824bb3` đã xóa
  notebook nhưng test contract vẫn tham chiếu hai notebook smoke; đây
  không phải regression từ patch pipeline.
- Patch sau targeted `3879a7c`: bỏ `PRRS` đơn lẻ khỏi exact-priority, tính bonus
  bounded theo số cụm đặc hiệu, và thêm alias-aware grounded clause cho `6905`.
  Năm suite targeted chạy local **150/150 PASS**; đối chiếu corpus thật
  cho kết quả `129215=8035/11` và câu `6905` còn đúng một điều khoản 37 từ.
- Notebook targeted đã thêm suite `test_dense_rag.py`, patch markers cho hai sửa
  mới và in trực tiếp danh sách retrieval mismatch ID.
- Còn bắt buộc: targeted 12 ở commit mới, sau đó smoke 30 trên Kaggle với cache thật.
