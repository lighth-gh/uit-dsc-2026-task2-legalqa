# Review diagnostics V8 và kế hoạch Stage 4

Ngày review: 14/09/2026. Phạm vi: `legalqa_main_stage3_v8_diagnostics.zip`, đối chiếu submission và phần code liên quan; chưa sửa pipeline hoặc tạo notebook Stage 4.

## 1. Kết luận

Stage 3 đã hoàn tất về mặt dữ liệu và đóng gói. Vấn đề chính là chất lượng generation: lặp, sử dụng hết ngân sách token và có câu chưa trả lời đúng trọng tâm. Fallback còn có trường hợp chọn đoạn nguồn không đáp ứng câu hỏi. Nên làm Stage 4 gồm hậu xử lý CPU, đánh giá trên dev100, rồi tùy chọn sinh lại có chọn lọc bằng GPU. Không nên coi việc xóa lặp là đủ để sửa mọi câu.

## 2. Những kiểm tra đã qua

- ZIP diagnostics đọc được toàn bộ, kiểm CRC không có lỗi. Các file hiện diện trong danh sách của manifest đều khớp kích thước và SHA-256.
- Có đủ 1.000 dự đoán public, đúng schema `{id: {"answer": text}}`, không thiếu/thừa ID, không có answer rỗng; đã chạy `validate_predictions` của repo. JSON được đọc với kiểm tra khóa trùng.
- Journal có 1.000 ID duy nhất; prediction và audit từng dòng khớp file cuối. Hash prediction, identity của journal, hash questions và identity adapter khớp manifest/selection.
- Retrieval public khớp câu hỏi và ID; mỗi câu có 4 contexts không rỗng. Các parent ID trong audit đều tồn tại trong retrieval tương ứng. Điều này chứng minh dữ liệu liên kết đúng, không chứng minh retrieval đúng nội dung.
- Không trùng ID giữa train/dev/holdout; 768 mẫu SFT không trùng ID hoặc câu hỏi sau chuẩn hóa với dev100. Đây chưa phải kiểm chứng mọi dạng trùng nghĩa.
- Parameter audit ghi tổng base + LoRA là 3.801.281.793, dưới ngưỡng 4 tỷ của cấu hình. Đây là số liệu từ audit, không phải lần đếm lại tensor trong review.
- `submission.zip` trong repo giống hệt `C:\Users\HP\Downloads\submission (4).zip`.

## 3. Bất ổn có bằng chứng

### 3.1. Fine-tune cải thiện METEOR nhưng tăng lặp và chạm giới hạn token

| Mẫu/model | METEOR | ROUGE-L | Chạm giới hạn token | Bị gắn cờ lặp |
|---|---:|---:|---:|---:|
| Dev100 base | 0,46178 | 0,50844 | 1/100 | 2/100 |
| Dev100 epoch-01 | 0,56356 | 0,53082 | 20/100 | 12/100 |
| Dev100 epoch-02 | 0,54768 | 0,53352 | 23/100 | 17/100 |
| Public epoch-01 | Chưa có nhãn | Chưa có nhãn | 179/1.000 | 95/1.000 |

Định nghĩa sàng lọc lặp của review: lấy các dòng sau strip dài ít nhất 35 ký tự; tỷ lệ dòng xuất hiện thêm sau lần đầu >=25%. Đây là heuristic để tìm ứng viên, không phải quy tắc tự động xóa. 51 câu public có tỷ lệ này >=50%.

Độ dài trung bình trên dev100 tăng từ 304,47 từ ở base lên 539,02 từ ở epoch-01; reference trung bình 328,68 từ. Nhóm 20 câu epoch-01 chạm giới hạn có METEOR trung bình 0,4640 và ROUGE-L 0,3476. Chỉ số nhóm này là tương quan mô tả, không chứng minh giảm độ dài sẽ tăng điểm.

Epoch-01 là lựa chọn đúng theo quy tắc METEOR chính của dự án. Epoch-02 có nhiều dấu hiệu lặp hơn và METEOR thấp hơn; chưa có lý do chuyển checkpoint. Các số liệu không đủ để kết luận nguyên nhân huấn luyện sâu hơn hoặc khẳng định overfitting.

### 3.2. Có vòng lặp nặng; xóa lặp xong vẫn có thể chưa có đáp án

- Public `76855`: hỏi phí cấp bản sao tài liệu lưu trữ; lặp cùng câu dẫn bắt đầu bằng “Căn cứ theo khoản 8 Điều 3…” 45 lần, chưa trả lời nội dung phí.
- Public `124207`: dòng “Mẫu này được sử dụng cho người đăng ký tập sự hành nghề công chứng.” xuất hiện 84 lần.
- Public `80189`: cùng dòng mô tả thông báo thay đổi người đại diện xuất hiện 12 lần.

Code hiện tại dùng greedy generation với `repetition_penalty=1.0`; chưa có bước nhận diện vòng lặp trong `answer_flags`. Cấu hình này chưa bổ sung biện pháp chống lặp, nhưng không đủ bằng chứng để quy mọi lỗi cho riêng tham số đó.

### 3.3. Nhãn `generated_truncated` không chứng minh đáp án đã hoàn chỉnh

- 179 câu public chạm giới hạn 1.536 token; 157 được giữ theo route `generated_truncated`, 22 chuyển fallback.
- Trong 157 câu này, 22 câu cuối cùng kết thúc bằng dấu hai chấm. Ví dụ `76855` chỉ còn chuỗi câu dẫn, vẫn được chấp nhận.
- `complete_truncated_answer` cắt về dấu câu/dòng trước; regex hiện nhận cả `:` và `;`. Nó không kiểm tra đủ các ý của câu hỏi và không khôi phục được nội dung chưa sinh.
- 67/95 câu bị gắn cờ lặp cũng chạm giới hạn; 112 câu chạm giới hạn không bị heuristic lặp này gắn cờ. Cần phân biệt vòng lặp làm hết token với một đáp án dài còn thiếu phần cuối.

### 3.4. Fallback và grounding còn điểm yếu

- Có 38 câu public dùng `source_fallback`. Trong đó, 19 có `refusal_evidence_support.strong=false`; đây là tín hiệu xem xét, không đủ để kết luận cả 19 sai.
- Có 26 raw outputs bị phát hiện số hiệu văn bản không có trong evidence và được đưa qua fallback. Không được diễn giải số này thành 26 câu cuối vẫn bịa số hiệu.
- Public `118471` hỏi hệ số lương Thẩm phán/Thư ký năm 2022, nhưng final fallback trích đoạn “Đại hội đồng ấn định” mà không có hệ số cần hỏi. Cả việc xóa lặp lẫn tăng token đều không tự sửa được lỗi này.
- Dev `164143` hỏi có bị xử phạt nhiều lần khi không đội mũ bảo hiểm không; đáp án chỉ nêu mức tiền phạt, trong khi reference giải thích nguyên tắc xử phạt. Audit vẫn đánh dấu evidence support mạnh. Khớp từ khóa chưa đồng nghĩa trả lời đúng trọng tâm.
- Dev `155609` hỏi nội dung đánh giá an toàn thấm; fallback trích đoạn phân loại và báo cáo an toàn đập. Cần kiểm tra lựa chọn đoạn nguồn, không chỉ hình thức đáp án.

Các nhận xét trên đối chiếu câu hỏi, output, context và reference trong dataset; không phải rà soát hiệu lực pháp luật ngoài dataset. Chưa có nhãn relevance để đo recall retrieval toàn tập, và chưa tái dựng chính xác prompt đã pack bằng tokenizer trong review này.

### 3.5. Giới hạn đánh giá và dữ liệu đầu vào Stage 4

- METEOR 0,56356 vẫn thấp hơn mục tiêu 0,65. Không có cơ sở hứa hậu xử lý sẽ đạt 0,65.
- Dev100 đã dùng chọn checkpoint. Nếu tiếp tục thử nhiều quy tắc trên cùng tập này sẽ tăng nguy cơ chọn cấu hình quá hợp tập; nên giới hạn số phương án và khóa quy tắc trước xác nhận độc lập.
- Diagnostics có dữ liệu holdout nhưng không có prediction/metrics holdout. Không thể chấm holdout chỉ bằng ZIP hiện tại.
- Manifest có `selected_adapter/adapter_model.safetensors` nhưng ZIP không chứa file đó. Code đóng diagnostics chỉ đưa JSON/JSONL/TXT vào ZIP: đây là chủ ý, không phải lỗi hỏng file.
- Review chưa chạy lại scorer vì Python local thiếu `nltk`. Điểm được đọc từ báo cáo và dùng per-question metrics để tổng hợp nhóm; chưa có kết quả đo tác động của bất kỳ sửa chữa nào.

## 4. Thiết kế Stage 4 đề xuất

Tên notebook dự kiến: `legalqa_main_04_repair_submit.ipynb`.

Phần logic nên nằm trong module riêng, ví dụ `legalqa/repair.py`; notebook quản lý input, cấu hình, chạy từng bước và hiển thị báo cáo. Chưa cần thay đổi Stage 1–3, retrieval settings hoặc cách chọn adapter.

### Bước 1 — Nạp và xác minh artifact (CPU)

Input tối thiểu: diagnostics ZIP. Đọc prediction public/dev100, reference dev100, audit, retrieval, manifest và selection. Dùng prediction cuối làm bản gốc; không lấy `.partial.json` làm đầu ra.

Kiểm CRC, các hash hiện diện, ID/schema, identity và sự khớp giữa audit/journal/prediction. Phân biệt file trọng số bị loại có chủ ý khỏi diagnostics với file JSON cần thiết bị thiếu. Dừng khi các input cần thiết sai hoặc không nhất quán.

Input thêm cho chế độ GPU: generator/tokenizer và selected adapter từ output/model dataset Kaggle; kiểm tra đúng revision và hash adapter. Diagnostics một mình đủ cho hậu xử lý văn bản CPU, không đủ để sinh lại.

### Bước 2 — Phân loại ứng viên (CPU)

Lưu mọi lý do theo ID, không cộng số nhóm chồng lấp:

| Nhóm sơ bộ | Số lượng public |
|---|---:|
| Gắn cờ lặp | 95 |
| Chạm giới hạn token | 179 |
| Fallback | 38 |
| Hợp của ba nhóm | 223 |

223 là danh sách sàng lọc ban đầu, không phải 223 câu bắt buộc phải sửa hoặc sinh lại; cũng không bao phủ mọi lỗi ngữ nghĩa. Các phép kiểm tra nội dung bổ sung chỉ mở rộng danh sách khi có lý do ghi nhận rõ.

Chia hướng xử lý: (a) có khối lặp nhưng phần còn lại đủ đáp án; (b) sau bỏ lặp chỉ còn câu dẫn/thiếu ý; (c) chạm token nhưng có vẻ đã đủ ý; (d) fallback hoặc context không đáp ứng trọng tâm; (e) chưa đủ bằng chứng để sửa tự động.

### Bước 3 — Sửa lặp bảo thủ (CPU)

- Bắt đầu bằng các câu/khối dài trùng nguyên văn liên tiếp; giữ thứ tự phần thông tin duy nhất. Heuristic đếm dòng chỉ dùng sàng lọc.
- Không dùng độ giống nghĩa để xóa tự động; bảo toàn số tiền, ngày tháng, số hiệu, điều/khoản, từ phủ định và các mục danh sách khác nhau.
- Sau sửa kiểm tra phần còn lại có nội dung trả lời hay chỉ là câu dẫn. Trường hợp như `76855` phải chuyển sang bước cần bổ sung đáp án, không xuất một dòng dẫn đã “hết lặp”.
- Không cắt mọi answer xuống một giới hạn từ cố định. Kết thúc bằng dấu chấm cũng không được coi là bằng chứng đáp án đầy đủ.
- Lưu bản trước/sau, lý do sửa, số ký tự/từ bị loại và các cờ chất lượng; giữ nguyên bản gốc để đối chiếu/rollback.

### Bước 4 — Đánh giá trước khi áp dụng public

Tái lập scorer BTC và baseline epoch-01 từ dev100 trước; xác minh score khớp báo cáo trong sai số số thực. Sau đó áp dụng cùng quy tắc sửa cho toàn bộ dev100 và chấm lại.

Giới hạn phương án thử. Báo cáo METEOR/ROUGE-L toàn tập, từng câu, nhóm bị sửa và các trường hợp giảm mạnh. Không dùng gold để chọn riêng bản gốc/bản sửa cho từng ID; gold chỉ phục vụ đánh giá quy tắc chung.

Tiêu chí nhận mặc định: METEOR toàn tập không giảm so với baseline, lỗi lặp nặng giảm, không tạo lỗi schema/rỗng/mất ý đã biết. ROUGE-L và các ví dụ giảm điểm phải được báo cáo, không âm thầm đổi tiêu chí chính của repo. Nếu không đạt, giữ bản gốc làm bản được chọn và xuất báo cáo thử nghiệm.

Dev100 là kiểm tra phát triển, không phải bằng chứng độc lập. Nếu thực hiện vòng sửa bằng GPU, cân nhắc xác nhận một mẫu holdout được xác định trước sau khi khóa quy tắc; việc này cần retrieval/generation mới vì chưa có predictions holdout.

### Bước 5 — Sinh lại có chọn lọc (GPU, tùy chọn)

Chỉ chạy với các ID còn lỗi sau bước CPU và có evidence phù hợp. Trước hết kiểm tra/chọn lại đoạn liên quan trong các contexts đã cache; giữ nguyên dữ kiện pháp lý của nguồn.

Với trường hợp context cache thiếu nội dung cần hỏi, đưa vào danh sách chưa giải quyết. Tìm kiếm lại corpus/index là phần mở rộng riêng, không tự mở rộng Stage 4 thành xây lại retrieval. Tuyệt đối không viết thêm đáp án thiếu căn cứ hoặc dùng gold làm context.

Thử trên khoảng 10–20 câu dev đã chọn trước để có cả lỗi lặp, thiếu ý và fallback. Tinh chỉnh nhẹ kiểm soát lặp/điều kiện dừng; chỉ tăng token khi bằng chứng cho thấy câu dài thực sự thiếu ý. Giữ ngân sách context của model và ghi cấu hình thực tế. Chỉ triển khai rộng hơn sau đánh giá.

Dùng journal Stage 4 riêng có ID, hash input, cấu hình sửa và identity model/adapter; resume đúng phần chưa làm. Không sửa identity của checkpoint Stage 3 để ép dùng lại cache. Đặt giới hạn thời gian/số lần thử; lỗi hoặc không cải thiện thì giữ bản gốc và ghi unresolved.

### Bước 6 — Đóng gói và báo cáo

Output trong `/kaggle/working/legalqa_main_stage4_v8/`:

- `submission.original.json`: bản gốc đối chiếu.
- `submission.repaired.json`: bản ứng viên sau sửa, cùng schema.
- `submission_repaired.zip`: chỉ chứa `submission.json` ở gốc, khi ứng viên đạt các điều kiện nhận.
- `repair.audit.json`, `repair.metrics.json`, `repair.manifest.json`: thay đổi, kết quả chấm, quyết định chọn và nguồn gốc.
- `repair.unresolved.json`: các ID còn thiếu bằng chứng hoặc sửa không thành công.
- Journal riêng nếu bật sinh lại bằng GPU.

Xác minh lại 1.000 ID duy nhất, không rỗng, tên file ZIP, đọc lại ZIP và hash JSON. ID không được chọn sửa phải có answer giống bản gốc. Báo cáo riêng số ứng viên, số sửa CPU, số sinh lại, số giữ gốc và số chưa giải quyết.

## 5. Kiểm thử cần có khi triển khai

- Hành vi xóa một khối lặp dài thực sự, kể cả khối nhiều dòng.
- Không xóa hai mục gần giống nhưng khác con số, phủ định hoặc điều kiện.
- Câu dẫn lặp kiểu `76855` phải bị giữ trong hàng đợi cần xử lý sau xóa lặp.
- Không gắn nhãn hoàn chỉnh chỉ vì dừng ở `:`/`;`; không mặc định mọi câu chạm token đều sai.
- Không đưa raw output đã bị guard loại trở lại submission khi chỉ sửa văn bản final.
- Journal riêng resume đúng và từ chối identity khác; giữ nguyên ID ngoài tập sửa.
- Tái lập baseline scorer, chấm dev100 trước/sau và kiểm schema/ZIP cuối.

## 6. Trình tự thực hiện nên chọn

Làm phần CPU và báo cáo dev100 trước. Chỉ bổ sung GPU selective repair khi có model/adapter đúng và phần CPU cho thấy còn lỗi cần sinh lại. Giữ epoch-01, không huấn luyện lại hoặc chạy lại toàn bộ public ở bước đầu. Các vấn đề retrieval/chunking rộng hơn được ghi nhận riêng, chưa nằm trong sửa chữa tự động Stage 4.
