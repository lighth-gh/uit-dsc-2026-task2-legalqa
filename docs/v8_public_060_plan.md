# Kế hoạch LegalQA v8: public 0.5599 → mục tiêu 0.60

Ngày phân tích: 18/09/2026. Phạm vi: phân tích ba ZIP diagnostics stage 1–3 do người dùng cung cấp và lập kế hoạch; chưa sửa pipeline, train lại hoặc tạo submission mới.

Người dùng xác nhận **0.5599 là điểm sau stage 4/repair**. Vì vậy không được gán điểm này cho stage 3, hoặc coi các lỗi trước repair là toàn bộ dư địa cải thiện còn lại. Cần đối chiếu đúng submission và recipe repair trong bước đầu.

## 1. Kết luận để ra quyết định

Giữ epoch 1 làm mốc. Ưu tiên lần lượt: xác lập baseline sau repair → giảm sinh sai nguồn/lặp trên các câu còn lỗi → cải thiện lựa chọn và đóng gói evidence → sửa retrieval có mục tiêu → chỉ train lại sau khi đã đo hiệu quả các bước trước.

Mục tiêu cần tăng tuyệt đối **0.0401**, khoảng **7.16% tương đối**. Diagnostics xác định được các hướng thử nghiệm, nhưng không chứng minh một cấu hình nào chắc chắn đạt public 0.60. Local METEOR và public score không được coi là tương đương nếu chưa xác nhận metric leaderboard và đúng artifact.

## 2. Bằng chứng từ ba ZIP

Nguồn:

- `legalqa_main_stage1_v8_diagnostics (4).zip`: config, data report, training data report, training manifest, checkpoint metadata.
- `legalqa_main_stage2_v8_diagnostics (4).zip`: selection, ba bộ metrics/predictions/audit dev100, retrieval dev100/public.
- `legalqa_main_stage3_v8_diagnostics (6).zip`: progress, checkpoint và partial submission public.

| Bản trên cùng dev100 | METEOR | ROUGE-L | Chạm trần sinh | Số từ trung bình |
|---|---:|---:|---:|---:|
| Base | 0.461776 | 0.508439 | 1/100 | 304.47 |
| Epoch 1, được chọn | **0.618283** | **0.590862** | 12/100 | 488.01 |
| Epoch 2 | 0.602641 | 0.589332 | 7/100 | 420.79 |
| Tham chiếu | — | — | — | 328.68 |

**Stage 1:** 5.600 ví dụ train, dev 700, holdout 700; dev100 chỉ là tập chọn checkpoint nhỏ. Train dùng retrieval lexical, prompt tối đa 2.048 token. Inference dùng full retrieval và prompt tối đa 4.096 token. Đây là độ lệch cấu hình có thật; tác động đến điểm còn phải ablation. Chỉ 33/5.600 đáp án train dài hơn 1.536 token, 7/5.600 dài hơn 2.048; không có cơ sở tăng output budget cho mọi câu.

**Stage 2:** Epoch 1 hơn base 0.156507 METEOR, nên giữ lợi ích SFT. Epoch 2 thấp hơn epoch 1 0.015642; paired bootstrap 10.000 lượt, seed 2026 cho khoảng 95% [-0.042482, +0.009283]. Đây chưa phải bằng chứng chắc chắn về overfit; không nên train thêm epoch chỉ vì training loss giảm. Bootstrap theo câu có thể lạc quan nếu còn phụ thuộc giữa các câu gần nhau.

12 câu `generated_truncated` của epoch 1 có METEOR trung bình **0.351225**, so với **0.660581** ở 83 câu `generated`. Có 17/100 câu METEOR dưới 0.30. Các nhóm không được xem là đối chứng nhân quả: câu khó vốn có thể dễ bị cắt hơn.

**Stage 3:** ZIP đang `paused`, mới có **900/1.000** câu, không có submission cuối đầy đủ. Trong 900 câu: 122 chạm trần token (13.56%); route gồm 756 generated, 111 generated_truncated, 33 source_fallback; 32 câu raw bị cờ số hiệu không có trong evidence. Vì 11 câu chạm trần chuyển sang fallback nên không đồng nhất 122 với 111. Có 265/900 prompt từ 4.000 token trở lên; riêng con số này chưa chứng minh mất căn cứ.

Tốc độ ghi trong audit: epoch 1 dev100 trung bình 74.36 giây/câu; public900 trung bình 74.29 giây/câu. Tham chiếu chi phí khoảng 2,07 GPU-giờ cho 100 câu, 14,45 GPU-giờ cho 700 câu và 20,64 GPU-giờ cho 1.000 câu nếu chạy tuần tự với tốc độ cũ, chưa tính retrieval/load. Thời gian thực tế phụ thuộc GPU, độ dài và mức song song.

## 3. Các lỗi cụ thể quyết định thứ tự ưu tiên

Các nhận xét dưới đây so sánh prediction với reference/context của bộ dữ liệu; không phải xác nhận pháp luật hiện hành.

| ID dev100 | Bằng chứng | Hướng xử lý |
|---|---|---|
| 97507 | Hỏi phụ trách kế toán; context đầu chứa Điều 20 NĐ 174/2016, khớp nguồn reference. Model lại dùng context thứ tư về phụ trách kế toán của Quỹ. METEOR 0.1704. | Thử giảm nguồn gây nhiễu, kiểm tra phạm vi áp dụng trước khi sinh. Không mặc định thiếu recall. |
| 9023 | Context đầu có nội dung đánh nhau tại sân bay từ nguồn reference, nhưng raw answer trộn quy định của nguồn khác rồi sinh đến trần. METEOR 0.2419. | Giữ gắn kết nguồn–điều khoản–mức tiền; kiểm tra ngữ cảnh được pack thực tế. |
| 164143 | Hỏi có bị xử phạt nhiều lần không; top contexts tập trung hành vi không đội mũ bảo hiểm. Reference trả lời nguyên tắc xử phạt. METEOR 0.2408. | Retrieval phải bắt được ý định pháp lý, không chỉ từ khóa hành vi. |
| 160199 | Có đúng số hiệu văn bản reference ở context thứ hai nhưng là Điều 3; reference cần Điều 4. METEOR 0.0948. | Thử mở rộng điều liền kề trong cùng văn bản rồi rerank lại, có giới hạn. |
| 87427 | Prediction dùng nguồn 2015, reference dùng nguồn 2021. | Kiểm tra phiên bản và phạm vi câu hỏi; không luôn ưu tiên văn bản mới nhất cho mọi câu. |
| 8593, 11029, 156051 | Sinh các mục lặp hoặc chuỗi khoản tăng dần đến trần; reference ngắn hơn rõ rệt. | Phát hiện vòng lặp, regenerate có kiểm soát; tăng token đơn thuần không giải quyết nguyên nhân. |

Chưa có tokenizer runtime trong môi trường local để tái tạo chính xác prompt v8, và ba ZIP không chứa corpus index/weights để chạy lại mô hình. Vì vậy chưa kết luận định lượng mức mất evidence khi pack hoặc Recall@k. Overlap từ vựng chỉ được dùng làm tín hiệu chẩn đoán, không gọi là gold recall.

## 4. Kế hoạch thử nghiệm

### P0 — Khóa baseline trước khi tối ưu

1. Ghi hash file đã đạt 0.5599; xác nhận recipe/version stage 4, adapter, config và metric leaderboard. Lấy đủ artifact 1.000 câu đã hoàn tất; không tự điền 100 câu thiếu của snapshot này.
2. Chạy đúng recipe repair đó trên dev100, dùng cùng input stage 3. Lưu raw stage3 và sau repair, chấm bằng cùng scorer. Nếu recipe đã dùng reference để chọn candidate trên từng câu thì đó chỉ là oracle diagnostic, không được dùng làm baseline triển khai.
3. Báo cáo riêng các câu repair thay đổi: trước/sau, thắng/thua, nhóm lỗi và tổng delta trên toàn tập. Phân biệt lỗi đã được repair chữa với lỗi còn sót.
4. Khóa split hiện có. Dev100 dùng sàng lọc nhanh; 600 câu dev còn lại dùng xác nhận cấu hình. Holdout700 chỉ đánh giá ứng viên cuối sau khi khóa cấu hình; nếu đã dùng holdout để chỉnh thì phải ghi rõ nó không còn độc lập.

**Hoàn tất P0:** có baseline sau repair tái lập được trên dữ liệu có nhãn, lineage rõ và schema không đổi. Không thể suy ra baseline này từ riêng ba ZIP hiện tại.

### P1 — Thử inference trước, giữ checkpoint epoch 1

Mỗi thử nghiệm đổi một yếu tố; dùng cùng retrieval và cách chấm khi điều đó hợp lệ. Các mức dưới đây là điểm khởi đầu để thử, không phải cấu hình đã được chứng minh.

| Thử nghiệm | Thay đổi | Giả thuyết và rủi ro |
|---|---|---|
| G1 | Với các câu còn lỗi sau repair, regenerate khi có bằng chứng vòng lặp hoặc nhầm phạm vi nguồn. Thử `repetition_penalty=1.03`, rồi 1.05 nếu cần, so với 1.0. | Giảm lặp; penalty có thể làm mất thuật ngữ pháp lý. Không phạt mạnh hoặc xóa mọi đoạn lặp vì reference cũng có trích dẫn rồi kết luận. |
| G2 | So sánh 4 parents hiện tại với 2 parents trên cùng retrieval. Log số parent thực sự đưa vào prompt. | Giảm nguồn gây nhiễu như ID 97507; có thể mất ngoại lệ, văn bản sửa đổi hoặc câu hỏi nhiều vế. Chỉ thử routing động sau khi đo được trade-off. |
| G3 | Giữ 4 parents và ngân sách 4.096; đóng gói theo khoản/điểm hoàn chỉnh, bảo toàn phần mở đầu của mức phạt và điều kiện, giữ số hiệu/heading. | Giảm mất quan hệ điều khoản; nếu khoản vượt budget phải chọn cửa sổ và đánh dấu thiếu, không ngầm cắt rồi coi là đủ. |
| G4 | Prompt ràng buộc trả lời từng vế, dùng đúng thực thể/phạm vi, trích phần liên quan và kết luận; giữ nguyên grounding và chống bịa. | Giảm sao chép toàn điều và trộn nguồn. Có thể giảm METEOR nếu rút quá ngắn; không áp độ dài cố định cho mọi câu. |

Chỉ tăng `max_new_tokens` từ 1.536 lên 2.048 cho nhóm chạm trần nhưng không lặp và còn nội dung có căn cứ cần trình bày. So sánh riêng với baseline; token budget tối đa tăng 33%, chi phí thực tế cần đo. Không regenerate chỉ vì raw bị cắt nếu bản repair hiện tại đã tốt.

Candidate chọn bằng tín hiệu có ở inference: lặp, căn cứ, số hiệu/điều khoản, phạm vi chủ thể, các vế câu hỏi. Không dùng reference hoặc ngưỡng riêng theo ID public. Giữ đáp án cũ nếu candidate không qua kiểm tra căn cứ. Không nối hai đáp án để tăng độ dài.

### P2 — Cải thiện retrieval đúng nhóm lỗi

1. Phân biệt: nguồn cần thiết chưa vào pool / vào pool nhưng không được chọn / đã chọn nhưng pack mất / đã có trong prompt nhưng model dùng sai. Với các case ambiguity, ghi rõ reference chỉ là một cách diễn giải của câu hỏi.
2. Với nhóm thiếu nguồn, thử `pool_k: 32 → 64`, giữ nguyên các k khác; reranker candidates tăng tối đa gấp đôi, đo recall diagnostic, điểm trả lời và thời gian. Chưa tăng đồng loạt BM25/dense/parents.
3. Với câu hỏi kiểu ID 164143, thử tách truy vấn hành vi và truy vấn nguyên tắc pháp lý, hợp nhất và rerank. Câu hỏi tổng quát cần hạn chế nguồn đặc thù của một quỹ/cơ quan nếu người hỏi không yêu cầu phạm vi đó.
4. Với đúng văn bản nhưng sai điều như ID 160199, thử mở rộng một điều trước/sau rồi rerank trong cùng budget. Với văn bản sửa đổi, giữ quan hệ điều bị sửa và nội dung sửa.
5. Khi audit chứng minh sai thứ hạng, thử riêng giảm `lexical_score_weight` từ 2.0 xuống 1.0. Không khẳng định bonus đang gây lỗi trên toàn tập khi chưa có score của toàn bộ pool.

Giữ context IDs, score, nguồn, số candidates và truncation trong log. Mọi cache phải có identity đúng với code/config/index; không sửa metadata để ép dùng cache không tương thích.

### P3 — Train lại có mục tiêu nếu P1/P2 chưa đủ

- Cố định cách retrieve/pack tốt nhất rồi tạo lại context train bằng chính pipeline đó; câu hỏi là đầu vào retrieval, đáp án chỉ làm target, không nhét đáp án vào context.
- Ưu tiên một lượt fine-tune mới có kiểm soát từ cùng base model: giữ 5.600 mẫu gốc, đồng bộ full retrieval và prompt budget với inference. So sánh checkpoint 0.5/1.0 epoch; chưa mặc định cần 2 epoch.
- Giữ LR 5e-5 ở thử nghiệm đồng bộ đầu để tách nguyên nhân; chỉ thử LR 2e-5–3e-5 ở thí nghiệm tiếp theo nếu có bằng chứng. Không gộp đổi retrieval, độ dài, LR và target style rồi gán điểm tăng cho một yếu tố.
- Giữ LoRA rank 16 ban đầu. Với prompt dài hơn phải smoke-test bộ nhớ trên Kaggle T4, log peak VRAM và throughput trước full train.
- Nếu sửa target để loại đoạn lặp do lỗi dữ liệu, phải audit mẫu và giữ nguyên nội dung pháp lý; không rút toàn bộ answer theo độ dài cố định.

## 5. Cách chọn ứng viên và ngân sách chạy

Trình tự tiết kiệm lượt: P0 → tối đa 3–4 cấu hình inference riêng trên dev100 → xác nhận tối đa 2 ứng viên trên dev600 → khóa một ứng viên → holdout700 → public1000.

- Trên dev100: ưu tiên candidate cải thiện khoảng >=0.01 METEOR so với baseline sau repair, không làm ROUGE-L giảm quá 0.005, không tăng lỗi thiếu căn cứ. Đây là ngưỡng quản lý thử nghiệm đề xuất, không bảo đảm ý nghĩa thống kê.
- Trên dev600: paired comparison theo cùng ID; yêu cầu delta dương và khoảng bootstrap 95% có cận dưới >0 để coi bằng chứng cải thiện đáng tin hơn. Báo cáo nhóm lỗi, độ dài và runtime. Nếu chỉ vài câu kéo điểm, chưa kết luận tổng quát.
- Trên holdout700: kiểm tra một lần sau khi khóa, kỳ vọng cải thiện cùng chiều; không quay lại chỉnh ngưỡng theo holdout mà vẫn gọi đó là tập độc lập.
- Mục tiêu kỹ thuật mong muốn trước submission: tăng bền vững khoảng 0.03–0.04 local trên tập xác nhận lớn so với đúng baseline sau repair. Đây là mục tiêu, không phải ánh xạ chắc chắn sang +0.0401 public.
- Mốc runtime đề xuất: tăng không quá 25% cho inference so với baseline cùng phần cứng, trừ khi mức tăng chất lượng đủ rõ để chấp nhận chi phí. Đo riêng chi phí retrieval và repair.

Chỉ so sánh “epoch nào tốt hơn cho từng câu” để chẩn đoán: chọn oracle giữa epoch 1/2 trên dev100 được 0.644689, tăng 0.026406 so với epoch 1. Con số này sử dụng reference, không thể triển khai trực tiếp và không dự báo public.

## 6. Lệnh kiểm chứng khi có prediction ứng viên

Chạy trong repo trên Kaggle, sau khi kiểm tra các đường dẫn thực tế tồn tại. Các đường dẫn sau là bố cục output đề xuất, không phải artifact đã có sẵn:

```bash
python -m legalqa evaluate --predictions /kaggle/working/exp/baseline.dev600.json --references /kaggle/working/exp/dev600.references.json --output /kaggle/working/exp/baseline.dev600.metrics.json --label baseline_repaired
python -m legalqa evaluate --predictions /kaggle/working/exp/candidate.dev600.json --references /kaggle/working/exp/dev600.references.json --output /kaggle/working/exp/candidate.dev600.metrics.json --label candidate
python -m legalqa compare --baseline /kaggle/working/exp/baseline.dev600.metrics.json --candidate /kaggle/working/exp/candidate.dev600.metrics.json --output /kaggle/working/exp/paired.dev600.json
```

Giữ scorer BTC, không thay cách tokenization để làm đẹp điểm. Trước nộp: đủ 1.000 ID đúng tập questions, mỗi ID có `answer` là chuỗi không rỗng; schema/file name đúng metadata của cuộc thi; ghi hash file cuối và config. Chỉ public submission thực tế đạt >=0.60 mới xác nhận đạt đích điểm.

## 7. Kiểm chứng đã làm và giới hạn

- Đọc trực tiếp cả ba ZIP; config, models lock, split manifest, dev100 và test questions giống nhau giữa ba stage.
- Selection, metrics/audit epoch 1 và retrieval dev100/public giống nhau giữa stage 2 và stage 3.
- 900 checkpoint records là ID duy nhất và prediction khớp partial JSON tương ứng.
- Chấm lại METEOR epoch 1 trên toàn bộ 100 câu bằng `nltk.translate.meteor_score` và tokenization `.split()` như scorer BTC: mean 0.6182826973445282; sai lệch từng câu với metrics lưu là **0.0**. Hash `vendor/scoring.py` local khớp hash trong report.
- ROUGE-L lấy từ diagnostics; chưa chấm lại ROUGE-L local do thiếu dependency `absl`. Không cài dependency, sửa source hoặc chạy GPU cho tác vụ lập kế hoạch này.
- Chưa có bằng chứng xác định recipe stage 4 nào tạo ra 0.5599; số liệu public900 ở trên là trước repair. Không cộng dồn các mức tăng giả định giữa các thử nghiệm, không suy public từ local bằng một offset cố định.
