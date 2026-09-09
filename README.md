# Pipeline LegalQA UIT DSC 2026 dưới 4B tham số

Bộ code độc lập này được dựng từ 5 tệp người dùng gửi trong lượt yêu cầu ngày 09/09/2026. Không dùng mã nguồn, checkpoint, lựa chọn mô hình hay tiêu chí release từ lịch sử chat. Mục tiêu là tạo một hệ thống Task 2 có thể huấn luyện, đánh giá đúng mã BTC và xuất submission để tái lập.

Cấu hình chính: BM25 và multilingual-e5-small → RRF → Vietnamese_Reranker → mở rộng ngữ cảnh theo điều luật → Vi-Qwen2-3B-RAG → kiểm tra đầu ra → JSON và ZIP. Fine-tune mô hình sinh bằng QLoRA trên câu hỏi và đáp án gốc của BTC, chọn checkpoint bằng METEOR của validation.

**Trạng thái bàn giao:** đã chạy kiểm định CPU cho dữ liệu, chia tập, chunking, BM25, ghép ngữ cảnh, mask nhãn SFT, cache và đóng gói. Chưa chạy trọng số thật, huấn luyện GPU hay đo điểm pipeline mới. Năm file đính kèm không chứa toàn bộ `train.json`, corpus và bộ câu hỏi test; môi trường tạo bộ code cũng không có PyTorch/GPU. Vì vậy đây là bộ triển khai để chạy và kiểm chứng trên Kaggle, chưa phải checkpoint đã được chứng minh tăng điểm. Xem `VALIDATION.md`.

## Mô hình và ngân sách

| Thành phần | Mô hình | Số tham số gốc theo kiến trúc |
| --- | --- | ---: |
| Dense encoder | intfloat/multilingual-e5-small | 117.653.760 |
| Cross-encoder | AITeamVN/Vietnamese_Reranker | 567.755.777 |
| Generator | AITeamVN/Vi-Qwen2-3B-RAG | 3.085.938.688 |
| Tổng chưa gắn LoRA | | 3.771.348.225 |
| LoRA rank 16 trên 7 projection của generator | | 29.933.568 |
| Tổng kể cả LoRA chưa merge | | **3.801.281.793** |

Ba mô hình nằm trong Excel được duyệt, theo các đường dẫn ở dòng 30, 5 và 3. Con số trên tính cả embedding, classification head và LM head dùng chung trọng số; không suy ra từ tên “3B” hay dung lượng file 4-bit. Lệnh `audit-models` khởi tạo kiến trúc đầy đủ trên thiết bị meta, đếm lại các tensor tham số và dừng nếu khác số dự kiến hoặc tổng đạt 4 tỷ. Quantization chỉ phục vụ VRAM. BM25, RRF và chỉ mục vector không có trọng số mô hình được học thêm.

## Chạy trên Kaggle bằng notebook

Import `legalqa_main_run.ipynb` vào Kaggle và bật GPU + Internet. Notebook clone nhánh `main` từ `https://github.com/lighth-gh/uit-dsc-2026-task2-legalqa.git` vào `/kaggle/working/uit-dsc-2026-task2-legalqa`, sau đó cài thư viện, tải trọng số và chạy toàn bộ pipeline trong Kaggle runtime; notebook chủ động dừng nếu chạy ngoài Kaggle.

Để kiểm tra tích hợp trước khi chạy pipeline đầy đủ, dùng `legalqa_smoke_pipeline.ipynb`. Notebook smoke mặc định index 250 file context và chạy 3 câu dev qua toàn bộ chuỗi retrieve → generate → evaluate; điểm smoke chỉ dùng để phát hiện lỗi chạy, không dùng để so sánh chất lượng.

Mặc định notebook dùng `train.json`, `public-official.json` và `selected-contexts.zip` trong repo vừa clone (`USE_REPO_DATA = True`). Với dữ liệu private hoặc Kaggle Dataset riêng, đặt `USE_REPO_DATA = False`, sửa `KAGGLE_DATASET_ROOT` tại cell cấu hình và chọn đúng tên file test.

| Cell có tiêu đề | Làm gì | Kết quả cần kiểm tra |
| --- | --- | --- |
| 1 Thiết lập Kaggle và đường dẫn | Chọn dữ liệu trong repo hoặc Kaggle Dataset | Runtime là Kaggle và các đường dẫn đúng |
| 2 Clone mã nguồn từ GitHub | Clone/cập nhật fast-forward nhánh `main` | In ra thư mục code và commit đang chạy |
| 3 Cài môi trường và kiểm định CPU | Cài dependencies, WordNet, chạy unittest | Mọi test đều qua |
| 4 Chia tập và khóa mô hình | Tạo train/dev/holdout, tải và cố định revision | `parameter_audit.json` dưới 4B |
| 5 Lập chỉ mục | Tạo SQLite BM25 và FAISS | Corpus có tài liệu/chunk không rỗng |
| 6 Smoke 30 câu | Truy xuất, sinh chưa SFT và chấm đúng BTC | Xem trực tiếp đáp án và audit |
| 7 Baseline và truy xuất tập phát triển | Lưu retrieval của dev và train | Báo cáo baseline và token coverage |
| 8 Fine tune | Huấn luyện một hoặc hai epoch | `training_data_report.json`, checkpoint mỗi epoch |
| 9 Chọn checkpoint trên dev | Sinh từ từng checkpoint và so với baseline | Chọn METEOR cao nhất cùng IDs |
| 10 Holdout | Đánh giá một lần trên tập đã giữ riêng | Kiểm tra khả năng tổng quát |
| 11 Public hoặc private submission | Dùng đúng cấu hình và checkpoint đã chọn | ZIP chứa đúng một JSON |

Cell 8, 10 và 11 dùng các cờ `RUN_SFT`, `RUN_HOLDOUT`, `RUN_SUBMISSION` để bạn chọn giai đoạn cần chạy. Mặc định các bước dài này tắt. Cờ này chỉ điều khiển thực nghiệm, không phải yêu cầu xin phép. Khi chạy lại, cache đúng fingerprint được tiếp tục; thay cấu hình làm fingerprint khác thì dùng tên output mới. Không bỏ kiểm tra fingerprint để dùng lại kết quả khác mô hình.

Notebook dùng một GPU cho mỗi subprocess; đặc biệt QLoRA chỉ thấy GPU 0. Nếu có hai T4, không mặc định coi chúng là một GPU có VRAM cộng gộp. Retrieval hoàn tất và nhả model trước khi generation bắt đầu. Chưa có benchmark tốc độ/VRAM thực tế của cấu hình này.

## Chạy bằng dòng lệnh

Giải nén ZIP, mở terminal tại thư mục có `config.json`. Các dòng dưới đây chạy riêng biệt. Thay đường dẫn dữ liệu bằng đường dẫn thực tế.

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
python -m unittest discover -s tests -v
python -m legalqa fetch-models
python -m legalqa audit-models
python -m legalqa prepare --train /path/train.json --test /path/public-official.json --output runs/data
python -m legalqa build-index --corpus /path/selected-contexts.zip --output runs/index
python -m legalqa retrieve --questions runs/data/dev30.questions.json --index runs/index --output runs/dev30.retrieval.json
python -m legalqa generate --questions runs/data/dev30.questions.json --retrieval runs/dev30.retrieval.json --output runs/dev30.base.json
python -m legalqa evaluate --predictions runs/dev30.base.json --references runs/data/dev30.references.json --output runs/dev30.base.metrics.json --label base
```

Sau smoke, tạo retrieval và baseline cho toàn bộ dev; tạo retrieval cho train mà chỉ đưa **question** vào retriever:

```bash
python -m legalqa retrieve --questions runs/data/dev.questions.json --index runs/index --output runs/dev.retrieval.json
python -m legalqa generate --questions runs/data/dev.questions.json --retrieval runs/dev.retrieval.json --output runs/dev.base.json
python -m legalqa evaluate --predictions runs/dev.base.json --references runs/data/dev.references.json --output runs/dev.base.metrics.json --label base
python -m legalqa diagnose-retrieval --qa runs/data/dev.json --retrieval runs/dev.retrieval.json --index runs/index --output runs/dev.coverage.json
python -m legalqa retrieve --questions runs/data/train.questions.json --index runs/index --output runs/train.retrieval.json
python -m legalqa fit --train runs/data/train.json --retrieval runs/train.retrieval.json --output runs/sft --gpu 0
```

`fit` đặt `CUDA_VISIBLE_DEVICES` trước khi import torch. Khi tiếp tục lần train bị ngắt, chỉ định `--resume runs/sft/checkpoint-N` cùng input và cấu hình. Kiểm tra `training_data_report.json`: đáp án quá dài không bị cắt lén; mẫu không vừa được ghi rõ ID và bỏ khỏi lượt train. Nếu số bỏ đáng kể, cần điều chỉnh ngân sách trước khi train dài.

Ví dụ đánh giá một checkpoint. Thay `checkpoint-N` bằng thư mục thực tế do Trainer tạo, không đoán N:

```bash
python -m legalqa generate --questions runs/data/dev.questions.json --retrieval runs/dev.retrieval.json --adapter runs/sft/checkpoint-N --output runs/dev.sft.json
python -m legalqa evaluate --predictions runs/dev.sft.json --references runs/data/dev.references.json --output runs/dev.sft.metrics.json --label sft_epoch
python -m legalqa compare --baseline runs/dev.base.metrics.json --candidate runs/dev.sft.metrics.json --output runs/dev.comparison.json
python -m legalqa select --reports runs/dev.base.metrics.json runs/dev.sft.metrics.json --output runs/selection.json
```

Lặp hai lệnh generate/evaluate cho từng checkpoint, rồi đưa tất cả báo cáo vào `select`. Nếu baseline tốt nhất thì giữ baseline. Không coi `adapter_last` mặc nhiên là tốt nhất. ROUGE-L dùng để theo dõi chất lượng bổ sung; không yêu cầu cả hai metric đều tăng mới được chọn mô hình, vì BTC xếp hạng chính theo METEOR.

Với checkpoint được chọn, chạy holdout bằng cùng các lệnh trên, đổi `dev` thành `holdout`. Khi sang test, retriever và generator chỉ nhận câu hỏi:

```bash
python -m legalqa retrieve --questions runs/data/test.questions.json --index runs/index --output runs/test.retrieval.json
python -m legalqa generate --questions runs/data/test.questions.json --retrieval runs/test.retrieval.json --adapter runs/sft/checkpoint-N --output runs/submission.json
python -m legalqa package --predictions runs/submission.json --questions runs/data/test.questions.json --output runs/submission.zip
```

Bỏ `--adapter` nếu chọn baseline. ZIP mặc định chứa `submission.json` ở gốc. **ZIP scorer được gửi không kèm `metadata.json` của bộ reference để xác nhận tên `metadata.files.input` trên máy chấm.** `submission.json` là tên mặc định theo log cung cấp; kiểm tra tên yêu cầu ở vòng thi, nếu khác thì dùng `package --filename ten_btc_yeu_cau.json`. Code không cần biết tên file reference trên máy BTC để chấm validation tại máy bạn.

## Thử nghiệm có kiểm soát

Ưu tiên giữ cố định embedding và reranker, tối ưu generation trước. Dùng `dev30` để tìm lỗi chạy, `dev100` để thử nhanh, toàn bộ `dev` để chọn cấu hình. Hai tập nhỏ là tập con của dev, không phải phép kiểm định độc lập. Holdout được giữ riêng ngay từ đầu.

| Thử nghiệm | Chỉ thay đổi | Mục đích |
| --- | --- | --- |
| RAG chưa SFT | Không có adapter | Mốc so sánh cho chính pipeline mới |
| SFT epoch 1 và epoch 2 | Một adapter tại một thời điểm | Học độ đầy đủ, thuật ngữ và cách diễn đạt của BTC |
| Ngữ cảnh 3, 4, 5 parent | `retrieval.parents_k` | Kiểm tra thêm bằng chứng có lợi hay làm nhiễu |
| Pool 24 và 40 | `retrieval.pool_k` | Xem reranker có bỏ lỡ đoạn liên quan |
| Output 1536, 2048, 3072 token | `generation.max_new_tokens` | Đo thiếu ý và chi phí; giữ tổng input/output ≤8192 |

Các con số là điểm xuất phát để đo, không phải cấu hình đã được tối ưu trên dữ liệu này. Nếu đổi tham số retrieval, tạo cache retrieval mới. Giữ nguyên IDs, reference và metric để so sánh. Không sử dụng Public Test làm validation có đáp án.

Có thể refit trên `train_all.json` sau khi đã chốt toàn bộ cấu hình và số epoch. Khi đó chạy retrieval `train_all.questions.json`, train vào thư mục mới và không tiếp tục báo điểm dev/holdout như số đo độc lập của model refit, vì những QA đó đã vào train. Giữ nguyên model đã được kiểm định là đường triển khai ít phát sinh khác biệt hơn.

## Tái lập và lưu thực nghiệm

Giữ cùng run: `config.json`, `models/models.lock.json`, `models/parameter_audit.json`, `split_manifest.json`, chỉ mục và manifest, adapter đã chọn, báo cáo METEOR/ROUGE-L, prediction manifest, audit, checkpoint JSONL và kết quả `python -m pip freeze`. Có thể tải trước trọng số rồi chạy offline; `from_pretrained` trong các stage đều dùng file cục bộ. Hub chỉ được dùng ở `fetch-models` để tải trọng số và ghi revision.

Sau khi hết phiên Kaggle, chỉ dữ liệu trong output đã Save Version hoặc đã tải xuống mới tiếp tục dùng được ở phiên sau. Thêm output đó làm input rồi chép các thư mục muốn tiếp tục sang `/kaggle/working`; cập nhật đường dẫn tại cell cấu hình. Để đóng gói cho BTC, thêm các trọng số/adapter vào gói code hoặc trình bày bước tải đúng revision theo lock, như quy định BTC cho phép.

Xem `PIPELINE.md` để đọc lập luận thiết kế và đối chiếu từng yêu cầu. Xem `VALIDATION.md` để phân biệt phần đã kiểm định tại đây với phần cần chạy trên môi trường GPU.
