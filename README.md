# Pipeline LegalQA UIT DSC 2026 dưới 4B tham số

Bộ code độc lập này được dựng từ 5 tệp người dùng gửi trong lượt yêu cầu ngày 09/09/2026. Không dùng mã nguồn, checkpoint, lựa chọn mô hình hay tiêu chí release từ lịch sử chat. Mục tiêu là tạo một hệ thống Task 2 có thể huấn luyện, đánh giá đúng mã BTC và xuất submission để tái lập.

Cấu hình quality V8: BM25 thường + BM25 cụm từ/chính xác và multilingual-e5-small → RRF → Vietnamese_Reranker kèm lexical/legal/recency boosts → mở rộng ngữ cảnh theo điều luật → Vi-Qwen2-3B-RAG → fallback cục bộ có kiểm tra thực thể → JSON và ZIP. Main run bắt buộc fine-tune mô hình sinh bằng QLoRA trên câu hỏi và đáp án gốc của BTC, rồi chọn checkpoint bằng METEOR trên dev100.

**Trạng thái:** V7 đạt METEOR 0,45299 trên dev100, tốt hơn V5 0,44316 và V6 0,42585 nhưng vẫn xa mục tiêu 0,65. Quality V8 giới hạn fallback vào cửa sổ bằng chứng mạnh, chặn trộn citation/thực thể và bắt buộc QLoRA; cần chạy lại smoke30/dev100 rồi mới chạy main. Xem `VALIDATION.md`.

**Mục tiêu tối ưu:** ưu tiên METEOR tuyệt đối khi chọn cấu hình/checkpoint; ROUGE-L chỉ dùng để phá hòa khi METEOR bằng nhau. Mốc cần đạt trên validation là **METEOR ≥ 0,65**.

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

Import một trong ba notebook vào Kaggle và bật GPU + Internet. Cả ba clone nhánh `main` từ `https://github.com/lighth-gh/uit-dsc-2026-task2-legalqa.git`, đọc cùng `config.json`, dùng dữ liệu từ `lighth/uit-dsc-2026-task2-legalqa-train` và tái sử dụng index/model từ `lighth/ver3-smoke-output`; notebook chủ động dừng nếu chạy ngoài Kaggle.

Chạy theo thứ tự: `legalqa_smoke_pipeline.ipynb` (30 câu), `legalqa_dev100_pipeline.ipynb` (100 câu), rồi `legalqa_main_run.ipynb` (full dev/SFT/submission). Trong **Add Input → Datasets**, gắn `lighth/ver3-smoke-output` và `lighth/uit-dsc-2026-task2-legalqa-train`. Full index 407.107 chunks và model weights được đọc từ Dataset, không build hoặc tải lại.

Ba notebook mặc định dùng `USE_REPO_DATA = False` và đường dẫn `/kaggle/input/datasets/lighth/...`. Với vòng private, sửa `KAGGLE_DATASET_ROOT` sang Dataset chứa đúng test private và đổi `PHASE` trong main-run; không đổi nguồn index/model nếu corpus không thay đổi.

| Cell có tiêu đề | Làm gì | Kết quả cần kiểm tra |
| --- | --- | --- |
| 1 Thiết lập Kaggle và đường dẫn | Chọn dữ liệu trong repo hoặc Kaggle Dataset | Runtime là Kaggle và các đường dẫn đúng |
| 2 Clone mã nguồn từ GitHub | Clone/cập nhật fast-forward nhánh `main` | In ra thư mục code và commit đang chạy |
| 3 Cài môi trường và kiểm định CPU | Cài dependencies, WordNet, chạy unittest | Mọi test đều qua |
| 4 Chia tập và khóa mô hình | Tạo train/dev/holdout, dùng model lock Version 3 | `parameter_audit.json` dưới 4B |
| 5 Xác nhận chỉ mục | Kiểm tra SQLite/FAISS Version 3 | Đúng 8.507 tài liệu và 407.107 chunks |
| 6 Smoke 30 câu | Truy xuất, sinh chưa SFT và chấm đúng BTC | Xem trực tiếp đáp án và audit |
| 7 Baseline và truy xuất tập phát triển | Baseline dev100; chọn 768 QA và tạo lexical retrieval cho train | Báo cáo baseline và cache train không dùng answer để truy xuất |
| 8 Fine tune | Bắt buộc QLoRA hai epoch | `training_result.json`, `training_data_report.json`, checkpoint mỗi epoch |
| 9 Chọn checkpoint trên dev100 | Sinh từ từng checkpoint và so với baseline | Chọn METEOR cao nhất; ROUGE-L chỉ phá hòa |
| 10 Holdout | Đánh giá một lần trên tập đã giữ riêng | Kiểm tra khả năng tổng quát |
| 11 Public hoặc private submission | Dùng đúng cấu hình và checkpoint đã chọn | `submission.zip` chứa đúng một JSON |
| 12 Đóng gói diagnostics | Gom config, dev100, QLoRA/checkpoint state và submission audit; không kèm weights | `legalqa_main_quality_v8_public_diagnostics.zip` |

Main run đặt `RUN_SFT = True`, `RUN_SUBMISSION = True` và sẽ dừng nếu QLoRA bị tắt, không tạo được `training_result.json`, không có checkpoint để đánh giá, hoặc checkpoint được chọn không có adapter. Baseline chỉ là mốc so sánh; submission bắt buộc dùng checkpoint QLoRA có METEOR cao nhất. Chỉ `RUN_HOLDOUT` mặc định tắt. Khi chạy lại, cache đúng fingerprint được tiếp tục; thay cấu hình làm fingerprint khác thì dùng tên output mới. Không bỏ kiểm tra fingerprint để dùng lại kết quả khác mô hình.

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

Sau smoke, tạo baseline trên dev100, chọn tập QLoRA xác định bằng seed và tạo lexical retrieval mà chỉ đưa **question** vào retriever:

```bash
python -m legalqa retrieve --questions runs/data/dev100.questions.json --index runs/index --output runs/dev100.retrieval.json
python -m legalqa generate --questions runs/data/dev100.questions.json --retrieval runs/dev100.retrieval.json --output runs/dev100.base.json
python -m legalqa evaluate --predictions runs/dev100.base.json --references runs/data/dev100.references.json --output runs/dev100.base.metrics.json --label base_v8
python -m legalqa prepare-sft --train runs/data/train.json --output runs/data/train.sft.json
python -m legalqa retrieve --questions runs/data/train.sft.questions.json --index runs/index --output runs/train.sft.lexical.retrieval.json --mode lexical
python -m legalqa fit --train runs/data/train.sft.json --retrieval runs/train.sft.lexical.retrieval.json --output runs/sft --gpu 0
```

`fit` đặt `CUDA_VISIBLE_DEVICES` trước khi import torch. Khi tiếp tục lần train bị ngắt, chỉ định `--resume runs/sft/checkpoint-N` cùng input và cấu hình. Kiểm tra `training_data_report.json`: đáp án quá dài không bị cắt lén; mẫu không vừa được ghi rõ ID và bỏ khỏi lượt train. Nếu số bỏ đáng kể, cần điều chỉnh ngân sách trước khi train dài.

Ví dụ đánh giá một checkpoint. Thay `checkpoint-N` bằng thư mục thực tế do Trainer tạo, không đoán N:

```bash
python -m legalqa generate --questions runs/data/dev100.questions.json --retrieval runs/dev100.retrieval.json --adapter runs/sft/checkpoint-N --output runs/dev100.sft.json
python -m legalqa evaluate --predictions runs/dev100.sft.json --references runs/data/dev100.references.json --output runs/dev100.sft.metrics.json --label sft_epoch
python -m legalqa compare --baseline runs/dev100.base.metrics.json --candidate runs/dev100.sft.metrics.json --output runs/dev100.comparison.json
python -m legalqa select --reports runs/dev100.sft.metrics.json --output runs/selection.json
```

Lặp hai lệnh generate/evaluate cho từng checkpoint, rồi đưa các báo cáo checkpoint vào `select`; baseline chỉ đi vào lệnh `compare`. Main run bắt buộc chọn một adapter QLoRA, không coi `adapter_last` mặc nhiên là tốt nhất. Nếu adapter tốt nhất vẫn kém baseline, notebook cảnh báo rõ nhưng không âm thầm bỏ QLoRA. ROUGE-L dùng để theo dõi chất lượng bổ sung; không yêu cầu cả hai metric đều tăng mới được chọn mô hình, vì BTC xếp hạng chính theo METEOR.

Với checkpoint được chọn, chạy holdout bằng cùng các lệnh trên, đổi `dev` thành `holdout`. Khi sang test, retriever và generator chỉ nhận câu hỏi:

```bash
python -m legalqa retrieve --questions runs/data/test.questions.json --index runs/index --output runs/test.retrieval.json
python -m legalqa generate --questions runs/data/test.questions.json --retrieval runs/test.retrieval.json --adapter runs/sft/checkpoint-N --output runs/submission.json
python -m legalqa package --predictions runs/submission.json --questions runs/data/test.questions.json --output runs/submission.zip
```

Trong quality V8 main run không bỏ `--adapter`: submission bắt buộc dùng checkpoint QLoRA đã chọn. ZIP mặc định chứa `submission.json` ở gốc. **ZIP scorer được gửi không kèm `metadata.json` của bộ reference để xác nhận tên `metadata.files.input` trên máy chấm.** `submission.json` là tên mặc định theo log cung cấp; kiểm tra tên yêu cầu ở vòng thi, nếu khác thì dùng `package --filename ten_btc_yeu_cau.json`. Code không cần biết tên file reference trên máy BTC để chấm validation tại máy bạn.

## Thử nghiệm có kiểm soát

Ưu tiên giữ cố định embedding và reranker, tối ưu generation trước. Dùng `dev30` để tìm lỗi chạy, `dev100` để thử nhanh, toàn bộ `dev` để chọn cấu hình. Hai tập nhỏ là tập con của dev, không phải phép kiểm định độc lập. Holdout được giữ riêng ngay từ đầu.

| Thử nghiệm | Chỉ thay đổi | Mục đích |
| --- | --- | --- |
| RAG chưa SFT | Không có adapter | Mốc so sánh cho chính pipeline mới |
| SFT epoch 1 và epoch 2 | Một adapter tại một thời điểm | Học độ đầy đủ, thuật ngữ và cách diễn đạt của BTC |
| Ngữ cảnh 3, 4, 5 parent | `retrieval.parents_k` | Kiểm tra thêm bằng chứng có lợi hay làm nhiễu |
| Pool 24 và 32 | `retrieval.pool_k` | Đo trade-off recall và thời gian reranker |
| Output 1536, 2048, 3072 token | `generation.max_new_tokens` | Đo thiếu ý và chi phí; giữ tổng input/output ≤8192 |

Các con số là điểm xuất phát để đo, không phải cấu hình đã được tối ưu trên dữ liệu này. Nếu đổi tham số retrieval, tạo cache retrieval mới. Giữ nguyên IDs, reference và metric để so sánh. Không sử dụng Public Test làm validation có đáp án.

Có thể refit trên `train_all.json` sau khi đã chốt toàn bộ cấu hình và số epoch. Khi đó chạy retrieval `train_all.questions.json`, train vào thư mục mới và không tiếp tục báo điểm dev/holdout như số đo độc lập của model refit, vì những QA đó đã vào train. Giữ nguyên model đã được kiểm định là đường triển khai ít phát sinh khác biệt hơn.

## Tái lập và lưu thực nghiệm

Giữ cùng run: `config.json`, `models/models.lock.json`, `models/parameter_audit.json`, `split_manifest.json`, tham chiếu Dataset/index Version 3, adapter đã chọn, báo cáo METEOR/ROUGE-L, prediction manifest, audit, checkpoint JSONL và kết quả `python -m pip freeze`. `from_pretrained` trong các stage đều dùng file cục bộ từ Dataset; Hub chỉ được dùng khi chủ động chạy `fetch-models` ngoài ba notebook đồng bộ.

Sau khi hết phiên Kaggle, chỉ dữ liệu trong output đã Save Version hoặc đã tải xuống mới tiếp tục dùng được ở phiên sau. Thêm output đó làm input rồi chép các thư mục muốn tiếp tục sang `/kaggle/working`; cập nhật đường dẫn tại cell cấu hình. Để đóng gói cho BTC, thêm các trọng số/adapter vào gói code hoặc trình bày bước tải đúng revision theo lock, như quy định BTC cho phép.

Xem `PIPELINE.md` để đọc lập luận thiết kế và đối chiếu từng yêu cầu. Xem `VALIDATION.md` để phân biệt phần đã kiểm định tại đây với phần cần chạy trên môi trường GPU.
