# Main4 adaptive: fallback và truncated

## Dữ liệu và baseline

Bản này dùng submission private **1.918 câu, điểm 0.5713** do người dùng xác nhận làm baseline.
Điểm main3 trước repair là **0.5704**. Đây là điểm được báo lại, không phải điểm dev hoặc điểm mới từ bản adaptive.

ZIP Stage 2 `(5)` khớp đủ 1.918 câu hỏi và retrieval. ZIP Stage 3 `(8)` là snapshot public
paused 900/1.000 câu, không trùng ID private: **không gắn ZIP này vào `PRIVATE_DIAGNOSTICS`**.
Dev100 hai ZIP có cùng predictions, gồm 12 câu hit-token-limit và 5 source-fallback.
Cả 5 fallback dev đều có số hiệu được sinh ra nhưng không có trong context.

Không thay model, adapter epoch-01, retrieval cache hoặc schema submission.
Luồng adaptive không tải lại corpus, train lại hoặc chạy reranker.

## Chạy trên Kaggle

1. Import `legalqa_main_04_repair_submit.ipynb` vào notebook Kaggle. Chọn GPU T4/P100,
   bật Internet để cài dependency. Adaptive dùng một GPU `cuda:0`, kể cả khi có T4 x2.
2. Add Input bốn nguồn:
   - `submission.zip` baseline 0.5713, hoặc `submission.json` được Kaggle giải nén.
   - `legalqa_main_stage2_v8_diagnostics (5).zip`, hoặc thư mục có `stage2_manifest.json`.
   - Model Version 3: thư mục `models` có `models.lock.json`, generator, embedding và reranker config.
   - Output Stage 2/3 có đầy đủ `selected_adapter/adapter_model.safetensors` và `adapter_config.json`.
     Diagnostics không chứa adapter weights.
3. Cell cấu hình mặc định:

   ```python
   MODE = 'adaptive_dev'
   BASELINE_SUBMISSION = None
   STAGE2_DIAGNOSTICS = None
   PRIVATE_DIAGNOSTICS = None
   PREVIOUS_OUTPUT = None
   GPU_MAX_ITEMS = 50
   WORK_HOURS = 9.0
   RUN_GPU = True
   ```

   `None` tự tìm duy nhất một baseline/Stage 2/adapter. Nếu có nhiều kết quả, điền đường dẫn
   được notebook liệt kê. Kiểm tra `MODEL_ROOT` tồn tại trong input đã gắn; đó là thư mục
   cha của `generator/`. Không cần `private-official.json` riêng hoặc index trong mode adaptive:
   câu hỏi được lấy từ Stage 2 đã kiểm hash và đối chiếu ID với baseline.
4. Save & Run All. Phiên đầu sinh smoke tối đa **5 ID mới**; khi paused, lưu toàn bộ output.
   Phiên sau Add Input output đó, đặt `PREVIOUS_OUTPUT` tới thư mục có `main04_state.json`,
   giữ nguyên input/code/mode. Các phiên tiếp theo dùng tối đa 50 ID mới.
5. Khi `adaptive_dev` complete, xem `adaptive/dev/{audit,heuristic}/decision.json` rồi đổi
   `MODE='adaptive_private'`, giữ nguyên output/resume. Code tự chọn nhóm đạt điều kiện dev;
   không cần điền winner hoặc chọn đáp án theo ID. Private cũng bắt đầu bằng smoke tối đa 5 ID.
6. Khi private complete, tải `adaptive/submission_adaptive.zip`. ZIP chỉ chứa `submission.json`.
   Nếu không nhóm nào qua dev, ZIP chứa nguyên baseline; xem báo cáo trước khi nộp.

`adaptive_audit` có thể chạy trước dev để chỉ lập hàng đợi và thống kê, đặt `RUN_GPU=False`.
Mode này dùng đúng tokenizer và kiểm identity của model/adapter nhưng không load model GPU.

### Nếu báo không tìm thấy Stage 2

Log `Chọn input cụ thể cho legalqa_main_stage2_v8_diagnostics*.zip: []` nghĩa là bộ tìm input cũ
không thấy ZIP đúng tên hoặc `stage2_manifest.json` trong `/kaggle/input`. Notebook mới nhận diện
ZIP bằng nội dung nên hỗ trợ ZIP đổi tên và file có hậu tố `(5)`, kể cả `.ZIP`.
Nếu vẫn thiếu, notebook dừng trước khi cài thư viện, in các ZIP/manifest đang thấy và hướng dẫn Add Input.

Upload `legalqa_main_stage2_v8_diagnostics (5).zip` thành Kaggle Dataset rồi **Add Input** dataset đó,
hoặc gắn output Stage 2 có file này. File ở Downloads máy local không tự xuất hiện trong Kaggle.
Nếu có nhiều bản, đặt `STAGE2_DIAGNOSTICS` bằng đường dẫn ZIP/thư mục thực tế hiện trong Input.
Không dùng Stage 3 public `(8)` thay cho Stage 2 private. Việc kiểm CRC/hash/ID vẫn giữ nguyên.

Nếu có diagnostics private Stage 3 hoàn chỉnh, có thể đặt `PRIVATE_DIAGNOSTICS` từ đầu.
Code bắt buộc ID/câu hỏi, config, retrieval records, adapter và model phù hợp. Không dùng
diagnostics paused hoặc tự suy ra ánh xạ ID. Thêm/đổi diagnostics sau khi đã chạy cần output mới
và chạy lại dev vì identity đã thay đổi.

## Chính sách repair

- Có audit: chọn lịch sử fallback, hit-token-limit/generated-truncated và đáp án baseline dang dở.
  Audit là lịch sử generation; hàng đợi ghi cả việc baseline khác prediction gốc.
- Thiếu audit: đánh dấu **nghi vấn** khi đáp án dang dở, độ dài tokenizer ≥ 90% trần gốc 1.536,
  hoặc khớp fallback tái dựng từ prompt packing (kể cả bản qua CPU V1).
  Không khẳng định một đáp án dài đã thực sự chạm trần. Không chọn câu chỉ vì lặp.
- Output bắt đầu 2.048, tăng 3.072 rồi 4.096 chỉ khi tiếp tục hit limit mà không lỗi căn cứ/lặp.
  Input tối đa 4.096; tổng không vượt 8.192 và cửa sổ thực của model/tokenizer.
  Nếu cửa sổ nhỏ hơn, ladder được giới hạn và ghi vào ngân sách thực của từng attempt;
  không tự giảm input. Model không đủ 4.096 + 2.048 thì dừng báo lỗi.
- Lượt đầu giữ bốn context theo thứ tự gốc. Khi lỗi căn cứ hoặc lặp, cho một lượt prompt tập trung,
  dùng parent có điểm hỗ trợ cao nhất và tối đa một parent cùng văn bản, giữ thứ tự gốc và biên đơn vị pháp lý.
  Tối đa bốn attempts mỗi ID; luôn sinh lại từ đầu, không nối phần trả lời cũ.
- Giữ greedy, repetition penalty 1.0, không cấm n-gram. Candidate được xóa lặp CPU V1 rồi kiểm tra lại.
- Không loại trước generation vì `strong=false`. Đầu ra phải dừng ở EOS, không hit limit,
  fallback/refusal, số hiệu ngoài evidence, artifact, xung đột thực thể, vòng lặp hoặc phần dẫn dang dở.
  Evidence trên **packed context thực tế** phải có coverage ≥ 0,65, cụm chung ≥ 3 từ, đáp ứng
  kiểm tra thời điểm/độ tuổi khi áp dụng; không bắt buộc anchor hai từ đầu.
- Dev chạy cả cách nhận diện bằng audit và che audit. Mỗi nhóm fallback/truncated được chấm trên
  toàn dev100 với các câu khác giữ baseline CPU V1. Chỉ bật nhóm có thay đổi, METEOR không giảm
  và không tăng số câu lặp nặng/dang dở; bản ghép cũng phải qua cùng điều kiện.
  Không dùng gold trong prompt, nhận diện lỗi hay chọn candidate từng ID.

Journal lưu ngay sau mỗi attempt: ngân sách thực, context IDs, prompt hash, generation,
kiểm tra căn cứ và lý do giữ/thay. Resume kiểm baseline/data/model/tokenizer/adapter/policy/code;
attempt đang chạy khi bị ngắt có thể cần chạy lại, các attempt đã commit được giữ.
`GPU_MAX_ITEMS` đếm ID mới trong phiên, dùng chung cho hai cách nhận diện dev; retry cùng ID
không tăng số ID nhưng vẫn tính vào deadline. Setup và chấm cũng nằm trong `WORK_HOURS`.
Lỗi GPU dừng và giữ journal để resume, không tự hạ cấu hình hay xuất submission chưa hoàn tất.

## Đầu ra

| File trong `adaptive/` | Nội dung |
|---|---|
| `identity.json` | Nguồn, baseline digest, policy, model/adapter/tokenizer hashes |
| `audit.json` | Hàng đợi và số lượng từng nhóm từ `adaptive_audit` |
| `dev/{audit,heuristic}/candidates.json` | ID, lý do, nhóm và cách phát hiện |
| `dev/{audit,heuristic}/decision.json` | Điểm, chênh lệch, gate từng nhóm và bản ghép |
| `{dev,private}/{audit,heuristic}/attempts.jsonl` | Checkpoint từng ID + attempt |
| `{dev,private}/{audit,heuristic}/outcomes.json` | Câu đã thay, số attempts, token, thời gian và lý do từ chối |
| `{dev,private}/{audit,heuristic}/unresolved.json` | Các câu đã thử nhưng chưa sửa đạt |
| `private/{audit,heuristic}/skipped.json` | Câu chưa thử vì nhóm không qua dev |
| `dev.status.json`, `private.status.json`, `status.json` | Trạng thái và thống kê |
| `submission_adaptive.zip` | Bản cuối đầy đủ ID, các đáp án không được nhận repair giữ nguyên baseline |

## Kiểm chứng local

Đã đọc và kiểm manifest/hash/journal trên ZIP thật: 1.918 private và 100 dev. Từ chối Stage 3
paused không phù hợp. Submission nguồn giữ nguyên SHA-256; có 26 đáp án kết thúc bằng phần dẫn dang dở.

Đã chấm lại toàn dev100 bằng scorer BTC, NLTK 3.9.1:

| Bản | METEOR | ROUGE-L |
|---|---:|---:|
| Epoch-01 gốc | 0.6182826973 | 0.5908622415 |
| CPU V1 làm baseline adaptive | 0.6202453105 | 0.5938193946 |

Kiểm thử CPU/mock GPU bao gồm loader, audit mismatch, nhận diện, EOS/override, ladder/context limit,
guard đầu ra, focus, resume giữa attempts, lỗi GPU, gate từng nhóm, smoke → dev → private và ZIP.
Các suite regression liên quan: `test_repair`, `test_repair_v2`, `test_experiments`, `test_core`,
`test_stages`, `test_generation_multigpu`. Notebook phải compile và payload khớp source.

```powershell
python -m unittest discover -s tests -p test_adaptive.py -v
python scripts/build_stage4_notebook.py
python -m unittest discover -s tests -p 'test_repair*.py' -v
python -m unittest discover -s tests -p test_experiments.py -v
```

Máy local không có PyTorch/Transformers hoặc GPU để sinh candidate thật. Chưa có điểm adaptive dev,
runtime GPU thực hoặc điểm private mới; các số phía trên chỉ xác nhận tái lập baseline.
Heuristic và guard không chứng minh đáp án đúng hoàn toàn. Dev100 đã dùng chọn adapter, do đó vẫn
phải xác nhận hiệu quả qua lần nộp private sau khi chạy notebook.

Các mode P1/P2/repair_v2 cũ giữ nguyên hành vi; xem [hướng dẫn cũ](main04_v2.md).
