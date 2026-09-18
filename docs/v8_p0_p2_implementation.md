# Triển khai P0–P2 cho LegalQA v8

Ngày triển khai: 18/09/2026. Các thay đổi mới đều **opt-in**; `config.json` và recipe tạo baseline cũ không đổi.

## P0 — baseline đã khóa

Record máy đọc được: `docs/v8_p0_baseline_lock.json`.

- Nguồn Stage 3 SHA-256: `a19932405fe8ae65713d290c362a1ba2b968ee71d090ea4479dcdeeda018afe7`.
- Adapter: `epoch-01`.
- Public JSON SHA-256: `a3cdecf5d26b9c4ba0f2825b6f359a902462e204b1c9a0e3fbb615e102f9155c`.
- Dev100 sau repair: METEOR `0.6202453105`, ROUGE-L `0.5938193946`.
- Public `0.5599` là điểm leaderboard do người dùng báo, không thể chấm lại local.
- CRC, hash, schema, ID, journal và identity của artifact 1.000 câu đều đã qua kiểm tra.

Tái kiểm tra P0:

```powershell
python -m legalqa.experiments lock-baseline `
  --root stage4_result_20260918 `
  --output docs/v8_p0_baseline_lock.check.json `
  --public-score 0.5599
```

## P1 — inference ablation

Bộ runner chỉ chọn câu bằng tín hiệu có tại inference: lặp, dang dở, chạm token, fallback và evidence đã cache. Gold/reference không đi vào prompt hoặc quy tắc giữ từng câu. Candidate mới bị từ chối nếu fallback, tiếp tục chạm token, thiếu evidence mạnh, xung đột thực thể/số hiệu, từ chối trả lời hoặc còn lặp/dang dở.

Các biến được thử độc lập:

| Tên | Một thay đổi duy nhất |
|---|---|
| `g1_penalty_103` | `repetition_penalty=1.03` |
| `g1_penalty_105` | `repetition_penalty=1.05` |
| `g2_contexts_2` | prompt dùng 2 thay vì 4 parent, retrieval giữ nguyên |
| `g3_complete_units` | cửa sổ context co về biên khoản/điểm/đoạn hoàn chỉnh |
| `g4_grounded_prompt` | ràng buộc từng vế, đúng phạm vi và không trộn nguồn |

Ví dụ chạy một biến trên dev100 ở môi trường có model/adapter GPU:

```powershell
python -m legalqa.experiments inference `
  --diagnostics "legalqa_main_stage3_v8_diagnostics (5).zip" `
  --baseline stage4_v2_verified_20260918 `
  --models MODEL_ROOT `
  --adapter ADAPTER_ROOT `
  --variant g1_penalty_103 `
  --output runs/v8_060/g1_penalty_103_dev `
  --split dev `
  --max-items 50
```

Runner có journal và identity; nếu paused, chạy lại đúng lệnh để tiếp tục. Không dùng lại output khi đổi variant/code/model. `decision.json` chỉ đánh dấu qua vòng sàng lọc khi METEOR tăng ít nhất `0.01`, ROUGE-L không giảm quá `0.005`, và số câu lặp nặng không tăng.

Chỉ sau khi cùng variant qua dev mới được chạy public; runner bắt buộc truyền kết quả dev tương ứng:

```powershell
python -m legalqa.experiments inference `
  --diagnostics "legalqa_main_stage3_v8_diagnostics (5).zip" `
  --baseline stage4_v2_verified_20260918 `
  --models MODEL_ROOT `
  --adapter ADAPTER_ROOT `
  --variant g1_penalty_103 `
  --dev-result runs/v8_060/g1_penalty_103_dev `
  --output runs/v8_060/g1_penalty_103_public `
  --split public `
  --max-items 50
```

Audit generation giờ ghi số context thật sự, token từng context, penalty, n-gram, chiến lược đóng gói và danh sách parent ID.

## P2 — retrieval ablation

Config đã sinh dưới `experiments/v8_060/retrieval/`; manifest khóa patch và hash tại `experiments/v8_060/manifest.json`.

| Tên | Một thay đổi duy nhất |
|---|---|
| `r1_pool_64` | rerank pool 32 → 64 |
| `r2_intent_query` | thêm query nguyên tắc xử phạt khi câu hỏi hỏi một/nhiều hành vi hoặc nhiều lần |
| `r3_adjacent_articles` | dành 8/32 slot cho Điều liền trước/sau của top seed, vẫn rerank đúng 32 candidate |
| `r4_lexical_weight_1` | lexical adjustment 2.0 → 1.0 |
| `r5_scope_penalty` | penalty nhẹ nguồn Quỹ khi câu hỏi không yêu cầu phạm vi Quỹ |

Retrieval log thêm `query_variants`, `pool`, `adjacent`, `candidate_count`, score và thời gian rerank. Mở rộng Điều chỉ lấy parent cùng `doc_id`; không ghép văn bản khác.

Quy trình cho từng config, ví dụ `r1_pool_64`:

```powershell
$CFG = "experiments/v8_060/retrieval/r1_pool_64.json"
$OUT = "runs/v8_060/r1_pool_64"

python -m legalqa --config $CFG --models MODEL_ROOT retrieve `
  --questions DEV100_QUESTIONS --index INDEX_ROOT --output "$OUT.retrieval.json"

python -m legalqa --config $CFG --models MODEL_ROOT generate `
  --questions DEV100_QUESTIONS --retrieval "$OUT.retrieval.json" `
  --adapter ADAPTER_ROOT --output "$OUT.raw.json"

python -m legalqa.experiments postprocess `
  --predictions "$OUT.raw.json" --audit "$OUT.raw.audit.json" `
  --output "$OUT.repaired.json"

python -m legalqa --config $CFG evaluate `
  --predictions "$OUT.repaired.json" --references DEV100_REFERENCES `
  --output "$OUT.metrics.json" --label r1_pool_64

python -m legalqa --config $CFG evaluate `
  --predictions stage4_v2_verified_20260918/dev.selected.json `
  --references DEV100_REFERENCES `
  --output "$OUT.baseline.metrics.json" --label baseline_repaired

python -m legalqa --config $CFG compare `
  --baseline "$OUT.baseline.metrics.json" `
  --candidate "$OUT.metrics.json" --output "$OUT.paired.json"
```

Chạy `diagnose-retrieval` trước generation để loại variant không cải thiện answer-token coverage diagnostic. Chỉ số này không phải gold Recall@k. Với variant còn triển vọng, chạy đủ dev100; xác nhận tối đa hai ứng viên trên dev600 rồi mới khóa một cấu hình cho holdout700/public1000.

## Giới hạn hiện tại

Máy local không có index 407.107 chunks, model gốc và adapter weights nên mới kiểm được code, artifact P0 và luồng CPU; chưa có kết quả GPU cho P1/P2. Không được coi các config mới là tốt hơn cho đến khi có `decision.json`, metric dev100/dev600 và runtime thực.
