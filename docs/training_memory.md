# Sửa bộ nhớ Stage 1 theo log ngày 14/09/2026

## Chẩn đoán

Log `legalqa-main-01-qlora-train (5).log`, dòng 1911: rank 1 bị CUDA OOM
trong backward sau 139/1400 optimizer steps. Cần cấp thêm 2,74 GiB,
GPU còn 2,28 GiB; PyTorch đã cấp 10,31 GiB, giữ dự phòng 1,63 GiB.
Đây là bằng chứng thiếu **VRAM**; log không chứng minh RAM hệ thống đã hết.
Hai worker đã báo `cuda:0` và `cuda:1`, đều Tesla T4, effective batch 8.

Diagnostics `(3).zip` có 5.600 mẫu, tổng 13.736.538 token;
median 2.433, p99 3.344, dài nhất 5.067 token (ID 37801,
2.039 prompt + 3.028 answer). Traceback không chỉ rõ phép toán cấp phát
2,74 GiB. Logits/loss toàn chuỗi là nguồn đỉnh bộ nhớ phù hợp với kích thước
này; cần đo trên T4 để xác nhận mức giảm và khả năng chạy qua mẫu dài nhất.

## Thay đổi

- `training_memory.enable_fused_loss`: dùng forward Qwen2 của Liger 0.5.10
  cho riêng instance đang train, trước khi PEFT/DDP bọc model. Chỉ thay
  linear head + cross entropy; không sửa RoPE, attention, RMSNorm hay MLP.
  Fused loss chia nhỏ phần logits trung gian. Kiểm tra head đóng băng,
  không bias; thiếu dependency thì dừng rõ lỗi.
- Giữ causal shift, labels `-100`, `num_items_in_batch` và cơ chế
  `average_tokens_across_devices` của Trainer. Đây là cùng mục tiêu loss,
  không phải cam kết tensor/adapter giống từng bit khi dùng kernel khác.
- Giải phóng retrieval records và questions sau khi tạo manifest/samples;
  lưu input/labels bằng int32, attention mask uint8; chuyển sang int64 đúng
  lúc collate. Không đưa model hay activation ra RAM hệ thống.
- Giữ 5.600 mẫu, toàn bộ answer/context, thứ tự mẫu, batch, accumulation,
  hai GPU, checkpointing, optimizer, BM25 và cấu hình retrieval.

## Kiểm chứng local

`python -B -m unittest discover -s tests -v`: 97 tests, 96 pass, 1 CUDA skip.
Test CUDA tự chạy trên Linux có CUDA và requirements; so sánh loss/gradient
FP16 của Qwen2 có frozen head, vocabulary 151.936, mask prompt/padding và
denominator khác số token local. Tolerance loss rtol/atol 2e-4;
gradient rtol 0,03 / atol 2e-4. Đây là kiểm tra số học, chưa phải METEOR/ROUGE.

Fixture CPU theo đúng độ dài/mask của 5.600 mẫu, ID token mô phỏng:

| Phép đo | Cũ | Mới |
|---|---:|---:|
| Token storage mỗi worker, xấp xỉ | 715.262.952 byte | 123.628.842 byte |
| Chuyển dữ liệu batch sang NumPy, median 3 lượt | 0,799 s | 0,175 s |

Số bộ nhớ trên không phải RSS toàn tiến trình: phía cũ tính list và token
integers dùng chung một lần; phía mới tính array payload. Benchmark CPU
không đo tốc độ train GPU. Chưa có phép đo end-to-end trên hai T4 và chưa
có prediction mới để chấm METEOR/ROUGE.

## Đo trên Kaggle

Notebook Stage 1 hiện đã cài `requirements.txt` và chạy toàn bộ unittest.
Với bản code mới, test CUDA bên trên sẽ chạy ngay trước pipeline.
Benchmark thêm, từ thư mục repo:

```bash
python scripts/benchmark_training_loss.py --device cuda:0 --model-config /kaggle/working/stage1_runtime_models/generator/config.json
```

Kiểm tra đường dẫn config tồn tại trước khi chạy; có thể dùng config của
generator trong dataset model đã gắn. Benchmark ghi peak VRAM và median
forward/backward của **head + loss**, loại lượt warm-up khỏi thời gian.
Nó không thay thế phép đo full transformer/DDP. Sau đó đối chiếu thời gian
optimizer step và peak VRAM cả hai rank trong lượt train thực.

Code mới đổi fingerprint. Có thể dùng lại **retrieval** của commit cũ qua
`RETRIEVAL_INPUT`; xem [hướng dẫn nhập cache](retrieval_reuse.md). Đã kiểm tra
trên chính diagnostics `(3).zip`: 5.600 bản ghi giữ nguyên hoàn toàn.
Checkpoint optimizer cũ vẫn chưa resume trực tiếp qua implementation loss mới;
diagnostics ZIP không chứa adapter weights/optimizer. Không sửa tay hash.

## Nguồn đối chiếu

- [Transformers 4.51.3 causal loss](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/loss/loss_utils.py).
- [Liger 0.5.10 Qwen2 forward](https://github.com/linkedin/Liger-Kernel/blob/v0.5.10/src/liger_kernel/transformers/model/qwen2.py).
- [Liger loss normalization](https://github.com/linkedin/Liger-Kernel/blob/v0.5.10/src/liger_kernel/transformers/model/loss_utils.py).
- [Fused head/loss: chia chunk, không tạo gradient head khi frozen](https://github.com/linkedin/Liger-Kernel/blob/v0.5.10/src/liger_kernel/ops/fused_linear_cross_entropy.py).
