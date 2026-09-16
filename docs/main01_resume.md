# Main 01: output lần 1 → Input lần 2

Phạm vi bản sửa ngày 16/09/2026: notebook 01 và phần chuẩn bị QLoRA/import
retrieval mà notebook gọi. Không đổi config, top-k BM25, 5.600 QA, số epoch,
prompt, target, LoRA, batch hiệu dụng hoặc schema submission.

## Cách chạy

1. Đưa bản sửa lên repo `lighth-gh/uit-dsc-2026-task2-legalqa` mà notebook clone,
   rồi dùng notebook 01 mới. Chọn **GPU T4 x2**, Internet bật để clone/cài gói.
2. Add Input Version 3 chứa `index/corpus.sqlite`, `index/index_manifest.json`,
   `models/models.lock.json` và ba thư mục model có config/trọng số.
3. Lần đầu Add Input dataset chứa `train.json` + `public-official.json`.
   Để `VERSION3_ROOT=None`, `DATASET_ROOT=None`; notebook tự tìm theo file.
4. Lần tiếp theo Add Input **toàn bộ output notebook lần trước**, giữ Version 3.
   Để `INPUT_MODE='auto'`, `PREVIOUS_OUTPUT=None`, `RETRIEVAL_INPUT=None`.
   Nếu snapshot đã có split, không cần gắn lại dataset gốc. Input chỉ đọc;
   artifact được kiểm tra và copy sang `/kaggle/working` để tiếp tục ghi.
5. Bộ dò resume bỏ qua manifest thiếu artifact trong inventory (ví dụ diagnostics
   đã giải nén thiếu trọng số/optimizer). Nhiều output đầy đủ: ưu tiên nguồn trong
   `PREFERRED_PREVIOUS_NOTEBOOK`, mặc định `lighth/legalqa-main-01-qlora-train`
   theo notebook đã chọn. Đặt `None` để tắt ưu tiên. `PREVIOUS_OUTPUT` cụ thể luôn
   thắng ưu tiên này và không quét nhầm diagnostics bên dưới ROOT đó.
   Nếu vẫn có nhiều output, hoặc nhiều index/dataset, notebook dừng để chọn ROOT.
   Không tự chọn bản mới nhất bằng tên file hoặc mtime. SHA256 vẫn được kiểm tra
   đầy đủ khi restore; bước dò chỉ kiểm tra cấu trúc, sự tồn tại và kích thước.

| INPUT_MODE | Hành vi |
|---|---|
| `auto` | Ưu tiên output Stage 1 để resume; chỉ có diagnostics ZIP thì import retrieval, train mới |
| `resume` | Bắt buộc tìm thấy output Stage 1; giữ checkpoint và optimizer |
| `retrieval` | Tự tìm cache hoàn tất trong folder/ZIP, kiểm tra rồi dùng code mới để train lại |
| `fresh` | Bỏ qua các output cũ đang gắn Input; dùng phiên Kaggle mới |

`PREVIOUS_OUTPUT` nhận thư mục snapshot hoặc mount chứa một snapshot.
`RETRIEVAL_INPUT` nhận folder, file cache, hoặc diagnostics ZIP.
Diagnostics ZIP không có trọng số/optimizer nên không đủ để resume training.

## Những phần được sử dụng lại

- Hoàn tất retrieval: dùng `train.sft.lexical.retrieval.json`, không gọi BM25
  lần nữa. `fit` kiểm tra câu hỏi/ID, code, model, cấu hình retrieval, index.
- Retrieval dở dang: journal `.checkpoint.jsonl` và `.meta.json` được mang theo;
  chỉ truy vấn các câu còn thiếu. Dòng cuối ghi dở được cắt bỏ khi mở journal.
- Training: khôi phục checkpoint hợp lệ có `global_step` lớn nhất, gồm adapter,
  optimizer, scheduler, scaler, Trainer state và RNG của **cả hai rank**.
  Checkpoint mới hơn nhưng ghi dở không nằm trong snapshot đã xác minh.
- BM25 LRU cache điểm từng từ là RAM của tiến trình, không được lưu sang phiên
  khác. Kết quả retrieval theo câu hỏi và journal mới là phần tái sử dụng lâu dài.

Resume khóa đúng commit trong manifest. Bản sửa RAM trong Python chỉ có tác dụng
khi commit được chạy có bản sửa; supervisor RAM trong notebook mới vẫn hoạt động
với commit cũ. Không tự thay loss/training code của optimizer cũ. Code cũ train
1 GPU bị chặn; chuyển sang `retrieval` sẽ giữ kết quả BM25 nhưng train lại với
2 GPU, không phải resume optimizer cũ. Import cache khác commit chỉ chấp nhận
các source hash đã audit và toàn bộ dependency retrieval còn khớp.

### Lỗi resume trong log (12): bitsandbytes CUDA prefetch

Log mới xác nhận chọn đúng output, khóa commit `6e1eb19`, tái sử dụng retrieval
và gọi `fit --resume .../checkpoint-1218` với hai rank T4. Sau đó bitsandbytes
thoát ở `/src/csrc/pythonInterface.cpp:400`. Trong
[bitsandbytes 0.45.5](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.45.5/csrc/pythonInterface.cpp#L393-L401),
dòng này là `cudaMemPrefetchAsync`, không phải thông báo hết RAM.

Nguyên nhân phù hợp với đường chạy resume: tensor optimizer được deserialize
có thể giữ thuộc tính `is_paged`/`page_deviceid`, trong khi bộ nhớ mới là tensor
CUDA thông thường. Hàm
[load_state_dict](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.45.5/bitsandbytes/optim/optimizer.py)
không tái tạo allocation managed cho các tensor này; bước optimizer tiếp theo
vẫn thử prefetch theo cờ đã lưu. Chưa kiểm chứng nguyên nhân trên checkpoint
thực tế ở máy sửa code vì không có CUDA/checkpoint tải về.

Notebook mới nhúng `scripts/bnb_resume_compat.py` thành `sitecustomize.py` riêng
trong `/kaggle/working/legalqa_bnb_resume_runtime`. Chỉ worker torchrun cài hook,
đúng bitsandbytes 0.45.5. Sau khi loader gốc kiểm tra/nạp optimizer, hook copy
tensor còn cờ paged sang bộ nhớ CUDA thông thường ở GPU của parameter, giữ dtype,
shape và giá trị; bỏ cờ paged lỗi. Moment 8-bit, quantization metadata, step,
learning rate, scheduler, scaler và RNG không bị khởi tạo lại. State mới chưa
từng được lưu vẫn do allocator gốc tạo. State phục hồi sẽ ở VRAM thay cho paging;
số tensor/byte và GPU của từng rank được ghi trong log và
`runtime_compat/bnb_resume_rank*.json`, kèm hash mã hook.

Hook nằm ngoài package `legalqa` nên áp dụng được ngay cả khi notebook khóa
commit cũ, không đổi code/config hash của checkpoint hoặc cache. Để sửa lỗi
này, import lại **notebook mới** vào Kaggle, giữ output cũ và `INPUT_MODE='auto'`
(hoặc `resume`); không cần bỏ checkpoint hay chuyển sang train mới.

Trước khi nạp model lớn, notebook chạy `scripts/check_bnb_resume.py` trên hai GPU:
optimizer paged nhỏ thực hiện một bước, save/load lên GPU như Trainer, thực hiện
bước tiếp theo và so cả parameter lẫn state với nhánh chạy liên tục. Cần thấy
`BNB_RESUME_SMOKE_OK` cho rank 0 và rank 1. Nếu kiểm tra CUDA này thất bại,
notebook dừng sớm. Đây là kiểm tra optimizer nhỏ; vẫn cần xác nhận lượt train thật
vượt step 1218. Kiểm thử CPU chỉ xác nhận routing, dữ liệu state và hook, không
thay thế kiểm tra CUDA.

## RAM và hai GPU

Đọc/import cache bằng [ijson](https://github.com/ICRAR/ijson): giữ từng bản ghi
context thay vì toàn bộ JSON ở cả hai worker. Token được đổi thành int32/uint8
ngay khi tạo, trả về đúng thứ tự QA trước đây. Với trần 5.600 × 8.192 token,
ba mảng token chiếm tối đa khoảng 394 MiB/worker, chưa gồm object overhead,
tokenizer, model và bộ nhớ tạm. Hai worker nạp model lần lượt để tránh hai đợt
staging trọng số trên CPU cùng lúc. Không cắt ngắn đáp án hay giảm top-k.

Vẫn dùng NF4, fused linear cross entropy, gradient checkpointing và DDP;
`1 × 4 × 2 = 8` mẫu/batch hiệu dụng. Tắt pinned host buffers của DataLoader.
Đây là hai replica train đồng bộ, không cộng hai VRAM thành một GPU 32 GB.
BM25 lexical chạy CPU, hai GPU có công việc khi QLoRA bắt đầu.

Ngưỡng mặc định: còn dưới 3.072 MiB RAM thì hai rank thống nhất dừng ở optimizer
step kế tiếp và lưu checkpoint; supervisor kiểm tra mỗi 2 giây, dừng process
group dưới 1.536 MiB để export snapshot cuối cùng. RAM khả dụng lấy mức nhỏ hơn
của `/proc/meminfo` và giới hạn cgroup. Hard stop có thể mất các step chưa lưu;
checkpoint hoàn chỉnh trước đó vẫn được chọn khi resume. Không thể bảo đảm
không có OOM do allocation đột ngột/VRAM/hạ tầng trước khi đo chạy thật.

Log cần thấy `rank=0/2 ... cuda:0` và `rank=1/2 ... cuda:1`.
Sau lượt fit kết thúc có kiểm soát, `sft/distributed_training.json` phải có
world_size=2, device khác nhau, global_step bằng nhau >0, peak_allocated_bytes >0.
Code kiểm tra cả DDP wrapper và bằng chứng hai rank trước khi công nhận.

## Kiểm chứng

CPU regression (cài requirements hoặc tối thiểu numpy + ijson cho bộ test CPU):

```text
python -B -m unittest discover -s tests -q
```

Các test thực thi discovery với mount đổi tên/lồng nhiều cấp; giả lập snapshot
phiên 1 rồi restore phiên 2, xác nhận không gọi retrieve, không đổi cache và chọn
checkpoint-20 thay vì checkpoint-30 ghi dở; kiểm tra journal dở dang; so token,
labels, masks, thứ tự QA; từ chối cache sai fingerprint/ID/câu hỏi/JSON; kiểm tra
RAM container và supervisor dừng cả process group khi thiếu RAM/hết giờ.

Kiểm tra bộ đọc trên fixture 16.786.107 byte đo peak Python khoảng 352 KB bằng
tracemalloc. Đây là phép đo riêng reader, không phải tổng RSS hay VRAM train.

Chưa chạy QLoRA thực tế trên hai T4 ở máy sửa code (không có torch/CUDA). Trước
full run, dùng smoke config `training.max_examples=8` như `docs/lexical_ddp.md`,
giữ world_size=2, seq=8192, batch và loss. Để kiểm tra resume GPU, đặt
`LEGALQA_DEADLINE` thành `time.time()+180` riêng cho lệnh fit đầu: callback sẽ
dừng/lưu sau optimizer step đầu tiên. Fit lần hai bỏ deadline đó, truyền
`--resume <checkpoint đã lưu>` và cùng output/config. So global_step hai rank
trong proof, xác nhận bước tiếp theo và hai adapter epoch hoàn chỉnh.

Link output người dùng cung cấp: https://www.kaggle.com/code/lighth/legalqa-main-01-qlora-train.
Phiên sửa code chưa đọc được manifest từ link này; chưa xác nhận commit hoặc
tính đầy đủ của checkpoint thực tế. Notebook sẽ xác minh các file khi được
gắn vào Input, không mặc định coi một link notebook là output hợp lệ.
