# Báo cáo baseline 0.1 – UIT DSC 2026 LegalQA

## 1. Dữ liệu thực tế

| Thành phần | Kết quả kiểm tra |
|---|---:|
| Train | 7.000 câu hỏi–đáp án |
| Public | 1.000 câu hỏi, `answer = null` |
| Corpus | 8.532 văn bản |
| Kích thước corpus sau giải nén | 488.693.666 byte |
| Passage rỗng | 20 |
| Chunk sau tiền xử lý | 240.911 |
| Độ dài câu hỏi Train, trung vị / P90 | 19 / 28 từ |
| Độ dài đáp án Train, trung vị / P90 | 312 / 576 từ |

Chunk mặc định dài tối đa 620 từ, overlap 100 từ. Pipeline ưu tiên ranh giới `Điều` và `Phụ lục` trước khi dùng cửa sổ từ.

## 2. Pipeline đã triển khai

```text
Question
  ├─ BM25 FTS5 trên 8 token hiếm nhất → Top 12 chunk
  │    └─ rerank BM25 + keyword coverage + bi/trigram coverage
  │          └─ chọn cửa sổ tối đa 520 từ → extractive answer
  └─ BM25 trên câu hỏi Train → nearest question
       └─ chỉ dùng answer KNN khi similarity >= 0,72
```

Chế độ `hybrid` chọn KNN cho câu gần trùng Train, còn lại dùng extractive. Baseline không dùng model học sâu, API hay dữ liệu ngoài.

## 3. Validation

Leave-one-out trên 300 mẫu Train, seed 2026:

| Mode | METEOR exact-token gần đúng | ROUGE-L theo tokenizer BTC | Route |
|---|---:|---:|---|
| Extractive | 0,3602 | 0,4130 | 300 extractive |
| Hybrid | **0,3753** | **0,4274** | 273 extractive + 27 KNN |

Đây không phải điểm Codabench. METEOR nội bộ bỏ stemming/WordNet tiếng Anh để chạy không dependency; với văn bản tiếng Việt, sai khác dự kiến nhỏ nhưng vẫn phải coi Codabench là nguồn điểm chính thức.

## 4. Public prediction

- Đã sinh đủ 1.000/1.000 ID.
- Route: 897 extractive, 103 KNN.
- Không có answer rỗng hoặc `null`.
- Thời gian sinh: 236,08 giây trong môi trường CPU hiện tại.
- Tệp: `artifacts/submission_hybrid.json`.

## 5. Phát hiện về scoring program

- METEOR là metric chính và dùng `str.split()` trực tiếp, không word-segment tiếng Việt.
- ROUGE-L gọi tokenizer mặc định của `rouge_score` với regex chỉ giữ `[a-z0-9]`. Do đó nhiều chữ cái tiếng Việt có dấu bị loại hoặc tách thành mảnh. Evaluator nhẹ trong project mô phỏng hành vi này.
- Không nên tối ưu quyết định mô hình theo ROUGE-L trước METEOR.

## 6. Bước nâng cấp đề xuất

Ưu tiên theo thứ tự thực nghiệm:

1. Đo Recall@K của retrieval bằng cách gán pseudo-context: tìm các span dài từ answer trong corpus.
2. Ghép BM25 với `AITeamVN/Vietnamese_Embedding_v2` hoặc `intfloat/multilingual-e5-base`.
3. Rerank Top 50 bằng `AITeamVN/Vietnamese_Reranker` hoặc `Qwen/Qwen3-Reranker-0.6B`.
4. Fine-tune generator 1,5B bằng `question + retrieved context → answer`; kiểm soát tổng tham số toàn hệ thống dưới 4B.
5. Tối ưu độ dài answer theo nhóm câu hỏi (`mẫu`, `mức phạt`, `thủ tục`, `trách nhiệm`) thay vì dùng một ngưỡng 520 từ cho tất cả.

