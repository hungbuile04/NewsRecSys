# News Recommendation (NRMS + PLM) — README

Dự án này triển khai hệ thống recommendation (NRMS) sử dụng PLM (DistilBERT/BERT) để mã hóa news và học user-vector từ lịch sử. Bao gồm training, evaluation và script tạo submission. Kết quả nộp trên cuộc thi MIND News Recommendation Competition.

## Tính năng chính
- Pre-tokenize news để tăng tốc (see [`MINDTrainDataset`](src/mind/MINDDataset.py) and [`MINDValDataset`](src/mind/MINDDataset.py)).
- PLM-based news encoder: [`PLMBasedNewsEncoder`](src/recommendation/nrms/PLMBasedNewsEncoder.py).
- User encoder + NRMS model: [`UserEncoder`](src/recommendation/nrms/UserEncoder.py), [`NRMS`](src/recommendation/nrms/NRMS.py).
- Evaluation utilities: [`RecEvaluator`](src/evaluation/RecEvaluator.py).
- Inference / submission generator: [create_submission.py](create_submission.py).

## Tệp quan trọng
- Mã training: [`src/experiments/train.py`](src/experiments/train.py) (entrypoint: `train` / Hydra config in [`src/config/config.py`](src/config/config.py)).
- Dataset helper & pre-tokenize: [`src/mind/dataframe.py`](src/mind/dataframe.py), [`src/mind/MINDDataset.py`](src/mind/MINDDataset.py).
- Model components: [`src/recommendation/nrms/PLMBasedNewsEncoder.py`](src/recommendation/nrms/PLMBasedNewsEncoder.py), [`src/recommendation/nrms/AdditiveAttention.py`](src/recommendation/nrms/AdditiveAttention.py), [`src/recommendation/nrms/NRMS.py`](src/recommendation/nrms/NRMS.py).
- Inference / submission: [create_submission.py](create_submission.py) (load model + generate `prediction.txt`).
- Dataset download helper: [`dataset/download_mind.py`](dataset/download_mind.py).
- Project paths/constants: [`src/const/path.py`](src/const/path.py).
- Random seed util: [`src/utils/random_seed.py`](src/utils/random_seed.py).
- Quick eval example: [`src/experiments/evaluate_random.py`](src/experiments/evaluate_random.py).

## Yêu cầu
Xem `pyproject.toml` và `requirements.txt` (nếu có). PLM models dùng Hugging Face Transformers (e.g. `distilbert-base-uncased`).

## Thiết lập nhanh
1. Tạo môi trường và cài dependencies:
   - pip / poetry / uv: tùy cấu hình (`pyproject.toml`).
2. Tải dữ liệu MIND (ví dụ): chạy [`dataset/download_mind.py`](dataset/download_mind.py) hoặc chuẩn dữ liệu theo cấu trúc `/content/data/...`.

## Chạy training
Ví dụ chạy (Hydra được dùng trong `src/experiments/train.py`):
- Tham khảo entrypoint: [`src/experiments/train.py`](src/experiments/train.py) (hàm `train`).
- Ví dụ command (notebook đang dùng `uv run` trong nhiều nơi):
  - uv run python src/experiments/train.py data_dir='/path/to/data' batch_size=96 gradient_accumulation_steps=2 history_size=20 epochs=1
- Lưu ý gradient checkpointing patch trong file để tiết kiệm VRAM (xem nội dung trong [`src/experiments/train.py`](src/experiments/train.py)).

## Tạo submission / Inference
- Sử dụng [create_submission.py](create_submission.py):
  - python create_submission.py --news_path /path/news.tsv --behaviors_path /path/behaviors.tsv --checkpoint_path /path/pytorch_model.bin --output_path prediction.txt
- Script này dùng encoder model [`PLMBasedNewsEncoder`](src/recommendation/nrms/PLMBasedNewsEncoder.py) và cosine-similarity rank.

## Evaluation
- Có hàm evaluation helper: [`src/evaluation/RecEvaluator.py`](src/evaluation/RecEvaluator.py).
- Ví dụ evaluation random: [`src/experiments/evaluate_random.py`](src/experiments/evaluate_random.py).

## Kiểm thử
- Thư mục test: `test/` (unit tests cho encoder, attention, dataset, evaluation).
- Chạy:
  - pytest -q

---

Nếu cần bản README tiếng Anh hoặc muốn thêm badge / hướng dẫn deploy nhanh (Colab / Docker), báo để cập nhật.
