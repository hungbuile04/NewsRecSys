import os
import glob
import re
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
# Tắt WANDB nếu không dùng để tránh lỗi login trên Colab
os.environ["WANDB_DISABLED"] = "true"

import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_utils import get_last_checkpoint

from config.config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer

def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device) -> RecMetrics:
    net.eval()
    EVAL_BATCH_SIZE = 1
    # Tối ưu num_workers cho validation
    eval_dataloader = DataLoader(
        eval_mind_dataset, 
        batch_size=EVAL_BATCH_SIZE, 
        pin_memory=True,
        num_workers=2 
    )

    val_metrics_list: list[RecMetrics] = []
    # Dùng tqdm để theo dõi tiến độ eval
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset", leave=False):
        # Inference
        batch["news_histories"] = batch["news_histories"].to(device)
        batch["candidate_news"] = batch["candidate_news"].to(device)
        batch["target"] = batch["target"].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)

        # Convert To Numpy
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true: torch.Tensor = batch["target"].flatten().cpu().to(torch.int).numpy()

        # Calculate Metrics
        val_metrics_list.append(RecEvaluator.evaluate_all(y_true, y_score))

    rec_metrics = RecMetrics(
        **{
            "ndcg_at_10": np.average([metrics_item.ndcg_at_10 for metrics_item in val_metrics_list]),
            "ndcg_at_5": np.average([metrics_item.ndcg_at_5 for metrics_item in val_metrics_list]),
            "auc": np.average([metrics_item.auc for metrics_item in val_metrics_list]),
            "mrr": np.average([metrics_item.mrr for metrics_item in val_metrics_list]),
        }
    )

    return rec_metrics

def get_latest_run_dir(base_dir: Path) -> Path:
    """
    Tìm thư mục chạy gần nhất trong base_dir để resume training.
    Trả về None nếu không tìm thấy thư mục nào hợp lệ.
    """
    if not base_dir.exists():
        return None
    
    # Lấy tất cả subfolder
    all_subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not all_subdirs:
        return None
    
    # Sắp xếp theo thời gian chỉnh sửa (mới nhất đứng cuối)
    latest_dir = max(all_subdirs, key=os.path.getmtime)
    return latest_dir

def train(
    pretrained: str,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_len: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start Initialization")
    
    """
    0. Definite Parameters & Functions
    """
    EVAL_BATCH_SIZE = 1
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)

    """
    --- LOGIC RESUME ĐƯỢC THÊM VÀO ĐÂY ---
    """
    # Kiểm tra xem có thư mục cũ để resume không
    # Giả sử MODEL_OUTPUT_DIR là thư mục cha chứa các folder timestamp
    latest_dir = get_latest_run_dir(MODEL_OUTPUT_DIR)
    
    model_save_dir = None
    last_checkpoint = None
    
    if latest_dir:
        # Kiểm tra xem trong folder mới nhất có checkpoint nào không
        last_checkpoint = get_last_checkpoint(str(latest_dir))
        if last_checkpoint:
            logging.info(f"Found existing checkpoint at {last_checkpoint}. Will resume training.")
            model_save_dir = latest_dir
        else:
            logging.info(f"Found directory {latest_dir} but no checkpoint. Creating new run.")
    
    # Nếu không tìm thấy checkpoint hợp lệ, tạo folder mới theo timestamp
    if model_save_dir is None:
        model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created new training directory: {model_save_dir}")

    """
    1. Init Model
    """
    logging.info("Initialize Model")
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    )

    """
    2. Load Data & Create Dataset
    """
    logging.info("Initialize Dataset")

    train_news_df = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
    train_behavior_df = read_behavior_df(MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv")
    train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio, history_size, device)

    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    """
    3. Train
    """
    logging.info("Training Start")
    
    training_args = TrainingArguments(
        output_dir=str(model_save_dir),      # Folder output đã xử lý logic resume
        logging_strategy="steps",
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="epoch",
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        logging_dir=LOG_OUTPUT_DIR,
        logging_steps=10,                    # Giảm log lại chút cho đỡ rối (10 bước log 1 lần)
        report_to="none",
        dataloader_num_workers=2,            # Tăng tốc độ load dữ liệu
        disable_tqdm=False,                  # Hiện thanh progress
    )

    trainer = Trainer(
        model=nrms_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # QUAN TRỌNG: Resume nếu tìm thấy checkpoint
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    """
    4. Evaluate model by Validation Dataset
    """
    logging.info("Evaluation Start")
    # Lưu model cuối cùng
    logging.info("Evaluation Start")

    # Đường dẫn lưu trên Google Drive
    drive_save_dir = "/content/drive/MyDrive/my_model/final_model"

    # Tạo thư mục nếu chưa có
    os.makedirs(drive_save_dir, exist_ok=True)

    # Lưu model vào Drive
    trainer.save_model(output_dir=drive_save_dir)
    logging.info(f"Model saved to {drive_save_dir}")

    # Đánh giá
    metrics = evaluate(trainer.model, eval_dataset, device)
    logging.info(f"Evaluation Metrics: {metrics.dict()}")
    

    metrics = evaluate(trainer.model, eval_dataset, device)
    logging.info(f"Evaluation Metrics: {metrics.dict()}")


@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)
        train(
            cfg.pretrained,
            cfg.npratio,
            cfg.history_size,
            cfg.batch_size,
            cfg.gradient_accumulation_steps,
            cfg.epochs,
            cfg.learning_rate,
            cfg.weight_decay,
            cfg.max_len,
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e # Raise lại để biết lỗi gì trên Colab

if __name__ == "__main__":
    main()