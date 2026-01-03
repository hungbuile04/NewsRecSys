import hydra
import numpy as np
import torch
from pathlib import Path
from config.config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MODEL_OUTPUT_DIR
# from evaluation.RecEvaluator import RecEvaluator, RecMetrics # Tạm bỏ nếu không eval
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from torch import nn
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
# from utils.text import create_transform_fn_from_pretrained_tokenizer # Không cần dùng nữa

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
    resume_checkpoint: str = None,
    data_dir: str = "/content/data/large_full",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start")
    """
    0. Definite Parameters & Functions
    """
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    
    # --- THAY ĐỔI 1: Khởi tạo Tokenizer trực tiếp ---
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # create a run name
    run_path = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
    run_name = run_path.name

    # Mount Google Drive logic (như cũ)
    try:
        drive_base = Path("/content/drive/MyDrive") / "recsys_model"
        drive_base.mkdir(parents=True, exist_ok=True)
        model_save_dir = drive_base / run_name
    except Exception:
        model_save_dir = run_path

    model_save_dir.mkdir(parents=True, exist_ok=True)

    """
    1. Init Model
    """
    logging.info("Initialize Model")
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(device)

    """
    2. Load Data & Create Dataset
    """
    data_path = Path(data_dir) 
    logging.info(f"Loading Data from: {data_path}")

    train_news_df = read_news_df(data_path / "news.tsv")
    train_behavior_df = read_behavior_df(data_path / "behaviors.tsv")
    
    # --- THAY ĐỔI 2: Truyền Tokenizer và max_len vào Dataset ---
    logging.info("Initializing Dataset (Pre-tokenizing)...")
    train_dataset = MINDTrainDataset(
        behavior_df=train_behavior_df,
        news_df=train_news_df,
        tokenizer=tokenizer,      # <--- Mới
        max_len=max_len,          # <--- Mới
        npratio=npratio,
        history_size=history_size,
        device=device # Truyền device nếu muốn (nhưng nên để CPU xử lý data)
    )

    eval_dataset = None # Bỏ qua eval

    """
    3. Train
    """
    logging.info("Training Start")
    training_args = TrainingArguments(
        output_dir=str(model_save_dir),
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        num_train_epochs=epochs,         # Dùng biến epochs từ config
        dataloader_num_workers=4,        # Tăng lên 4 worker để tận dụng CPU
        fp16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
        logging_dir=str(LOG_OUTPUT_DIR),
        report_to="none",
    )

    # --- ĐOẠN GRADIENT CHECKPOINTING BẠN HỎI ---
    if training_args.gradient_checkpointing:
        logging.info("Applying Gradient Checkpointing Patch for NRMS (Manual Mode)")
        
        def enable_checkpointing(gradient_checkpointing_kwargs=None, **kwargs):
            # Lấy model gốc (DistilBERT) nằm sâu bên trong NRMS
            distilbert = nrms_net.news_encoder.plm
            
            # Bật cờ checkpointing
            distilbert.gradient_checkpointing = True
            if hasattr(distilbert, "config"):
                distilbert.config.gradient_checkpointing = True
                # TẮT CACHE LÀ BẮT BUỘC khi dùng gradient checkpointing
                if hasattr(distilbert.config, "use_cache"):
                    distilbert.config.use_cache = False
            
            logging.info("Gradient Checkpointing has been MANUALLY ENABLED.")

        # Gán hàm này cho model để Trainer gọi được
        nrms_net.gradient_checkpointing_enable = enable_checkpointing

    trainer = Trainer(
        model=nrms_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    if resume_checkpoint is not None:
        logging.info(f"Resuming training from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        logging.info("Starting training from scratch")
        trainer.train()

    # Lưu model cuối
    final_save_path = model_save_dir / "final_model"
    trainer.save_model(str(final_save_path))
    logging.info(f"Final model saved at {final_save_path}")

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
            cfg.resume_checkpoint,
            cfg.data_dir
        )
    except Exception as e:
        logging.error(e)
        raise e # Raise lỗi để còn debug nếu chết

if __name__ == "__main__":
    main()