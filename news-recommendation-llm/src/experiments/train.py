import hydra
import numpy as np
import torch
from pathlib import Path
from config.config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MODEL_OUTPUT_DIR
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from torch import nn
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed

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
    
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    run_path = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
    run_name = run_path.name

    try:
        drive_base = Path("/content/drive/MyDrive") / "recsys_model"
        drive_base.mkdir(parents=True, exist_ok=True)
        model_save_dir = drive_base / run_name
    except Exception:
        model_save_dir = run_path
    model_save_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Initialize Model")
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(device)

    data_path = Path(data_dir) 
    logging.info(f"Loading Data from: {data_path}")
    train_news_df = read_news_df(data_path / "news.tsv")
    train_behavior_df = read_behavior_df(data_path / "behaviors.tsv")
    
    logging.info("Initializing Dataset (Pre-tokenizing)...")
    train_dataset = MINDTrainDataset(
        behavior_df=train_behavior_df,
        news_df=train_news_df,
        tokenizer=tokenizer,
        max_len=max_len,
        npratio=npratio,
        history_size=history_size,
        device=device
    )

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
        num_train_epochs=epochs,
        dataloader_num_workers=4,
        fp16=True,
        
        # BẬT Checkpointing để tiết kiệm RAM (cho phép Batch 96)
        gradient_checkpointing=True,  
        
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
        logging_dir=str(LOG_OUTPUT_DIR),
        report_to="none",
    )

    # Patch Gradient Checkpointing thủ công cho Custom Model
    if training_args.gradient_checkpointing:
        logging.info("Applying Gradient Checkpointing Patch for NRMS (Manual Mode)")
        def enable_checkpointing(gradient_checkpointing_kwargs=None, **kwargs):
            distilbert = nrms_net.news_encoder.plm
            distilbert.gradient_checkpointing = True
            if hasattr(distilbert, "config"):
                distilbert.config.gradient_checkpointing = True
                if hasattr(distilbert.config, "use_cache"):
                    distilbert.config.use_cache = False
            logging.info("Gradient Checkpointing has been MANUALLY ENABLED.")
        nrms_net.gradient_checkpointing_enable = enable_checkpointing

    trainer = Trainer(
        model=nrms_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
    )
    
    if resume_checkpoint is not None:
        logging.info(f"Resuming training from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        logging.info("Starting training from scratch")
        trainer.train()

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
        raise e

if __name__ == "__main__":
    main()