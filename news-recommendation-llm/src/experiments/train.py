import hydra
import numpy as np
import torch
from pathlib import Path
from config.config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MIND_LARGE_TRAIN_DATASET_DIR, MODEL_OUTPUT_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer


def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device) -> RecMetrics:
    net.eval()
    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)

    val_metrics_list: list[RecMetrics] = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
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
    logging.info("Start")
    """
    0. Definite Parameters & Functions
    """
    EVAL_BATCH_SIZE = 1
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)

    # create a run name
    run_path = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
    run_name = run_path.name

    # Try mount Google Drive (Colab). If available, save checkpoints directly to Drive.
    in_colab = False
    try:
        drive_base = Path("/content/drive/MyDrive") / "recsys_model"
        drive_base.mkdir(parents=True, exist_ok=True)
        model_save_dir = drive_base / run_name
        in_colab = True
    except Exception:
        model_save_dir = run_path

    model_save_dir.mkdir(parents=True, exist_ok=True)

    """
    1. Init Model
    """
    logging.info("Initialize Model")
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )

    """
    2. Load Data & Create Dataset
    """
    logging.info("Initialize Dataset")

    train_news_df = read_news_df(MIND_LARGE_TRAIN_DATASET_DIR / "news.tsv")
    train_behavior_df = read_behavior_df(MIND_LARGE_TRAIN_DATASET_DIR / "behaviors.tsv")
    train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio, history_size, device)

    eval_dataset = None
    # val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    # val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    # eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    """
    3. Train
    """
    logging.info("Training Start")
    training_args = TrainingArguments(
        output_dir=str(model_save_dir),
        logging_strategy="steps",
        logging_steps=100,                                             # Log mỗi 100 step
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="steps",                                         # Save checkpoint theo step thay vì epoch
        save_steps=1000,                                               # Save mỗi 1000 step
        num_train_epochs=2,                                            # Train bằng 2 epoch, có thể sửa ở config nhưng thôi làm thế này tiện hơn
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
        logging_dir=str(LOG_OUTPUT_DIR),
        report_to="none",
    )

    trainer = Trainer(
        model=nrms_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    """
    4. Evaluate model by Validation Dataset
    """
    logging.info("Training Completed.")             # Chỗ này không cần eval nữa vì train trên MIND Large full rồi
    # logging.info("Evaluation")
    # metrics = evaluate(trainer.model, eval_dataset, device)
    # logging.info(metrics.dict())

    # Lưu model cuối cùng
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
        )
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    main()
