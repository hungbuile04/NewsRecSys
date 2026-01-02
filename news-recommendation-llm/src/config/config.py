from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    random_seed: int = 42
    pretrained: str = "distilbert-base-uncased"
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 32
    gradient_accumulation_steps: int = 8  # batch_size = 32 x 8 = 256
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_len: int = 30

    resume_checkpoint: Optional[str] = None  # resume checkpoint nếu có

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)