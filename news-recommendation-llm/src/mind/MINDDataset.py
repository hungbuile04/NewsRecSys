import random
from typing import Callable, List

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

EMPTY_NEWS_ID, EMPTY_IMPRESSION_IDX = "EMPTY_NEWS_ID", -1


class MINDTrainDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        tokenizer,  # Thay vì transform_fn, ta truyền trực tiếp tokenizer
        max_len: int, # Truyền max_len để padding
        npratio: int,
        history_size: int,
        device: torch.device = torch.device("cpu") # Mặc định để CPU để tiết kiệm VRAM cho Model
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.npratio: int = npratio
        self.history_size: int = history_size
        self.device = device

        # ---------------------------------------------------------------------
        # TỐI ƯU HÓA: PRE-TOKENIZATION (Tokenize toàn bộ News 1 lần duy nhất)
        # ---------------------------------------------------------------------
        print(f"🔄 [MINDTrainDataset] Pre-tokenizing {len(self.news_df)} news titles...")
        
        # 1. Tạo mapping News ID -> Index
        # Mặc định index cuối cùng sẽ là index dành cho EMPTY_NEWS_ID (padding)
        self.news_ids = self.news_df["news_id"].to_list()
        self.news_titles = self.news_df["title"].to_list()
        
        self.news_id_to_index = {nid: i for i, nid in enumerate(self.news_ids)}
        
        # Thêm mục cho EMPTY_NEWS_ID
        self.empty_news_index = len(self.news_titles) # Index cuối cùng
        self.news_id_to_index[EMPTY_NEWS_ID] = self.empty_news_index
        
        # Thêm tiêu đề rỗng cho Empty News
        all_titles_to_tokenize = self.news_titles + [""]

        # 2. Tokenize batch toàn bộ
        # Lưu ý: Return trực tiếp PyTorch Tensor
        self.tokenized_news = tokenizer(
            all_titles_to_tokenize,
            return_tensors="pt",
            max_length=max_len,
            padding="max_length",
            truncation=True
        )
        
        # Đưa input_ids vào biến class (Lưu trên CPU RAM là tốt nhất để tránh OOM GPU)
        # Nếu RAM dư dả và muốn cực nhanh thì có thể .to(device), nhưng cẩn thận VRAM.
        self.news_input_ids = self.tokenized_news["input_ids"]
        # self.news_attn_mask = self.tokenized_news["attention_mask"] # Nếu model cần mask

        print("✅ [MINDTrainDataset] Pre-tokenization Completed.")

        # ---------------------------------------------------------------------
        # Pre-process Behavior (Giữ nguyên logic cũ)
        # ---------------------------------------------------------------------
        self.behavior_df = self.behavior_df.with_columns(
            [
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 1])
                .alias("clicked_idxes"),
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 0])
                .alias("non_clicked_idxes"),
            ]
        )

    def __getitem__(self, behavior_idx: int) -> dict:
        """
        Returns:
            torch.Tensor: history_news
            torch.Tensor: candidate_news
            torch.Tensor: labels
        """
        # Extract Values
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )
        
        poss_idxes, neg_idxes = (
            behavior_item["clicked_idxes"].to_list()[0],
            behavior_item["non_clicked_idxes"].to_list()[0],
        )
        
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )

        # Sampling
        if len(poss_idxes) == 0: # Fallback nếu không có click nào (dù hiếm)
             sample_poss_idxes = [len(impressions)-1]
        else:
             sample_poss_idxes = random.sample(poss_idxes, 1)
             
        sample_neg_idxes = self.__sampling_negative(neg_idxes, self.npratio)

        sample_impression_idxes = sample_poss_idxes + sample_neg_idxes
        random.shuffle(sample_impression_idxes)

        sample_impressions = impressions[sample_impression_idxes]

        # Extract IDs
        candidate_news_ids = [imp_item["news_id"] for imp_item in sample_impressions]
        labels = [imp_item["clicked"] for imp_item in sample_impressions]
        
        # History slicing
        history_news_ids = history[-self.history_size:]
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

        # ---------------------------------------------------------------------
        # LOGIC MỚI: LOOKUP TENSOR TRỰC TIẾP (KHÔNG TOKENIZE LẠI)
        # ---------------------------------------------------------------------
        
        # 1. Map ID -> Index
        # Sử dụng .get(nid, self.empty_news_index) để an toàn nếu gặp ID lạ
        candidate_indices = [self.news_id_to_index.get(nid, self.empty_news_index) for nid in candidate_news_ids]
        history_indices = [self.news_id_to_index.get(nid, self.empty_news_index) for nid in history_news_ids]

        # 2. Slice Tensor từ bộ nhớ đã cache
        candidate_news_tensor = self.news_input_ids[torch.tensor(candidate_indices)]
        history_news_tensor = self.news_input_ids[torch.tensor(history_indices)]
        
        labels_tensor = torch.tensor(labels).argmax()

        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "target": labels_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)

    def __sampling_negative(self, neg_idxes: list[int], npratio: int) -> list[int]:
        if len(neg_idxes) < npratio:
            return neg_idxes + [EMPTY_IMPRESSION_IDX] * (npratio - len(neg_idxes))

        return random.sample(neg_idxes, self.npratio)


class MINDValDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        tokenizer, 
        max_len: int,
        history_size: int,
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.history_size: int = history_size

        # ---------------------------------------------------------------------
        # TỐI ƯU HÓA: PRE-TOKENIZATION CHO VAL
        # ---------------------------------------------------------------------
        print(f"🔄 [MINDValDataset] Pre-tokenizing {len(self.news_df)} news titles...")
        
        self.news_ids = self.news_df["news_id"].to_list()
        self.news_titles = self.news_df["title"].to_list()
        self.news_id_to_index = {nid: i for i, nid in enumerate(self.news_ids)}
        
        self.empty_news_index = len(self.news_titles)
        self.news_id_to_index[EMPTY_NEWS_ID] = self.empty_news_index
        
        all_titles_to_tokenize = self.news_titles + [""]

        self.tokenized_news = tokenizer(
            all_titles_to_tokenize,
            return_tensors="pt",
            max_length=max_len,
            padding="max_length",
            truncation=True
        )
        self.news_input_ids = self.tokenized_news["input_ids"]
        print("✅ [MINDValDataset] Pre-tokenization Completed.")

    def __getitem__(self, behavior_idx: int) -> dict:
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )

        candidate_news_ids = [imp_item["news_id"] for imp_item in impressions]
        labels = [imp_item["clicked"] for imp_item in impressions]
        
        history_news_ids = history[-self.history_size:]
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

        # Lookup Index
        candidate_indices = [self.news_id_to_index.get(nid, self.empty_news_index) for nid in candidate_news_ids]
        history_indices = [self.news_id_to_index.get(nid, self.empty_news_index) for nid in history_news_ids]

        # Slice Tensor
        candidate_news_tensor = self.news_input_ids[torch.tensor(candidate_indices)]
        history_news_tensor = self.news_input_ids[torch.tensor(history_indices)]
        
        one_hot_label_tensor = torch.tensor(labels)

        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "target": one_hot_label_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)


if __name__ == "__main__":
    from const.path import MIND_SMALL_VAL_DATASET_DIR
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from utils.logger import logging
    from utils.random_seed import set_random_seed

    from src.mind.dataframe import read_behavior_df, read_news_df

    set_random_seed(42)

    # Test code cho logic mới
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    logging.info("Load Data")
    behavior_df, news_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv"), read_news_df(
        MIND_SMALL_VAL_DATASET_DIR / "news.tsv"
    )

    logging.info("Init MINDTrainDataset")
    # Lưu ý: Tham số đã thay đổi
    train_dataset = MINDTrainDataset(
        behavior_df, 
        news_df, 
        tokenizer=tokenizer, 
        max_len=30, 
        npratio=4, 
        history_size=20
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    logging.info("Start Iteration")
    for batch in train_dataloader:
        logging.info(f"{batch}")
        break