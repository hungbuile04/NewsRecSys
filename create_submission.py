import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
import numpy as np
import json
from tqdm import tqdm

# ======================================================
# 1. Load model architecture consistently with training
# ======================================================
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(
                input_dim, hidden_dim
            ),  # in: (batch_size, seq_len, input_dim), out: (batch_size, seq_len, hidden_dim)
            nn.Tanh(),  # in: (batch_size, seq_len, hidden_dim), out: (batch_size, seq_len, hidden_dim)
            nn.Linear(
                hidden_dim, 1, bias=False
            ),  # in: (batch_size, seq_len, hidden_dim), out: (batch_size, seq_len, 1)
            nn.Softmax(dim=-2),
        )
        self.attention.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attention_weight = self.attention(input)
        return input * attention_weight


class PLMBasedNewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained="bert-base-uncased",
        multihead_attn_num_heads=16,
        additive_attn_hidden_dim=200,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)
        plm_hidden = AutoConfig.from_pretrained(pretrained).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=plm_hidden, num_heads=multihead_attn_num_heads, batch_first=True
        )

        self.additive_attention = AdditiveAttention(plm_hidden, additive_attn_hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        V = self.plm(input_ids, attention_mask=attention_mask).last_hidden_state
        V2, _ = self.multihead_attention(V, V, V)
        A = self.additive_attention(V2)
        return torch.sum(A, dim=1)        # [batch, hidden]


# ======================================================
# 2. Load checkpoint
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = "model/checkpoint-2943"
pretrained_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

news_encoder = PLMBasedNewsEncoder(pretrained=pretrained_name)
state_dict = torch.load(f"{model_dir}/pytorch_model.bin", map_location=DEVICE)
news_encoder.load_state_dict(state_dict, strict=False)
news_encoder.to(DEVICE).eval()

# ======================================================
# 3. Load news.tsv and encode all news
# ======================================================
def load_news(news_path):
    news_index = {}
    with open(news_path, "r") as f:
        for line in f:
            nid, category, subcat, title, abstract, url, title_ent, abs_ent = line.strip("\n").split("\t")
            news_index[nid] = title
    return news_index

def encode_news(news_index):
    news_vecs = {}
    for nid, title in tqdm(news_index.items(), desc="Encoding news"):
        encoded = tokenizer(title, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        mask = encoded["attention_mask"].to(DEVICE)
        vec = news_encoder(input_ids, attention_mask=mask)
        news_vecs[nid] = vec.detach().cpu()
    return news_vecs


# ======================================================
# 4. Load behaviors.tsv and compute scores
# ======================================================
def generate_prediction(behaviors_path, news_vecs, outfile="prediction.txt"):
    out = open(outfile, "w")

    with open(behaviors_path, "r") as f:
        for line in tqdm(f, desc="Predicting"):
            parts = line.strip("\n").split("\t")
            imp_id = parts[0]
            history = parts[3].split() if parts[3] != "" else []
            impressions = parts[4].split()

            # user vector = mean pooling of history
            if len(history) > 0:
                hist_vecs = [news_vecs[n] for n in history if n in news_vecs]
                user_vec = torch.mean(torch.cat(hist_vecs, dim=0), dim=0)
            else:
                user_vec = torch.zeros_like(list(news_vecs.values())[0][0])

            user_vec = user_vec.unsqueeze(0)

            # candidate ids
            cand_ids = [impr.split("-")[0] for impr in impressions]
            scores = []

            for nid in cand_ids:
                if nid in news_vecs:
                    s = torch.cosine_similarity(user_vec, news_vecs[nid], dim=1).item()
                else:
                    s = 0.0
                scores.append(s)

            # ranking: larger score → smaller rank number
            order = np.argsort(scores)[::-1]
            ranks = np.zeros(len(scores), dtype=np.int32)
            ranks[order] = np.arange(1, len(scores) + 1)

            # convert list to EXACT required format: no spaces
            rank_str = json.dumps(ranks.tolist(), separators=(',', ':'))

            out.write(f"{imp_id} {rank_str}\n")

    out.close()


# ======================================================
# RUN
# ======================================================
news_path = "/Users/buihung/RecSys/data/MINDlarge_test/news.tsv" #Thay đường dẫn đến file data đầu vào ở đây
behaviors_path = "/Users/buihung/RecSys/data/MINDlarge_test/behaviors.tsv"

news_index = load_news(news_path)
news_vecs = encode_news(news_index)
generate_prediction(behaviors_path, news_vecs, outfile="prediction.txt")
