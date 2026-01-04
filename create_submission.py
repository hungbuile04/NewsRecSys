import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
import numpy as np
import json
import os
import argparse
from tqdm import tqdm

# ======================================================
# 1. Model Architecture (FIXED)
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
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Softmax(dim=-2),
        )
        self.attention.apply(init_weights)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        context = self.attention(input_val)
        output = torch.sum(input_val * context, dim=1)
        return output

class PLMBasedNewsEncoder(nn.Module):
    def __init__(self, pretrained: str = "distilbert-base-uncased"):
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)
        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=plm_hidden_size, num_heads=16, batch_first=True
        )
        self.additive_attention = AdditiveAttention(plm_hidden_size, 200)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        V = self.plm(input_val).last_hidden_state
        multihead_attn_output, _ = self.multihead_attention(V, V, V)
        additive_attn_output = self.additive_attention(multihead_attn_output)
        
        # [QUAN TRỌNG] Đã xóa torch.sum(...) thừa ở đây
        # Trả về vector embedding [Batch, Hidden] chuẩn
        return additive_attn_output

class UserEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.additive_attention = AdditiveAttention(hidden_dim, 200)

    def forward(self, news_histories: torch.Tensor, news_encoder: nn.Module) -> torch.Tensor:
        batch_size, hist_size, seq_len = news_histories.size()
        news_histories = news_histories.view(batch_size * hist_size, seq_len)
        news_histories_encoded = news_encoder(news_histories)
        news_histories_encoded = news_histories_encoded.view(batch_size, hist_size, -1)
        user_vector = self.additive_attention(news_histories_encoded)
        return user_vector

class NRMS(nn.Module):
    def __init__(self, news_encoder: nn.Module, user_encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.hidden_size = hidden_size

# ======================================================
# 2. Logic
# ======================================================
def load_model(checkpoint_path: str, device: torch.device):
    print(f"🔄 Loading model from: {checkpoint_path}")
    pretrained = "distilbert-base-uncased"
    hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size)
    model = NRMS(news_encoder, user_encoder, hidden_size)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    return model, pretrained

def generate_submission(model, tokenizer, news_path, behaviors_path, output_path, device, max_len=30):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    print("🔄 Reading News Data...")
    news_titles = {}
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            nid, title = parts[0], parts[3]
            news_titles[nid] = title

    print(f"🔄 Encoding {len(news_titles)} news items...")
    news_vecs = {}
    batch_size = 128
    news_ids = list(news_titles.keys())
    
    for i in tqdm(range(0, len(news_ids), batch_size)):
        batch_ids = news_ids[i : i + batch_size]
        batch_titles = [news_titles[nid] for nid in batch_ids]
        
        inputs = tokenizer(
            batch_titles, 
            padding="max_length", 
            truncation=True, 
            max_length=max_len, 
            return_tensors="pt"
        ).input_ids.to(device)
        
        with torch.no_grad():
            vecs = model.news_encoder(inputs)
        
        for nid, vec in zip(batch_ids, vecs):
            news_vecs[nid] = vec

    print("🔄 Processing Behaviors & Generating Predictions...")
    
    with open(output_path, "w") as out:
        with open(behaviors_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                parts = line.strip().split("\t")
                imp_id = parts[0]
                history = parts[3].split(" ") if len(parts) > 3 and parts[3] else []
                impressions = parts[4].split(" ")
                
                hist_vecs = [news_vecs[nid].unsqueeze(0) for nid in history if nid in news_vecs]
                
                if hist_vecs:
                    user_vec = torch.mean(torch.cat(hist_vecs, dim=0), dim=0).unsqueeze(0)
                else:
                    user_vec = torch.zeros(1, model.hidden_size).to(device)

                cand_ids = [impr.split("-")[0] for impr in impressions]
                scores = []
                for nid in cand_ids:
                    if nid in news_vecs:
                        # Giờ đây news_vecs[nid] đã là vector [Hidden], unsqueeze(0) thành [1, Hidden]
                        # user_vec là [1, Hidden]
                        # Cosine Similarity chạy trên dim=1 sẽ hoạt động đúng
                        s = torch.cosine_similarity(user_vec, news_vecs[nid].unsqueeze(0), dim=1).item()
                    else:
                        s = 0.0
                    scores.append(s)

                order = np.argsort(scores)[::-1]
                ranks = np.zeros(len(scores), dtype=np.int32)
                ranks[order] = np.arange(1, len(scores) + 1)

                rank_str = json.dumps(ranks.tolist(), separators=(',', ':'))
                out.write(f"{imp_id} {rank_str}\n")

    print(f"✅ Done! Submission saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--news_path", type=str, required=True)
    parser.add_argument("--behaviors_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="prediction.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pretrained = load_model(args.checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    generate_submission(
        model, tokenizer, 
        args.news_path, 
        args.behaviors_path, 
        args.output_path, 
        device
    )