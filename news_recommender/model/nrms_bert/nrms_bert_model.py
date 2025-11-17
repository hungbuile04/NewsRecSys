import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        return attn_output

class NewsEncoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', 
                 num_attention_heads=16, 
                 news_embed_dim=400):
        super().__init__()
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size  # 768 for bert-base
        
        # Project BERT output to news embedding dimension
        self.projection = nn.Linear(bert_dim, news_embed_dim)
        
        # Multi-head self-attention
        self.multi_head_attention = MultiHeadSelfAttention(
            news_embed_dim, num_attention_heads
        )
        
        # Additive attention for aggregation
        self.additive_attention = nn.Sequential(
            nn.Linear(news_embed_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input_ids, attention_mask):
        # input_ids: (batch_size, seq_len)
        # BERT encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Get last hidden state: (batch_size, seq_len, 768)
        word_embeddings = bert_output.last_hidden_state
        
        # Project to news_embed_dim
        word_embeddings = self.projection(word_embeddings)
        word_embeddings = self.dropout(word_embeddings)
        
        # Multi-head self-attention
        word_embeddings = self.multi_head_attention(
            word_embeddings, 
            mask=~attention_mask.bool()
        )
        
        # Additive attention for word-level aggregation
        attention_weights = self.additive_attention(word_embeddings)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        news_vector = torch.sum(word_embeddings * attention_weights, dim=1)
        
        return news_vector  # (batch_size, news_embed_dim)

class UserEncoder(nn.Module):
    def __init__(self, news_embed_dim=400, num_attention_heads=16):
        super().__init__()
        # Multi-head self-attention for user history
        self.multi_head_attention = MultiHeadSelfAttention(
            news_embed_dim, num_attention_heads
        )
        
        # Additive attention for news-level aggregation
        self.additive_attention = nn.Sequential(
            nn.Linear(news_embed_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        
    def forward(self, news_vectors, mask=None):
        # news_vectors: (batch_size, num_clicked_news, news_embed_dim)
        
        # Multi-head self-attention
        user_vectors = self.multi_head_attention(news_vectors, mask=mask)
        
        # Additive attention for aggregation
        attention_weights = self.additive_attention(user_vectors)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                mask.unsqueeze(-1), float('-inf')
            )
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        user_vector = torch.sum(user_vectors * attention_weights, dim=1)
        
        return user_vector  # (batch_size, news_embed_dim)

class NRMSBert(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased',
                 num_attention_heads=16,
                 news_embed_dim=400):
        super().__init__()
        self.news_encoder = NewsEncoder(
            bert_model_name, 
            num_attention_heads, 
            news_embed_dim
        )
        self.user_encoder = UserEncoder(
            news_embed_dim, 
            num_attention_heads
        )
        
    def forward(self, clicked_news_input_ids, clicked_news_masks,
                candidate_news_input_ids, candidate_news_masks,
                clicked_mask=None):
        """
        clicked_news_input_ids: (batch_size, num_clicked, seq_len)
        clicked_news_masks: (batch_size, num_clicked, seq_len)
        candidate_news_input_ids: (batch_size, num_candidates, seq_len)
        candidate_news_masks: (batch_size, num_candidates, seq_len)
        clicked_mask: (batch_size, num_clicked) - True for padded positions
        """
        batch_size = clicked_news_input_ids.size(0)
        num_clicked = clicked_news_input_ids.size(1)
        num_candidates = candidate_news_input_ids.size(1)
        
        # Encode clicked news
        clicked_news_input_ids = clicked_news_input_ids.view(
            batch_size * num_clicked, -1
        )
        clicked_news_masks = clicked_news_masks.view(
            batch_size * num_clicked, -1
        )
        clicked_news_vectors = self.news_encoder(
            clicked_news_input_ids, 
            clicked_news_masks
        )
        clicked_news_vectors = clicked_news_vectors.view(
            batch_size, num_clicked, -1
        )
        
        # Encode user
        user_vector = self.user_encoder(
            clicked_news_vectors, 
            mask=clicked_mask
        )
        
        # Encode candidate news
        candidate_news_input_ids = candidate_news_input_ids.view(
            batch_size * num_candidates, -1
        )
        candidate_news_masks = candidate_news_masks.view(
            batch_size * num_candidates, -1
        )
        candidate_news_vectors = self.news_encoder(
            candidate_news_input_ids, 
            candidate_news_masks
        )
        candidate_news_vectors = candidate_news_vectors.view(
            batch_size, num_candidates, -1
        )
        
        # Click prediction (dot product)
        scores = torch.bmm(
            candidate_news_vectors, 
            user_vector.unsqueeze(-1)
        ).squeeze(-1)
        
        return scores  # (batch_size, num_candidates)