import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import numpy as np

class KnowledgeGraphEmbedding:
    """Load and manage entity/relation embeddings from WikiData KG"""
    def __init__(self, entity_embedding_path, relation_embedding_path):
        self.entity_embeddings = self._load_embeddings(entity_embedding_path)
        self.relation_embeddings = self._load_embeddings(relation_embedding_path)
        self.entity_dim = 100
        self.relation_dim = 100
        
    def _load_embeddings(self, file_path):
        """Load embeddings from .vec file"""
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    entity_id = parts[0]
                    embedding = np.array([float(x) for x in parts[1:]])
                    embeddings[entity_id] = embedding
        return embeddings
    
    def get_entity_embedding(self, entity_id):
        """Get embedding for an entity"""
        return self.entity_embeddings.get(entity_id, None)
    
    def get_relation_embedding(self, relation_id):
        """Get embedding for a relation"""
        return self.relation_embeddings.get(relation_id, None)

class NewsEncoderWithKG(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased',
                 num_attention_heads=16,
                 news_embed_dim=400,
                 entity_embed_dim=100,
                 use_entity=True,
                 use_relation=True):
        super().__init__()
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size  # 768
        
        self.use_entity = use_entity
        self.use_relation = use_relation
        
        # Project BERT output
        self.title_projection = nn.Linear(bert_dim, news_embed_dim)
        
        # Entity embedding layer
        if use_entity:
            self.entity_projection = nn.Linear(entity_embed_dim, news_embed_dim)
        
        # Multi-head self-attention for title words
        self.title_attention = nn.MultiheadAttention(
            news_embed_dim, num_attention_heads, batch_first=True
        )
        
        # Multi-head self-attention for entities (if used)
        if use_entity:
            self.entity_attention = nn.MultiheadAttention(
                news_embed_dim, num_attention_heads, batch_first=True
            )
        
        # Additive attention for word-level aggregation
        self.title_additive_attention = nn.Sequential(
            nn.Linear(news_embed_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        
        # Additive attention for entity-level aggregation
        if use_entity:
            self.entity_additive_attention = nn.Sequential(
                nn.Linear(news_embed_dim, 200),
                nn.Tanh(),
                nn.Linear(200, 1)
            )
        
        # Final aggregation layer
        if use_entity:
            self.final_attention = nn.Sequential(
                nn.Linear(news_embed_dim, 200),
                nn.Tanh(),
                nn.Linear(200, 1)
            )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input_ids, attention_mask, entity_embeddings=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            entity_embeddings: (batch_size, num_entities, entity_dim) - optional
        """
        # Encode title with BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        word_embeddings = bert_output.last_hidden_state  # (B, seq_len, 768)
        
        # Project to news_embed_dim
        word_embeddings = self.title_projection(word_embeddings)
        word_embeddings = self.dropout(word_embeddings)
        
        # Multi-head self-attention on title
        word_embeddings, _ = self.title_attention(
            word_embeddings, word_embeddings, word_embeddings,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Additive attention for title
        title_attn_weights = self.title_additive_attention(word_embeddings)
        title_attn_weights = F.softmax(title_attn_weights, dim=1)
        title_vector = torch.sum(word_embeddings * title_attn_weights, dim=1)
        
        # If entity embeddings are provided, incorporate them
        if self.use_entity and entity_embeddings is not None:
            # Project entity embeddings
            entity_vecs = self.entity_projection(entity_embeddings)
            entity_vecs = self.dropout(entity_vecs)
            
            # Multi-head self-attention on entities
            entity_vecs, _ = self.entity_attention(
                entity_vecs, entity_vecs, entity_vecs
            )
            
            # Additive attention for entities
            entity_attn_weights = self.entity_additive_attention(entity_vecs)
            entity_attn_weights = F.softmax(entity_attn_weights, dim=1)
            entity_vector = torch.sum(entity_vecs * entity_attn_weights, dim=1)
            
            # Combine title and entity representations
            combined = torch.stack([title_vector, entity_vector], dim=1)
            final_attn_weights = self.final_attention(combined)
            final_attn_weights = F.softmax(final_attn_weights, dim=1)
            news_vector = torch.sum(combined * final_attn_weights, dim=1)
        else:
            news_vector = title_vector
        
        return news_vector

class UserEncoder(nn.Module):
    def __init__(self, news_embed_dim=400, num_attention_heads=16):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            news_embed_dim, num_attention_heads, batch_first=True
        )
        
        self.additive_attention = nn.Sequential(
            nn.Linear(news_embed_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        
    def forward(self, news_vectors, mask=None):
        # Multi-head self-attention
        user_vectors, _ = self.multi_head_attention(
            news_vectors, news_vectors, news_vectors,
            key_padding_mask=mask
        )
        
        # Additive attention
        attention_weights = self.additive_attention(user_vectors)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                mask.unsqueeze(-1), float('-inf')
            )
        
        attention_weights = F.softmax(attention_weights, dim=1)
        user_vector = torch.sum(user_vectors * attention_weights, dim=1)
        
        return user_vector

class NRMSBertKG(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased',
                 num_attention_heads=16,
                 news_embed_dim=400,
                 entity_embed_dim=100,
                 use_entity=True,
                 use_relation=False):
        super().__init__()
        self.news_encoder = NewsEncoderWithKG(
            bert_model_name,
            num_attention_heads,
            news_embed_dim,
            entity_embed_dim,
            use_entity,
            use_relation
        )
        self.user_encoder = UserEncoder(
            news_embed_dim,
            num_attention_heads
        )
        self.use_entity = use_entity
        
    def forward(self, clicked_news_input_ids, clicked_news_masks,
                candidate_news_input_ids, candidate_news_masks,
                clicked_mask=None,
                clicked_entity_embeddings=None,
                candidate_entity_embeddings=None):
        """
        Args:
            clicked_news_input_ids: (batch_size, num_clicked, seq_len)
            clicked_news_masks: (batch_size, num_clicked, seq_len)
            candidate_news_input_ids: (batch_size, num_candidates, seq_len)
            candidate_news_masks: (batch_size, num_candidates, seq_len)
            clicked_mask: (batch_size, num_clicked)
            clicked_entity_embeddings: (batch_size, num_clicked, num_entities, entity_dim)
            candidate_entity_embeddings: (batch_size, num_candidates, num_entities, entity_dim)
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
        
        if self.use_entity and clicked_entity_embeddings is not None:
            clicked_entity_embeddings = clicked_entity_embeddings.view(
                batch_size * num_clicked, -1, clicked_entity_embeddings.size(-1)
            )
        else:
            clicked_entity_embeddings = None
        
        clicked_news_vectors = self.news_encoder(
            clicked_news_input_ids,
            clicked_news_masks,
            clicked_entity_embeddings
        )
        clicked_news_vectors = clicked_news_vectors.view(
            batch_size, num_clicked, -1
        )
        
        # Encode user
        user_vector = self.user_encoder(clicked_news_vectors, mask=clicked_mask)
        
        # Encode candidate news
        candidate_news_input_ids = candidate_news_input_ids.view(
            batch_size * num_candidates, -1
        )
        candidate_news_masks = candidate_news_masks.view(
            batch_size * num_candidates, -1
        )
        
        if self.use_entity and candidate_entity_embeddings is not None:
            candidate_entity_embeddings = candidate_entity_embeddings.view(
                batch_size * num_candidates, -1, candidate_entity_embeddings.size(-1)
            )
        else:
            candidate_entity_embeddings = None
        
        candidate_news_vectors = self.news_encoder(
            candidate_news_input_ids,
            candidate_news_masks,
            candidate_entity_embeddings
        )
        candidate_news_vectors = candidate_news_vectors.view(
            batch_size, num_candidates, -1
        )
        
        # Click prediction
        scores = torch.bmm(
            candidate_news_vectors,
            user_vector.unsqueeze(-1)
        ).squeeze(-1)
        
        return scores