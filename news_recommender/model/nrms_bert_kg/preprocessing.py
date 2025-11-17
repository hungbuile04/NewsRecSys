import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import json
from typing import Dict, List

class MINDDatasetWithKG(Dataset):
    """MIND Dataset with Knowledge Graph Embeddings"""
    def __init__(self, behaviors_file, news_file, 
                 entity_embedding_path, relation_embedding_path,
                 tokenizer, max_title_len=30, max_history_len=50, 
                 max_entities=5, mode='train'):
        """
        Args:
            behaviors_file: path to behaviors.tsv
            news_file: path to news.tsv
            entity_embedding_path: path to entity_embedding.vec
            relation_embedding_path: path to relation_embedding.vec
            tokenizer: BertTokenizer
            max_title_len: maximum title length
            max_history_len: maximum number of clicked news
            max_entities: maximum number of entities per news
            mode: 'train', 'dev', or 'test'
        """
        self.tokenizer = tokenizer
        self.max_title_len = max_title_len
        self.max_history_len = max_history_len
        self.max_entities = max_entities
        self.mode = mode
        
        # Load entity and relation embeddings
        print("Loading entity embeddings...")
        self.entity_embeddings = self._load_embeddings(entity_embedding_path)
        print(f"Loaded {len(self.entity_embeddings)} entity embeddings")
        
        print("Loading relation embeddings...")
        self.relation_embeddings = self._load_embeddings(relation_embedding_path)
        print(f"Loaded {len(self.relation_embeddings)} relation embeddings")
        
        self.entity_dim = 100  # TransE embedding dimension
        
        # Load news data
        print("Loading news data...")
        self.news_dict = self._load_news(news_file)
        print(f"Loaded {len(self.news_dict)} news articles")
        
        # Load behaviors data
        print("Loading behaviors data...")
        self.behaviors = self._load_behaviors(behaviors_file)
        print(f"Loaded {len(self.behaviors)} behavior samples")
        
    def _load_embeddings(self, file_path):
        """Load embeddings from .vec file"""
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    entity_id = parts[0]
                    embedding = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    embeddings[entity_id] = embedding
        return embeddings
    
    def _parse_entities(self, entity_str):
        """Parse entity string from news.tsv"""
        if not entity_str or entity_str == '[]':
            return []
        
        try:
            entities = json.loads(entity_str)
            entity_ids = []
            for entity in entities:
                if 'WikidataId' in entity:
                    entity_ids.append(entity['WikidataId'])
            return entity_ids
        except:
            return []
    
    def _get_entity_embeddings(self, entity_ids):
        """Get embeddings for a list of entity IDs"""
        embeddings = []
        for entity_id in entity_ids[:self.max_entities]:
            if entity_id in self.entity_embeddings:
                embeddings.append(self.entity_embeddings[entity_id])
            else:
                # Use zero vector for unknown entities
                embeddings.append(np.zeros(self.entity_dim, dtype=np.float32))
        
        # Pad to max_entities
        while len(embeddings) < self.max_entities:
            embeddings.append(np.zeros(self.entity_dim, dtype=np.float32))
        
        return np.array(embeddings[:self.max_entities], dtype=np.float32)
    
    def _load_news(self, news_file):
        """Load and preprocess news data"""
        news_dict = {}
        
        with open(news_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                news_id = parts[0]
                category = parts[1]
                subcategory = parts[2]
                title = parts[3]
                abstract = parts[4] if len(parts) > 4 else ""
                title_entities = parts[6] if len(parts) > 6 else "[]"
                abstract_entities = parts[7] if len(parts) > 7 else "[]"
                
                # Tokenize title
                encoded = self.tokenizer.encode_plus(
                    title,
                    max_length=self.max_title_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Parse entities
                entity_ids = self._parse_entities(title_entities)
                entity_embeddings = self._get_entity_embeddings(entity_ids)
                
                news_dict[news_id] = {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'entity_embeddings': torch.FloatTensor(entity_embeddings),
                    'category': category,
                    'subcategory': subcategory
                }
        
        # Add padding news
        padding_encoded = self.tokenizer.encode_plus(
            "",
            max_length=self.max_title_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        padding_entities = np.zeros((self.max_entities, self.entity_dim), dtype=np.float32)
        
        news_dict['PADDING'] = {
            'input_ids': padding_encoded['input_ids'].squeeze(0),
            'attention_mask': padding_encoded['attention_mask'].squeeze(0),
            'entity_embeddings': torch.FloatTensor(padding_entities),
            'category': '',
            'subcategory': ''
        }
        
        return news_dict
    
    def _load_behaviors(self, behaviors_file):
        """Load behaviors data"""
        behaviors = []
        
        with open(behaviors_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                impression_id = parts[0]
                user_id = parts[1]
                time = parts[2]
                history = parts[3].split() if parts[3] else []
                impressions = parts[4].split() if len(parts) > 4 else []
                
                if self.mode == 'test':
                    for imp in impressions:
                        news_id = imp
                        behaviors.append({
                            'impression_id': impression_id,
                            'user_id': user_id,
                            'history': history,
                            'candidate': news_id,
                            'label': None
                        })
                else:
                    for imp in impressions:
                        news_id, label = imp.split('-')
                        behaviors.append({
                            'impression_id': impression_id,
                            'user_id': user_id,
                            'history': history,
                            'candidate': news_id,
                            'label': int(label)
                        })
        
        return behaviors
    
    def __len__(self):
        return len(self.behaviors)
    
    def __getitem__(self, idx):
        behavior = self.behaviors[idx]
        
        # Get history news
        history_news_ids = behavior['history'][-self.max_history_len:]
        
        # Pad history if needed
        num_padding = self.max_history_len - len(history_news_ids)
        if num_padding > 0:
            history_news_ids = ['PADDING'] * num_padding + history_news_ids
        
        # Get history news data
        history_input_ids = []
        history_masks = []
        history_entity_embeddings = []
        clicked_mask = []
        
        for news_id in history_news_ids:
            if news_id in self.news_dict:
                history_input_ids.append(self.news_dict[news_id]['input_ids'])
                history_masks.append(self.news_dict[news_id]['attention_mask'])
                history_entity_embeddings.append(self.news_dict[news_id]['entity_embeddings'])
                clicked_mask.append(news_id == 'PADDING')
            else:
                history_input_ids.append(self.news_dict['PADDING']['input_ids'])
                history_masks.append(self.news_dict['PADDING']['attention_mask'])
                history_entity_embeddings.append(self.news_dict['PADDING']['entity_embeddings'])
                clicked_mask.append(True)
        
        history_input_ids = torch.stack(history_input_ids)
        history_masks = torch.stack(history_masks)
        history_entity_embeddings = torch.stack(history_entity_embeddings)
        clicked_mask = torch.tensor(clicked_mask, dtype=torch.bool)
        
        # Get candidate news
        candidate_id = behavior['candidate']
        if candidate_id in self.news_dict:
            candidate_input_ids = self.news_dict[candidate_id]['input_ids']
            candidate_mask = self.news_dict[candidate_id]['attention_mask']
            candidate_entity_embeddings = self.news_dict[candidate_id]['entity_embeddings']
        else:
            candidate_input_ids = self.news_dict['PADDING']['input_ids']
            candidate_mask = self.news_dict['PADDING']['attention_mask']
            candidate_entity_embeddings = self.news_dict['PADDING']['entity_embeddings']
        
        # Get label
        label = behavior['label'] if behavior['label'] is not None else -1
        
        return {
            'history_input_ids': history_input_ids,
            'history_masks': history_masks,
            'history_entity_embeddings': history_entity_embeddings,
            'clicked_mask': clicked_mask,
            'candidate_input_ids': candidate_input_ids.unsqueeze(0),
            'candidate_mask': candidate_mask.unsqueeze(0),
            'candidate_entity_embeddings': candidate_entity_embeddings.unsqueeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'impression_id': behavior['impression_id']
        }

def collate_fn_kg(batch):
    """Custom collate function for batching with KG"""
    return {
        'history_input_ids': torch.stack([x['history_input_ids'] for x in batch]),
        'history_masks': torch.stack([x['history_masks'] for x in batch]),
        'history_entity_embeddings': torch.stack([x['history_entity_embeddings'] for x in batch]),
        'clicked_mask': torch.stack([x['clicked_mask'] for x in batch]),
        'candidate_input_ids': torch.stack([x['candidate_input_ids'] for x in batch]),
        'candidate_mask': torch.stack([x['candidate_mask'] for x in batch]),
        'candidate_entity_embeddings': torch.stack([x['candidate_entity_embeddings'] for x in batch]),
        'labels': torch.stack([x['label'] for x in batch]),
        'impression_ids': [x['impression_id'] for x in batch]
    }

# Example usage
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = MINDDatasetWithKG(
        behaviors_file='MINDlarge_train/behaviors.tsv',
        news_file='MINDlarge_train/news.tsv',
        entity_embedding_path='MINDlarge_train/entity_embedding.vec',
        relation_embedding_path='MINDlarge_train/relation_embedding.vec',
        tokenizer=tokenizer,
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_kg
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Test one batch
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("History entity embeddings shape:", batch['history_entity_embeddings'].shape)
        print("Candidate entity embeddings shape:", batch['candidate_entity_embeddings'].shape)
        break