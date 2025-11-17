import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from collections import defaultdict
import argparse
import os

class PredictorWithKG:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        if 'val_auc' in checkpoint:
            print(f"  Validation AUC: {checkpoint['val_auc']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch'] + 1}")
    
    def predict_test_set(self, test_loader):
        """
        Predict scores for test set
        Returns: Dictionary mapping impression_id to list of scores
        """
        predictions = defaultdict(list)
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Predicting')
            for batch in pbar:
                # Move to device
                history_input_ids = batch['history_input_ids'].to(self.device)
                history_masks = batch['history_masks'].to(self.device)
                history_entity_embeddings = batch['history_entity_embeddings'].to(self.device)
                clicked_mask = batch['clicked_mask'].to(self.device)
                candidate_input_ids = batch['candidate_input_ids'].to(self.device)
                candidate_mask = batch['candidate_mask'].to(self.device)
                candidate_entity_embeddings = batch['candidate_entity_embeddings'].to(self.device)
                impression_ids = batch['impression_ids']
                
                # Forward pass
                scores = self.model(
                    history_input_ids, history_masks,
                    candidate_input_ids, candidate_mask,
                    clicked_mask,
                    history_entity_embeddings,
                    candidate_entity_embeddings
                )
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(scores).cpu().numpy().flatten()
                
                # Store predictions
                for imp_id, prob in zip(impression_ids, probs):
                    predictions[imp_id].append(float(prob))
        
        return predictions
    
    def generate_submission(self, predictions, test_behaviors_file, output_file):
        """
        Generate submission file in the required format
        
        Format: impression_id [ranked_news_ids]
        Example: 123 [N1 N2 N3 N4 N5]
        """
        # Parse test behaviors to get impression structure
        impression_news = {}
        
        with open(test_behaviors_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                impression_id = parts[0]
                impressions = parts[4].split() if len(parts) > 4 else []
                
                # Get news IDs (without labels in test set)
                news_ids = [imp for imp in impressions]
                impression_news[impression_id] = news_ids
        
        # Generate submission
        print(f"\nGenerating submission file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for imp_id in sorted(impression_news.keys(), key=lambda x: int(x)):
                if imp_id in predictions:
                    news_ids = impression_news[imp_id]
                    scores = predictions[imp_id][:len(news_ids)]
                    
                    # Rank news by scores (descending)
                    ranked_news = [news_id for _, news_id in 
                                   sorted(zip(scores, news_ids), reverse=True)]
                    
                    # Write in required format
                    f.write(f"{imp_id} [{' '.join(ranked_news)}]\n")
        
        print(f"✓ Submission file saved!")
        print(f"  Total impressions: {len(impression_news)}")

def main():
    parser = argparse.ArgumentParser(description='Generate predictions for MIND test set')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test data directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='prediction.txt',
                        help='Output submission file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--max_title_len', type=int, default=30,
                        help='Maximum title length')
    parser.add_argument('--max_history_len', type=int, default=50,
                        help='Maximum history length')
    parser.add_argument('--max_entities', type=int, default=5,
                        help='Maximum entities per news')
    parser.add_argument('--news_embed_dim', type=int, default=400,
                        help='News embedding dimension')
    parser.add_argument('--num_attention_heads', type=int, default=16,
                        help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load test dataset
    from preprocessing import MINDDatasetWithKG, collate_fn_kg
    
    print("\n" + "="*60)
    print("Loading Test Data")
    print("="*60)
    test_dataset = MINDDatasetWithKG(
        behaviors_file=f'{args.test_dir}/behaviors.tsv',
        news_file=f'{args.test_dir}/news.tsv',
        entity_embedding_path=f'{args.test_dir}/entity_embedding.vec',
        relation_embedding_path=f'{args.test_dir}/relation_embedding.vec',
        tokenizer=tokenizer,
        max_title_len=args.max_title_len,
        max_history_len=args.max_history_len,
        max_entities=args.max_entities,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_kg,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"\nTest samples: {len(test_dataset):,}")
    print(f"Test batches: {len(test_loader):,}")
    
    # Initialize model
    from nrms_bert_kg_model import NRMSBertKG
    
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)
    
    model = NRMSBertKG(
        bert_model_name='bert-base-uncased',
        num_attention_heads=args.num_attention_heads,
        news_embed_dim=args.news_embed_dim,
        entity_embed_dim=100,
        use_entity=True,
        use_relation=False
    )
    
    # Initialize predictor
    predictor = PredictorWithKG(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Load checkpoint
    print("\n" + "="*60)
    print("Loading Model Checkpoint")
    print("="*60)
    predictor.load_checkpoint(args.checkpoint)
    
    # Predict
    print("\n" + "="*60)
    print("Running Inference")
    print("="*60 + "\n")
    predictions = predictor.predict_test_set(test_loader)
    
    # Generate submission
    print("\n" + "="*60)
    print("Generating Submission File")
    print("="*60)
    predictor.generate_submission(
        predictions=predictions,
        test_behaviors_file=f'{args.test_dir}/behaviors.tsv',
        output_file=args.output
    )
    
    print(f"\n{'='*60}")
    print(f"Inference Completed!")
    print(f"{'='*60}")
    print(f"Submission file: {args.output}")
    print(f"Ready to submit to the competition!")

if __name__ == '__main__':
    main()