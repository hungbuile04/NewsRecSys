import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")

class Trainer:
    def __init__(self, model, train_loader, val_loader, 
                 device='cuda', lr=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer with different learning rates for BERT and other layers
        bert_params = list(model.news_encoder.bert.parameters())
        other_params = [p for n, p in model.named_parameters() 
                       if 'bert' not in n]
        
        self.optimizer = optim.Adam([
            {'params': bert_params, 'lr': lr * 0.1},  # Lower LR for BERT
            {'params': other_params, 'lr': lr}
        ], weight_decay=weight_decay)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        self.best_auc = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            history_input_ids = batch['history_input_ids'].to(self.device)
            history_masks = batch['history_masks'].to(self.device)
            clicked_mask = batch['clicked_mask'].to(self.device)
            candidate_input_ids = batch['candidate_input_ids'].to(self.device)
            candidate_mask = batch['candidate_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            scores = self.model(
                history_input_ids, history_masks,
                candidate_input_ids, candidate_mask,
                clicked_mask
            )
            
            # Calculate loss
            loss = self.criterion(scores.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                # Move to device
                history_input_ids = batch['history_input_ids'].to(self.device)
                history_masks = batch['history_masks'].to(self.device)
                clicked_mask = batch['clicked_mask'].to(self.device)
                candidate_input_ids = batch['candidate_input_ids'].to(self.device)
                candidate_mask = batch['candidate_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                scores = self.model(
                    history_input_ids, history_masks,
                    candidate_input_ids, candidate_mask,
                    clicked_mask
                )
                
                # Calculate loss
                loss = self.criterion(scores.squeeze(), labels)
                total_loss += loss.item()
                
                # Record predictions
                all_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def train(self, num_epochs, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_auc = self.train_epoch()
            print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
            
            # Validate
            val_loss, val_auc = self.validate()
            print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # Learning rate scheduling
            self.scheduler.step(val_auc)
            
            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f'Saved best model with AUC: {val_auc:.4f}')
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': val_loss
                }, checkpoint_path)

# Main training script
if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 64
    MAX_TITLE_LEN = 30
    MAX_HISTORY_LEN = 50
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load datasets
    from preprocessing import MINDDataset, collate_fn
    
    train_dataset = MINDDataset(
        behaviors_file=os.path.join(DATA_DIR, 'MINDsmall_train', 'behaviors.tsv'),
        news_file=os.path.join(DATA_DIR, 'MINDsmall_train', 'news.tsv'),
        tokenizer=tokenizer,
        max_title_len=MAX_TITLE_LEN,
        max_history_len=MAX_HISTORY_LEN,
        mode='train'
    )

    val_dataset = MINDDataset(
        behaviors_file=os.path.join(DATA_DIR, 'MINDsmall_dev', 'behaviors.tsv'),
        news_file=os.path.join(DATA_DIR, 'MINDsmall_dev', 'news.tsv'),
        tokenizer=tokenizer,
        max_title_len=MAX_TITLE_LEN,
        max_history_len=MAX_HISTORY_LEN,
        mode='dev'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    from nrms_bert_model import NRMSBert
    
    model = NRMSBert(
        bert_model_name='bert-base-uncased',
        num_attention_heads=16,
        news_embed_dim=400
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr=LEARNING_RATE
    )
    
    # Train
    trainer.train(num_epochs=NUM_EPOCHS)
    
    print(f"\nTraining completed! Best validation AUC: {trainer.best_auc:.4f}")