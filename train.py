from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.dataset_utils import imdbDataSet
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import time
import argparse
import os

def load_and_split_data(file_path, test_size=0.2):
    import pandas as pd
    df = pd.read_csv(file_path)
    return train_test_split(
        df, 
        test_size=test_size, 
        shuffle=True, 
        random_state=42, 
        stratify=df['sentiment']
    )

def initialize_model(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model

def create_data_loader(dataset, tokenizer, batch_size):
    return DataLoader(
        imdbDataSet(df=dataset, tokenizer=tokenizer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True
    )

def evaluate_model(model, loader, full_df, device):
    num_batches = 0 
    probs = []
    with torch.no_grad():
        for batch in loader:
            start_time = time.time()
            
            input_id = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_id, attention_mask=attention_mask)
            batch_probs = F.softmax(outputs.logits, dim=1)
            
            probs.extend(batch_probs.cpu().tolist())
            
            num_batches += 1
            elapsed = time.time() - start_time
            if num_batches % 50 == 0:
                print(f"Batch Number: {num_batches}, Time Elapsed: {elapsed:.4f} seconds")
                
    probs = np.array(probs)
    predicted_labels = np.where(probs[:, 0] > 0.5, 'positive', 'negative')
    accuracy = (predicted_labels == full_df.iloc[:len(probs)]['sentiment'].values).mean()
    print(f"Accuracy: {accuracy:.4f}")

def configure_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.SEQ_CLS,
        inference_mode=False
    )
    return get_peft_model(model, config)

def train_model(model, loader, device, num_epochs, output_dir='./models'):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        total_correct = 0
        batch_count = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['sentiment'].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            batch_count += 1
            
            if batch_count % 100 == 0:
                avg_loss = total_loss / total_samples
                accuracy = total_correct / total_samples
                print(f"Epoch {epoch+1} | Batch {batch_count} | "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

        avg_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {epoch_accuracy * 100:.2f}%\n")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

class IMDBFineTuner:
    def __init__(self, batch_size=16, num_epochs=5, output_dir='./model'):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model_id = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
        
    def run(self):
        train_df, test_df = load_and_split_data('data/IMDB_Dataset.csv')
        

        tokenizer, model = initialize_model(self.model_id, self.device)
        

        loader = create_data_loader(train_df, tokenizer, self.batch_size)

        
        model = configure_lora(model).to(self.device)
        
        train_model(model, loader, self.device, self.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT on IMDB dataset with LoRA')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print(f"Starting training with batch_size={args.batch_size}, epochs={args.num_epochs}")
    print(f"Using device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    
    fine_tuner = IMDBFineTuner(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    fine_tuner.run()