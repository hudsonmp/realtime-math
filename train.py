"""Training script for PaliGemma with LoRA on handwritten math recognition."""
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model
from data_preprocessing import MathWritingDataset, LaTeXTokenizer
import os
from tqdm import tqdm

class MathTrainer:
    def __init__(self, data_dir, model_name="google/paligemma-3b-pt-224", 
                 output_dir="./checkpoints", device="cuda"):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading model and processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("Loading datasets...")
        self.train_ds = MathWritingDataset(data_dir, split='train')
        self.valid_ds = MathWritingDataset(data_dir, split='valid')
        
        self.latex_tokenizer = LaTeXTokenizer()
        
    def collate_fn(self, batch):
        """Custom collate for variable-length text sequences."""
        stroke_texts = [item['stroke_text'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Tokenize inputs (stroke text)
        inputs = self.processor(
            text=stroke_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # Tokenize labels
        label_encodings = self.processor.tokenizer(
            labels,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        
        inputs['labels'] = label_encodings['input_ids']
        return inputs
    
    def train(self, epochs=10, batch_size=4, lr=1e-4, grad_accum=4):
        """Training loop."""
        train_loader = DataLoader(
            self.train_ds, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4
        )
        
        valid_loader = DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        num_training_steps = epochs * len(train_loader) // grad_accum
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps
        )
        
        print(f"\nTraining for {epochs} epochs...")
        print(f"Steps per epoch: {len(train_loader)}")
        print(f"Total steps: {num_training_steps}")
        
        self.model.train()
        global_step = 0
        best_cer = float('inf')
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print('='*60)
            
            epoch_loss = 0
            optimizer.zero_grad()
            
            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss / grad_accum
                loss.backward()
                
                if (step + 1) % grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += loss.item() * grad_accum
                
                if step % 100 == 0:
                    print(f"Step {step}, Loss: {loss.item() * grad_accum:.4f}")
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            # Validation every epoch
            val_loss, val_cer = self.validate(valid_loader)
            print(f"Validation Loss: {val_loss:.4f}, CER: {val_cer:.4f}")
            
            # Track and save best model based on CER
            if val_cer < best_cer:
                print(f"🎉 New best CER! {best_cer:.4f} -> {val_cer:.4f}")
                best_cer = val_cer
                self.save_checkpoint(epoch + 1, metric_name='cer', metric_value=val_cer)
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\nTraining complete!")
        self.save_checkpoint("final")
    
    def validate(self, valid_loader):
        """Validation loop with CER calculation."""
        self.model.eval()
        total_loss = 0
        total_cer = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                # Calculate loss
                outputs = self.model(**{**inputs, 'labels': labels})
                total_loss += outputs.loss.item()
                
                # Generate predictions for CER
                generated = self.model.generate(**inputs, max_length=64)
                
                # Decode and compute CER
                for pred_ids, label_ids in zip(generated, labels):
                    pred_text = self.processor.decode(pred_ids, skip_special_tokens=True)
                    label_text = self.processor.decode(label_ids, skip_special_tokens=True)
                    cer = self.latex_tokenizer.compute_cer(pred_text, label_text)
                    total_cer += cer
                    num_samples += 1
        
        self.model.train()
        avg_loss = total_loss / len(valid_loader)
        avg_cer = total_cer / num_samples if num_samples > 0 else 0
        
        print(f"  Validation CER: {avg_cer:.4f}")
        return avg_loss, avg_cer
    
    def save_checkpoint(self, epoch, metric_name=None, metric_value=None):
        """Save LoRA weights and optionally full checkpoint with metrics."""
        # Save LoRA adapters
        save_path = os.path.join(self.output_dir, f"lora_epoch_{epoch}")
        self.model.save_pretrained(save_path)
        
        # If this is a best checkpoint, also save full state dict with metrics
        if metric_name is not None and metric_value is not None:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                metric_name: metric_value,
            }
            checkpoint_path = os.path.join(
                self.output_dir, 
                f"best_epoch{epoch}_{metric_name}{metric_value:.4f}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Best checkpoint saved: {checkpoint_path}")
        else:
            print(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    trainer = MathTrainer(
        data_dir="mathwriting-2024-excerpt",
        output_dir="./math_lora_checkpoints"
    )
    
    trainer.train(
        epochs=10,
        batch_size=4,
        lr=1e-4,
        grad_accum=4
    )

