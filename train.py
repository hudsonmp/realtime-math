"""Training script for PaliGemma with LoRA on handwritten math recognition."""
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model
from data_preprocessing import MathWritingDataset, LaTeXTokenizer
import os
import shutil
from tqdm import tqdm

def ensure_hf_authentication():
    """Ensure Hugging Face authentication is set up for gated models."""
    from huggingface_hub import login, whoami
    import os
    
    # Check if already authenticated
    try:
        user_info = whoami()
        if user_info:
            print(f"✅ Already authenticated as: {user_info.get('name', 'user')}")
            return True
    except Exception:
        pass
    
    # Try to authenticate from environment variable or token file
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Authenticated using HF_TOKEN environment variable")
            return True
        except Exception as e:
            print(f"⚠️  Failed to authenticate with HF_TOKEN: {e}")
    
    # Try Colab secrets (if running in Colab)
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Authenticated using HF_TOKEN from Colab secrets")
            return True
    except ImportError:
        pass  # Not in Colab
    except Exception as e:
        print(f"⚠️  Could not authenticate with Colab secrets: {e}")
    
    # Final attempt: check if CLI login exists
    try:
        user_info = whoami()
        if user_info:
            print(f"✅ Using existing Hugging Face CLI login: {user_info.get('name', 'user')}")
            return True
    except Exception:
        pass
    
    # Authentication failed - provide helpful error message
    print("\n" + "="*70)
    print("❌ Hugging Face Authentication Required")
    print("="*70)
    print("\nThe model 'google/paligemma-3b-pt-224' is a gated repository.")
    print("You need to:")
    print("\n1. Request access at: https://huggingface.co/google/paligemma-3b-pt-224")
    print("2. Create a token at: https://huggingface.co/settings/tokens")
    print("3. Authenticate using one of these methods:")
    print("\n   Option A: Set environment variable:")
    print("   export HF_TOKEN='your_token_here'")
    print("\n   Option B: Login via CLI:")
    print("   huggingface-cli login")
    print("\n   Option C: Login programmatically:")
    print("   from huggingface_hub import login")
    print("   login(token='your_token_here')")
    print("="*70)
    
    raise RuntimeError(
        "Hugging Face authentication required. "
        "Please request access to the gated model and authenticate. "
        "See error message above for instructions."
    )

class MathTrainer:
    def __init__(self, data_dir, model_name="google/paligemma-3b-pt-224",
                 output_dir="./checkpoints", device="cuda", use_gradient_checkpointing=False):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Ensure authentication before loading model
        ensure_hf_authentication()

        print("\nLoading model and processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load model WITHOUT device_map to enable gradient checkpointing
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.model.to(device)

        # Enable gradient checkpointing if requested (saves memory)
        if use_gradient_checkpointing:
            print("Enabling gradient checkpointing (memory-efficient mode)...")
            self.model.gradient_checkpointing_enable()

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
        """Custom collate for variable-length text sequences and images."""
        stroke_texts = [item['stroke_text'] for item in batch]
        images = [item['image'] for item in batch]  # PIL Images from renderer
        labels = [item['label'] for item in batch]

        # Tokenize inputs (stroke text + images) - PaliGemma expects both!
        inputs = self.processor(
            text=stroke_texts,
            images=images,
            padding="longest",  # Changed from True to "longest" for efficiency
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )

        # Tokenize labels - need to match input sequence length!
        # PaliGemma uses suffix language modeling: image + text -> label
        label_encodings = self.processor.tokenizer(
            labels,
            padding="max_length",  # Pad to max_length
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        # Create labels tensor that matches input length
        # Set non-label tokens to -100 (ignored in loss)
        import torch
        batch_size = inputs['input_ids'].shape[0]
        seq_length = inputs['input_ids'].shape[1]
        labels_tensor = torch.full((batch_size, seq_length), -100, dtype=torch.long)

        # Copy label tokens to the end of sequence
        for i, label_ids in enumerate(label_encodings['input_ids']):
            # Find actual label length (excluding padding)
            label_length = (label_ids != self.processor.tokenizer.pad_token_id).sum().item()
            # Place labels at the end of the sequence
            labels_tensor[i, -label_length:] = label_ids[:label_length]

        inputs['labels'] = labels_tensor
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
        """Save LoRA weights. For best checkpoints, copy to a dedicated best folder."""
        # Save LoRA adapters (regular checkpoint)
        save_path = os.path.join(self.output_dir, f"lora_epoch_{epoch}")
        self.model.save_pretrained(save_path)
        print(f"Checkpoint saved: {save_path}")
        
        # If this is a best checkpoint, copy the LoRA folder with descriptive name
        if metric_name is not None and metric_value is not None:
            best_path = os.path.join(
                self.output_dir, 
                f"best_lora_epoch_{epoch}_{metric_name}_{metric_value:.4f}"
            )
            
            # Copy the LoRA checkpoint to best location
            if os.path.exists(best_path):
                shutil.rmtree(best_path)  # Remove old best if exists
            shutil.copytree(save_path, best_path)
            
            print(f"✨ Best checkpoint copied to: {best_path}")
            print(f"   (LoRA adapters only, ~27M params instead of full ~3B model)")


if __name__ == "__main__":
    # For Colab: Upload full MathWriting dataset to /content/mathwriting-2024
    # Directory structure should be: train/, valid/, test/, symbols/
    trainer = MathTrainer(
        data_dir="/content/mathwriting-2024",
        output_dir="./math_lora_checkpoints"
    )
    
    trainer.train(
        epochs=10,
        batch_size=4,
        lr=1e-4,
        grad_accum=4
    )

