"""
Quick smoke test for training pipeline.
Validates data loading, model initialization, and a few training steps.
Run this before pushing to Colab to catch errors early!
"""

import torch
from torch.utils.data import DataLoader, Subset
from train import MathTrainer
from data_preprocessing import MathWritingDataset
import sys
import os

def ensure_hf_authentication():
    """Ensure Hugging Face authentication is set up for gated models.
    Compatible with both Colab and local environments."""
    from huggingface_hub import login, whoami
    
    # Check if already authenticated
    try:
        user_info = whoami()
        if user_info:
            print(f"✅ Already authenticated as: {user_info.get('name', 'user')}")
            return True
    except Exception:
        pass
    
    # Try Colab secrets first (for Colab compatibility)
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Authenticated using HF_TOKEN from Colab secrets!")
            return True
    except ImportError:
        pass  # Not running in Colab
    except Exception as e:
        print(f"⚠️  Could not authenticate with Colab secrets: {e}")
    
    # Try environment variable
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Authenticated using HF_TOKEN environment variable")
            return True
        except Exception as e:
            print(f"⚠️  Failed to authenticate with HF_TOKEN: {e}")
    
    # Try Colab notebook_login as fallback (only if in Colab)
    try:
        import google.colab
        # If we're in Colab but no secret, try notebook_login
        print("\n📝 Please login manually:")
        from huggingface_hub import notebook_login
        notebook_login()
        print("✅ Manual authentication successful!")
        return True
    except ImportError:
        pass  # Not in Colab
    except Exception as e:
        print(f"⚠️  Manual authentication failed: {e}")
    
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
    print("\nThen authenticate using one of these methods:")
    print("\n   📱 In Google Colab:")
    print("   - Go to 🔑 (left sidebar) → Add new secret")
    print("   - Name: HF_TOKEN → Paste your token")
    print("   - Or run: notebook_login() in a cell")
    print("\n   💻 Local environment:")
    print("   Option A: Set environment variable:")
    print("   export HF_TOKEN='your_token_here'")
    print("\n   Option B: Login via CLI:")
    print("   huggingface-cli login")
    print("="*70)
    
    return False

def test_data_loading(data_dir, num_samples=10):
    """Test that data loads correctly."""
    print("\n" + "="*70)
    print("TEST 1: Data Loading")
    print("="*70)
    
    try:
        train_ds = MathWritingDataset(data_dir, split='train')
        print(f"✓ Train dataset loaded: {len(train_ds)} total samples")
        
        # Test loading a few samples
        for i in range(min(3, len(train_ds))):
            sample = train_ds[i]
            print(f"\n  Sample {i}:")
            print(f"    Label: {sample['label']}")
            print(f"    Stroke text length: {len(sample['stroke_text'])} chars")
            print(f"    Image size: {sample['image'].size}")
            assert sample['image'].size == (224, 224), "Image size mismatch!"
            assert len(sample['stroke_text']) > 0, "Empty stroke text!"
            assert len(sample['label']) > 0, "Empty label!"
        
        print("\n✓ Data loading PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Data loading FAILED: {e}")
        return False

def test_model_initialization(data_dir="mathwriting-2024-excerpt", device="cuda"):
    """Test model and processor initialization."""
    print("\n" + "="*70)
    print("TEST 2: Model Initialization")
    print("="*70)
    
    try:
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("  CUDA not available, falling back to CPU")
            device = "cpu"
        
        print(f"  Using device: {device}")
        
        # Ensure authentication before initializing trainer
        print("\n  Checking Hugging Face authentication...")
        if not ensure_hf_authentication():
            raise RuntimeError(
                "Authentication required. Please authenticate with Hugging Face "
                "before running tests. See instructions above."
            )
        
        # Initialize trainer (this loads model)
        print("\n  Initializing trainer...")
        trainer = MathTrainer(
            data_dir=data_dir,
            device=device
        )
        
        print("✓ Model and processor initialized")
        print("✓ LoRA configuration applied")
        print("\n✓ Model initialization PASSED")
        return trainer, device
    except RuntimeError as e:
        # Re-raise authentication errors with clear message
        print(f"\n✗ Model initialization FAILED: {e}")
        print("\n💡 TIP: Authentication is required for gated models.")
        print("   If running in Colab, add HF_TOKEN to Colab secrets (🔑 icon).")
        import traceback
        traceback.print_exc()
        return None, None
    except Exception as e:
        print(f"\n✗ Model initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_training_steps(trainer, device, num_steps=3):
    """Test a few training steps."""
    print("\n" + "="*70)
    print("TEST 3: Training Steps")
    print("="*70)
    
    try:
        # Create small subset for testing
        train_subset = Subset(trainer.train_ds, range(min(10, len(trainer.train_ds))))
        test_loader = DataLoader(
            train_subset,
            batch_size=2,
            shuffle=False,
            collate_fn=trainer.collate_fn
        )
        
        print(f"  Testing with {len(train_subset)} samples, batch_size=2")
        
        trainer.model.train()
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
        
        for step, batch in enumerate(test_loader):
            if step >= num_steps:
                break
            
            print(f"\n  Step {step+1}/{num_steps}")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = trainer.model(**batch)
            loss = outputs.loss
            
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Loss is finite: {torch.isfinite(loss).item()}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"    ✓ Backward pass successful")
        
        print("\n✓ Training steps PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Training steps FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation(trainer, device):
    """Test validation/inference."""
    print("\n" + "="*70)
    print("TEST 4: Validation/Inference")
    print("="*70)
    
    try:
        # Create small validation subset
        valid_subset = Subset(trainer.valid_ds, range(min(5, len(trainer.valid_ds))))
        test_loader = DataLoader(
            valid_subset,
            batch_size=2,
            shuffle=False,
            collate_fn=trainer.collate_fn
        )
        
        print(f"  Testing with {len(valid_subset)} validation samples")
        
        trainer.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                # Generate predictions
                generated = trainer.model.generate(**inputs, max_length=64)
                
                # Decode one example
                pred_text = trainer.processor.decode(generated[0], skip_special_tokens=True)
                label_text = trainer.processor.decode(labels[0], skip_special_tokens=True)
                
                print(f"\n  Example prediction:")
                print(f"    Ground truth: {label_text}")
                print(f"    Predicted:    {pred_text}")
                
                # Compute CER
                cer = trainer.latex_tokenizer.compute_cer(pred_text, label_text)
                print(f"    CER: {cer:.4f}")
                
                break  # Just test one batch
        
        print("\n✓ Validation/Inference PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Validation/Inference FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(data_dir="mathwriting-2024-excerpt", fast=True):
    """Run all tests.

    Args:
        data_dir: Path to dataset
        fast: If True, only test data loading and skip model initialization (default: True).
              Set to False to run full model tests (~5 minutes).
    """
    print("\n" + "="*70)
    print("🧪 TRAINING PIPELINE SMOKE TEST")
    print("="*70)

    if fast:
        print("\n⚡ FAST MODE: Testing data loading only (no model download)")
        print("This validates dataset compatibility in ~10 seconds.")
        print("Run with --full flag to test model initialization too.")
    else:
        print("\nFULL MODE: Testing complete pipeline including model")
        print("This will download model and test training - takes ~5 minutes.")

    # Check if running in Colab
    try:
        import google.colab
        print("📱 Running in Google Colab")
        if not fast:
            print("💡 Make sure HF_TOKEN is set in Colab secrets (🔑 icon)")
    except ImportError:
        print("💻 Running locally")
        if not fast:
            print("💡 Make sure HF_TOKEN environment variable is set or use: huggingface-cli login")

    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run all tests
    results = {}

    # Test 1: Data loading (always run - fast!)
    results['data_loading'] = test_data_loading(data_dir)
    if not results['data_loading']:
        print("\n❌ CRITICAL: Data loading failed. Fix this before continuing.")
        sys.exit(1)

    if fast:
        # Fast mode: skip expensive model tests
        print("\n" + "="*70)
        print("✅ FAST TEST PASSED!")
        print("="*70)
        print("\nData loading works correctly. To test model initialization:")
        print("  python test_pipeline.py --full")
        print("\nOr just push to Colab - data compatibility is verified!")
        print("="*70)
        sys.exit(0)

    # Full mode: test everything
    # Test 2: Model initialization
    trainer, device = test_model_initialization(data_dir, device)
    results['model_init'] = trainer is not None
    if not results['model_init']:
        print("\n❌ CRITICAL: Model initialization failed. Fix this before continuing.")
        sys.exit(1)

    # Test 3: Training steps
    results['training'] = test_training_steps(trainer, device)

    # Test 4: Validation
    results['validation'] = test_validation(trainer, device)

    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests PASSED! Safe to push to Colab.")
        print("="*70)
        sys.exit(0)
    else:
        print("\n⚠️  Some tests FAILED. Fix errors before pushing to Colab.")
        print("="*70)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument("--full", action="store_true",
                       help="Run full tests including model initialization (slow)")
    parser.add_argument("--data-dir", type=str, default="mathwriting-2024-excerpt",
                       help="Path to dataset directory")
    args = parser.parse_args()

    main(data_dir=args.data_dir, fast=not args.full)