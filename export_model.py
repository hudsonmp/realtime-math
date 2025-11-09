"""Export and quantize trained model for RUBIK Pi deployment."""
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import os

def ensure_hf_authentication():
    """Ensure Hugging Face authentication is set up for gated models."""
    from huggingface_hub import login, whoami

    # Check if already authenticated
    try:
        user_info = whoami()
        if user_info:
            print(f"✅ Already authenticated as: {user_info.get('name', 'user')}")
            return True
    except Exception:
        pass

    # Try environment variable
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Authenticated using HF_TOKEN")
            return True
        except Exception as e:
            print(f"⚠️  Authentication failed: {e}")

    # Try Colab secrets
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
            print("✅ Authenticated using Colab secrets")
            return True
    except:
        pass

    print("⚠️  No HF authentication found. May fail for gated models.")
    return False

class ModelExporter:
    def _get_device_config(self):
        """
        Auto-detect optimal device and dtype configuration.
        Returns (torch_dtype, device_map) tuple.
        """
        if torch.cuda.is_available():
            print(f"✓ CUDA detected: {torch.cuda.get_device_name(0)}")
            print("  Using bfloat16 with auto device mapping for GPU (A100 optimized)")
            return torch.bfloat16, "auto"
        else:
            print("✓ No CUDA detected: Using CPU")
            print("  Using float32 to avoid slow implicit upcasts on CPU")
            return torch.float32, "cpu"
    
    def __init__(self, base_model="google/paligemma-3b-pt-224",
                 lora_path="./math_lora_checkpoints/lora_epoch_final"):
        # Ensure authentication
        ensure_hf_authentication()

        print("Loading base model...")

        # Get optimal device configuration
        torch_dtype, device_map = self._get_device_config()
        
        # Try loading with optimal config, fallback to CPU if GPU fails
        try:
            self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if device_map != "cpu":
                print(f"⚠ GPU loading failed ({type(e).__name__}), falling back to CPU...")
                self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(
                    base_model,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            else:
                raise
        
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(base_model)
    
    def merge_and_save(self, output_dir="./merged_model"):
        """Merge LoRA weights with base model."""
        print("Merging LoRA weights...")
        merged_model = self.model.merge_and_unload()
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving merged model to {output_dir}...")
        merged_model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        print("✓ Model merged and saved!")
        return merged_model
    
    def quantize_int8(self, model, output_dir="./quantized_model"):
        """
        Quantize model to INT8 for GPU edge deployment.
        
        NOTE: This method uses bitsandbytes 8-bit quantization (GPU-optimized) as primary,
        with torch.quantization.quantize_dynamic (CPU-focused) as fallback.
        
        For maximum GPU performance, consider:
        - TensorRT (NVIDIA GPUs, requires ONNX conversion)
        - Optimum with GPTQ/AWQ (HuggingFace integrated)
        - ONNX Runtime (cross-platform)
        """
        print("Quantizing to INT8...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Try bitsandbytes first (GPU-optimized, HuggingFace compatible)
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            
            print("Using bitsandbytes 8-bit quantization (GPU-optimized)...")
            
            # Configure 8-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
            # Quantize model
            quantized_model = model.to('cuda')
            # Apply quantization to linear layers
            for name, module in quantized_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight = bnb.nn.Int8Params(
                        module.weight.data.to('cuda'),
                        requires_grad=False,
                        has_fp16_weights=False
                    )
            
            print(f"Saving quantized model to {output_dir}...")
            quantized_model.save_pretrained(output_dir)
            self.processor.save_pretrained(output_dir)
            quantization_method = "bitsandbytes"
            
        except ImportError:
            print("⚠ bitsandbytes not available, falling back to torch quantization (CPU-optimized)")
            print("  Install with: pip install bitsandbytes")
            
            # Fallback: torch dynamic quantization (CPU-focused)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            print(f"Saving quantized model to {output_dir}...")
            # Save full model object (not just state_dict) so it's reloadable
            torch.save(quantized_model, os.path.join(output_dir, "model_quantized_full.pt"))
            self.processor.save_pretrained(output_dir)
            quantization_method = "torch_dynamic"
        
        print("✓ Model quantized and saved!")
        
        # Compare on-disk sizes (like-for-like comparison)
        print("\nCalculating on-disk sizes...")
        original_dir = "./merged_model"  # Assumes merge_and_save was called first
        
        def get_directory_size(directory):
            """Calculate total size of all files in directory."""
            total_size = 0
            if not os.path.exists(directory):
                return 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size
        
        original_size_bytes = get_directory_size(original_dir)
        quantized_size_bytes = get_directory_size(output_dir)
        
        original_size_gb = original_size_bytes / (1024**3)
        quantized_size_gb = quantized_size_bytes / (1024**3)
        
        print(f"\nOn-disk model size comparison:")
        print(f"  Original (merged):  {original_size_gb:.2f} GB ({original_dir})")
        print(f"  Quantized ({quantization_method}): {quantized_size_gb:.2f} GB ({output_dir})")
        
        if original_size_bytes > 0:
            reduction = (1 - quantized_size_bytes/original_size_bytes) * 100
            print(f"  Reduction: {reduction:.1f}%")
        else:
            print("  ⚠ Warning: Could not find original model for comparison")
            print(f"    Make sure to run merge_and_save() before quantize_int8()")
        
        return quantized_model
    
    def load_quantized_model(self, quantized_dir="./quantized_model"):
        """
        Load a previously saved quantized model to verify it's restorable.
        Returns the loaded model and processor.
        """
        print(f"\nLoading quantized model from {quantized_dir}...")
        
        # Get optimal device configuration
        torch_dtype, device_map = self._get_device_config()
        
        # Try loading HuggingFace format first (bitsandbytes)
        try:
            # Try with optimal config first
            try:
                loaded_model = PaliGemmaForConditionalGeneration.from_pretrained(
                    quantized_dir,
                    torch_dtype=torch_dtype,
                    device_map=device_map
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if device_map != "cpu":
                    print(f"⚠ GPU loading failed ({type(e).__name__}), falling back to CPU...")
                    loaded_model = PaliGemmaForConditionalGeneration.from_pretrained(
                        quantized_dir,
                        torch_dtype=torch.float32,
                        device_map="cpu"
                    )
                else:
                    raise
            
            loaded_processor = AutoProcessor.from_pretrained(quantized_dir)
            print("✓ Successfully loaded quantized model (HuggingFace format)")
            return loaded_model, loaded_processor
            
        except Exception as e:
            # Fallback: Load full torch model
            print(f"HuggingFace loading failed ({e}), trying torch format...")
            model_path = os.path.join(quantized_dir, "model_quantized_full.pt")
            if os.path.exists(model_path):
                loaded_model = torch.load(model_path)
                loaded_processor = AutoProcessor.from_pretrained(quantized_dir)
                print("✓ Successfully loaded quantized model (torch format)")
                return loaded_model, loaded_processor
            else:
                raise FileNotFoundError(f"Could not find quantized model in {quantized_dir}")
    
    def test_inference(self, stroke_text, model=None):
        """Test inference with sample input."""
        print("\nTesting inference...")
        
        # Use provided model or default to self.model
        test_model = model if model is not None else self.model
        
        inputs = self.processor(
            text=stroke_text,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = test_model.generate(
                **inputs,
                max_length=64,
                num_beams=4
            )
        
        predicted = self.processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {stroke_text[:100]}...")
        print(f"Predicted: {predicted}")
        
        return predicted


if __name__ == "__main__":
    exporter = ModelExporter(
        lora_path="./math_lora_checkpoints/lora_epoch_final"
    )
    
    # Merge LoRA weights
    merged_model = exporter.merge_and_save()
    
    # Quantize for edge deployment
    quantized_model = exporter.quantize_int8(merged_model)
    
    # Test with sample using quantized model
    sample_stroke = "<stroke> 112 99 115 102 118 105"
    exporter.test_inference(sample_stroke, model=quantized_model)
    
    # Verify quantized model is restorable
    print("\n" + "="*60)
    print("VERIFICATION: Testing quantized model reload...")
    print("="*60)
    try:
        loaded_model, loaded_processor = exporter.load_quantized_model()
        print("✅ SUCCESS: Quantized model is restorable and ready for deployment!")
    except Exception as e:
        print(f"❌ ERROR: Could not reload quantized model: {e}")

