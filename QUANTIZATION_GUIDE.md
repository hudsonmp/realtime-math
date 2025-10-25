# Quantization Guide for RUBIK Pi Deployment

## Changes Made to `export_model.py`

### 1. Fixed Serialization (Lines 36-132)
**Problem:** Previously saved only `model.state_dict()`, making quantized models unloadable.

**Solution:** 
- **Primary method:** Uses `bitsandbytes` 8-bit quantization (GPU-optimized) with `save_pretrained()` for full HuggingFace compatibility
- **Fallback:** Uses `torch.save(model, path)` to save the complete model object (architecture + weights)
- Both approaches ensure the quantized model is fully restorable

### 2. Fixed Size Comparison (Lines 100-131)
**Problem:** Mixed in-memory parameter bytes with on-disk pickle size (apples-to-oranges comparison).

**Solution:**
- Now compares **on-disk sizes** for both original and quantized models
- Uses `get_directory_size()` to calculate total size of all files in model directories
- Provides accurate storage reduction metrics for deployment planning

### 3. Added Model Reloading Verification (Lines 134-161)
**New feature:** `load_quantized_model()` method demonstrates that saved models are restorable.
- Tries HuggingFace format first (bitsandbytes models)
- Falls back to torch format if needed
- Called in main block to verify successful serialization

### 4. Added Quantization Method Notes (Lines 37-47)
**Documentation:** Comments explain:
- Why `bitsandbytes` is recommended for GPU deployment (your use case)
- Why `torch.quantization.quantize_dynamic` is CPU-focused
- Alternative options: TensorRT (max performance), Optimum (HF-integrated), ONNX Runtime (cross-platform)

## Installation

```bash
# Core dependencies
pip install torch>=2.0.0 transformers>=4.40.0 peft>=0.8.0

# For GPU-optimized quantization (RECOMMENDED for RUBIK Pi)
pip install bitsandbytes>=0.41.0
```

## Usage

```python
from export_model import ModelExporter

# Initialize with your LoRA checkpoint
exporter = ModelExporter(
    lora_path="./math_lora_checkpoints/lora_epoch_final"
)

# Step 1: Merge LoRA weights with base model
merged_model = exporter.merge_and_save(output_dir="./merged_model")

# Step 2: Quantize for GPU edge deployment
quantized_model = exporter.quantize_int8(
    merged_model, 
    output_dir="./quantized_model"
)

# Step 3: Verify quantized model is restorable
loaded_model, loaded_processor = exporter.load_quantized_model(
    quantized_dir="./quantized_model"
)

# Step 4: Test inference
sample_stroke = "<stroke> 112 99 115 102 118 105"
prediction = exporter.test_inference(sample_stroke)
```

## Expected Output

```
Loading base model...
Loading LoRA weights...
Loading processor...
Merging LoRA weights...
Saving merged model to ./merged_model...
✓ Model merged and saved!

Quantizing to INT8...
Using bitsandbytes 8-bit quantization (GPU-optimized)...
Saving quantized model to ./quantized_model...
✓ Model quantized and saved!

Calculating on-disk sizes...

On-disk model size comparison:
  Original (merged):  5.42 GB (./merged_model)
  Quantized (bitsandbytes): 2.71 GB (./quantized_model)
  Reduction: 50.0%

VERIFICATION: Testing quantized model reload...
Loading quantized model from ./quantized_model...
✓ Successfully loaded quantized model (HuggingFace format)
✅ SUCCESS: Quantized model is restorable and ready for deployment!
```

## Deployment to RUBIK Pi

1. **Copy quantized model:** Transfer `./quantized_model/` directory to RUBIK Pi
2. **Load on device:**
   ```python
   from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
   
   model = PaliGemmaForConditionalGeneration.from_pretrained(
       "./quantized_model",
       device_map="auto"  # Auto-maps to GPU
   )
   processor = AutoProcessor.from_pretrained("./quantized_model")
   ```
3. **Run inference:** Model is ready for low-latency GPU inference

## Performance Expectations (3B Parameter VLM)

| Method | Device | Typical Latency | Memory | Storage |
|--------|--------|-----------------|--------|---------|
| Original FP16 | GPU | ~200-300ms | ~6GB | ~5.4GB |
| bitsandbytes 8-bit | GPU | ~150-250ms | ~3GB | ~2.7GB |
| torch quantize_dynamic | CPU | ~800-1200ms | ~2GB | ~2.5GB |
| TensorRT FP16 | GPU | ~100-150ms | ~4GB | ~5GB |

**Recommendation:** Use `bitsandbytes` (implemented) for balanced latency/memory on GPU.

## Troubleshooting

**If bitsandbytes fails to install:**
- Ensure CUDA is properly installed: `nvcc --version`
- Try pre-built wheels: `pip install bitsandbytes --prefer-binary`
- Fallback: Code automatically uses `torch.quantization` (CPU-focused)

**If quantized model fails to load:**
- Check that `./quantized_model/` contains model files
- Verify sufficient GPU memory available
- Try loading with `device_map="cpu"` for debugging

## Further Optimization Options

For maximum performance on RUBIK Pi, consider:

1. **TensorRT** (NVIDIA GPUs only, best latency):
   ```bash
   pip install tensorrt onnx
   # Convert model to ONNX, then optimize with TensorRT
   ```

2. **Optimum with GPTQ** (better compression):
   ```bash
   pip install optimum auto-gptq
   # Quantize with 4-bit GPTQ
   ```

3. **Flash Attention** (faster inference):
   ```bash
   pip install flash-attn
   # Enable in model config
   ```

