<!-- 93cbbece-40a9-4b2e-b328-0d1a15fbfd6e 3783521f-902a-4883-99a6-65c342b70870 -->
# Fine-Tune VLM for Handwritten Math Recognition

## 0. Development Workflow & Project Structure

### Workflow: Cursor → GitHub → Colab

**The Strategy:**

1. **Write code in Cursor** (local Mac) - get AI autocomplete, better editor, version control
2. **Push to GitHub** - sync between local and cloud
3. **Pull in Colab** - train on free GPU without uploading files manually
4. **Download trained model** - bring back to local, then deploy to RUBIK Pi

### File Organization Decision

**Key Decision:** How should you structure your code?

**Option A: Everything in one notebook** (Beginner-friendly)

- ✅ Simple - one file does everything
- ✅ Easy to debug - see all code in one place
- ❌ Hard to reuse code
- ❌ Difficult to test individual components
- ❌ Messy if project grows

**Option B: Modular Python files + orchestration notebook** (Recommended)

- ✅ Clean separation of concerns
- ✅ Reusable components you can test independently
- ✅ Professional structure
- ✅ Each file has one clear purpose
- ❌ Slightly more setup

**Your Choice:** I recommend Option B. Here's why and what to create:

```
realtime-math/
├── requirements.txt           # All pip dependencies
├── colab_training.ipynb       # Main notebook (runs in Colab)
├── data_preprocessing.py      # Stroke tokenization logic
├── train.py                   # Training loop and LoRA setup
├── export_model.py            # Quantization and export
├── utils/
│   ├── stroke_tokenizer.py   # Core tokenization functions
│   └── visualization.py       # Visualize strokes and predictions
└── data/                      # Your Mathwriting dataset
```

### Colab Notebook Structure

Your `colab_training.ipynb` will orchestrate everything:

```python
# Cell 1: Clone repo and setup
!git clone https://github.com/YOUR_USERNAME/realtime-math.git
%cd realtime-math
!pip install -r requirements.txt

# Cell 2: Import your modules (this is why modular code rocks)
from data_preprocessing import load_mathwriting_data, create_datasets
from train import setup_model_and_lora, train_model
from export_model import quantize_and_export

# Cell 3: Load data
train_ds, val_ds = create_datasets("data/mathwriting")

# Cell 4: Setup model
model, processor = setup_model_and_lora()

# Cell 5: Train
train_model(model, train_ds, val_ds)

# Cell 6: Export
quantize_and_export(model)

# Cell 7: Download results
from google.colab import files
!zip -r model.zip paligemma-mathwriting-lora/
files.download('model.zip')
```

**Why this works:** Each `.py` file you write in Cursor gets pulled from GitHub automatically. No manual uploading!

### Critical Thinking Questions for You

Before starting, think about:

1. **Version Control:** How will you handle different training experiments? (Hint: Git branches)
2. **Dataset Storage:** Where will you store Mathwriting data? In repo or uploaded to Colab separately?
3. **Checkpointing:** What if Colab disconnects mid-training? How do you resume?
4. **Testing:** How will you test each component before running expensive GPU training?

## 1. Environment Setup (Google Colab)

Create a Colab notebook with GPU runtime and install dependencies:

```python
# Install core libraries
!pip install torch torchvision transformers accelerate peft bitsandbytes datasets pillow
!pip install optimum onnx onnxruntime-gpu
```

**Key Libraries:**

- `transformers`: HuggingFace model loading
- `peft`: LoRA/QLoRA implementation
- `bitsandbytes`: 4-bit quantization for training
- `optimum`: Model optimization for deployment

## 2. Data Preprocessing

Implement digital ink tokenization following the paper's methodology:

**File: `data_preprocessing.py`**

```python
# Key steps from paper (Section 3):
# 1. Time sampling: Resample strokes at fixed intervals
# 2. Scale normalization: Normalize coordinates to fixed range
# 3. Coordinate representation: Use (x, y, time) triplets
# 4. Discretization: Convert continuous coords to discrete tokens
# 5. Token dictionary: Create vocabulary for stroke tokens
```

**Data Format:**

- Input: Stroke sequences `[(x1,y1,t1), (x2,y2,t2), ...]`
- Output: LaTeX math expression strings
- Convert strokes to text tokens that VLM can process

**Mathwriting Dataset Structure:**

- Parse dataset format (likely JSON/XML with stroke coordinates)
- Create train/validation splits
- Tokenize both input strokes and target text

## 3. Model Selection & Loading

Use **PaliGemma** (Gemma-2 based VLM) - optimal for your hardware:

```python
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

model_id = "google/paligemma-3b-pt-224"  # 3B params, fits in 8GB after quantization
processor = AutoProcessor.from_pretrained(model_id)
```

**Why PaliGemma:**

- Built on Gemma-2 (your preference)
- ~3B parameters (fits on RUBIK Pi with quantization)
- Designed for vision-language tasks
- Good documentation and community support

## 4. LoRA Configuration

Apply parameter-efficient fine-tuning:

**File: `train.py`**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Apply to attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# Only trains ~1% of parameters
```

## 5. Training Loop

Implement fine-tuning with the paper's approach:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./paligemma-mathwriting",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,  # Mixed precision
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=custom_collator  # Handles stroke tokenization
)

trainer.train()
```

**Custom Data Collator:**

- Converts stroke sequences to tokens
- Prepares vision encoder inputs (if using rendered strokes)
- Formats target math expressions

## 6. Model Optimization for Edge Deployment

Prepare model for RUBIK Pi 3 (QCS6490):

**Quantization to INT8/INT4:**

```python
# Export LoRA weights
model.save_pretrained("./paligemma-mathwriting-lora")

# Merge LoRA with base model
from peft import PeftModel
base_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
merged_model = PeftModel.from_pretrained(base_model, "./paligemma-mathwriting-lora")
merged_model = merged_model.merge_and_unload()

# Quantize to INT8
from optimum.onnxruntime import ORTQuantizer
quantizer = ORTQuantizer.from_pretrained(merged_model)
quantizer.quantize(save_dir="./quantized_model", quantization_config=qconfig)
```

**Convert to ONNX:**

```python
# For Qualcomm NPU compatibility
merged_model.save_pretrained("./final_model")
# Convert to ONNX format
```

## 7. Deployment to RUBIK Pi 3

**Requirements:**

- Qualcomm Neural Processing SDK (SNPE)
- ONNX Runtime for QNN backend
- Model needs to fit in ~6GB (leaving 2GB for OS)

**Deployment Steps:**

1. **Install Qualcomm SDK on RUBIK Pi:**

   - Download SNPE SDK for QCS6490
   - Setup QNN (Qualcomm Neural Network) backend

2. **Convert Model:**
   ```bash
   # Convert ONNX to DLC (Qualcomm format)
   snpe-onnx-to-dlc --input_network model.onnx --output_path model.dlc
   ```

3. **Optimize for NPU:**
   ```bash
   # Quantize for Hexagon DSP/NPU
   snpe-dlc-quantize --input_dlc model.dlc --output_dlc model_quantized.dlc
   ```

4. **Python Inference Script:**
   ```python
   # Use SNPE Python API for inference
   from qti.aisw.dlc_utils import snpe_dlc_utils
   # Load model and run inference
   ```


## 8. Testing & Validation

**On Colab (after training):**

- Test accuracy on validation set
- Measure inference time
- Compare with paper's results

**On RUBIK Pi 3:**

- Test inference latency (should be <100ms per stroke sequence)
- Monitor memory usage (should stay under 6GB)
- Test concurrent inference capability (12.5 TOPS should handle 2-3 streams)

## Key Files to Create

1. `colab_notebook.ipynb` - Main training notebook
2. `data_preprocessing.py` - Stroke tokenization implementation
3. `train.py` - Training script with LoRA
4. `export_model.py` - Quantization and ONNX export
5. `deploy_rubik.py` - Inference script for RUBIK Pi
6. `requirements.txt` - Python dependencies

## Expected Results

Based on the paper:

- Recognition accuracy: 90%+ on mathematical expressions
- Inference time: 50-100ms per expression (on RUBIK Pi)
- Model size: ~2-3GB (quantized)
- Memory usage: ~4-6GB during inference

### To-dos

- [ ] Create Colab notebook with GPU runtime and install all required dependencies (transformers, peft, bitsandbytes, etc.)
- [ ] Implement stroke tokenization pipeline following paper's methodology (time sampling, normalization, discretization)
- [ ] Load PaliGemma-3B model and configure LoRA for parameter-efficient fine-tuning
- [ ] Train model on Mathwriting dataset with LoRA configuration and monitor validation metrics
- [ ] Merge LoRA weights, quantize to INT8/INT4, and export to ONNX format
- [ ] Convert to Qualcomm DLC format, setup SNPE SDK on RUBIK Pi, and test inference