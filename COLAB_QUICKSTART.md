# 🚀 Google Colab Quickstart Guide

## ✅ What's Been Done

Your notebook is now fully configured to:
1. ✅ Download the **full MathWriting-2024 dataset** (2.9 GB) from Google Cloud Storage
2. ✅ Clone your Python files from GitHub automatically
3. ✅ Run comprehensive init and smoke tests
4. ✅ Train PaliGemma-3B with LoRA on the full dataset

## 📖 How to Use

### Step 1: Open Notebook in Colab

**Option A: Direct GitHub URL (Easiest)**
```
https://colab.research.google.com/github/hudsonmp/realtime-math/blob/main/colab_training.ipynb
```

**Option B: From Colab Interface**
1. Go to https://colab.research.google.com/
2. Click "File" → "Open notebook"
3. Click "GitHub" tab
4. Enter: `hudsonmp/realtime-math`
5. Select: `colab_training.ipynb`

### Step 2: Configure Runtime

1. Click "Runtime" → "Change runtime type"
2. Hardware accelerator: **GPU** (T4/A100)
3. Click "Save"

### Step 3: Add HuggingFace Token

1. Get token from: https://huggingface.co/settings/tokens
2. Request access: https://huggingface.co/google/paligemma-3b-pt-224
3. In Colab, click 🔑 icon (left sidebar)
4. Add new secret:
   - Name: `HF_TOKEN`
   - Value: `your_token_here`

### Step 4: Run Cells in Order

**Cell 1: Init Test** ⚠️ RUN THIS FIRST!
- Checks GPU, HF auth, paths, files
- Takes ~30 seconds
- Must pass before continuing

**Cells 2-5: Environment Setup**
- Installs dependencies
- Authenticates with HuggingFace

**Cell 6: Download Dataset** 📥
- **Automatically downloads 2.9 GB dataset from Google Cloud**
- URL: `https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz`
- Takes 5-10 minutes depending on connection
- Extracts to `/content/mathwriting-2024/`

**Cell 7: Clone Python Files** 📁
- Clones from: `https://github.com/hudsonmp/realtime-math.git`
- Copies `data_preprocessing.py`, `train.py`, `test_pipeline.py`

**Cell 8: Smoke Test** 🧪
- Validates entire pipeline with small subset
- Takes 2-3 minutes
- Tests: data loading, model init, training, validation
- **Run before full training!**

**Cells 9+: Training**
- Visualize samples
- Load full dataset
- Train PaliGemma-3B with LoRA

## 📊 Dataset Details

**Full Dataset (Auto-Downloaded):**
- Size: 2.9 GB compressed
- URL: https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
- Structure:
  ```
  /content/mathwriting-2024/
  ├── train/       (~195k .inkml files)
  ├── valid/       (~29k .inkml files)
  ├── test/        (~29k .inkml files)
  ├── symbols/     (~100 .inkml files)
  └── synthetic/   (~100 .inkml files)
  ```

**Total: ~253k samples**

## 🎯 Expected Timeline

| Task | Duration |
|------|----------|
| Environment setup | 2-3 min |
| Dataset download | 5-10 min |
| Smoke test | 2-3 min |
| Full training | 4-8 hours |

## ⚠️ Troubleshooting

**Download fails?**
- Manual download: https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
- Upload to Colab, then: `!tar -xzf mathwriting-2024.tgz -C /content/`

**HF auth fails?**
- Ensure token is added to Colab secrets (🔑 icon)
- Request access to PaliGemma: https://huggingface.co/google/paligemma-3b-pt-224

**Files missing?**
- Cell 7 should auto-clone from GitHub
- Fallback: Upload manually via file browser (📂 icon)

## 📚 References

- **Paper:** https://arxiv.org/abs/2404.10690
- **Dataset:** https://github.com/google-research/google-research/tree/master/mathwriting
- **Your Repo:** https://github.com/hudsonmp/realtime-math

## ✅ Verification Checklist

Before training:
- [ ] Init test passes all checks
- [ ] GPU is available (T4 or better)
- [ ] HF token added to secrets
- [ ] Dataset downloaded (2.9 GB, ~253k files)
- [ ] Python files cloned from GitHub
- [ ] Smoke test passes (4/4 tests)

Then you're ready to train! 🚀
