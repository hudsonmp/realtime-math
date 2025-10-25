<!-- b2dc0076-6ff3-4f6a-bc6a-e94471c4752d a9959b97-8fc7-4fe3-8335-b21e6529c337 -->
# Real-Time Handwritten Math Recognition System

## Goal

Build a transformer-based model using PyTorch to convert handwritten mathematical expressions (digital ink) into LaTeX format, which will feed into an AI tutoring system.

## Architecture Overview

We'll implement an encoder-decoder transformer architecture based on the research paper approach:

- **Encoder**: Process stroke sequences (x, y, t coordinates) 
- **Decoder**: Generate LaTeX tokens autoregressively
- **Tokenization**: Discrete stroke representation + LaTeX vocabulary

## Implementation Plan

### 1. Data Processing Pipeline (`data_preprocessing.py`)

- **InkML Parser**: Parse XML files to extract stroke data and normalized labels
- **Stroke Normalization**: 
- Scale normalization (normalize coordinates to fixed range)
- Time sampling (resample strokes to uniform temporal spacing)
- Center and scale coordinates
- **Stroke Tokenization** (`utils/stroke_tokenizer.py`):
- Discretize continuous coordinates into tokens
- Handle variable-length stroke sequences
- **LaTeX Tokenization**: Build vocabulary from normalized labels
- **PyTorch Dataset**: Create efficient data loaders for train/valid/test splits

### 2. Model Architecture (`train.py`)

- **Stroke Encoder**:
- Embedding layer for discretized stroke tokens
- Positional encoding for temporal information
- Multi-head self-attention transformer encoder layers
- **LaTeX Decoder**:
- Embedding layer for LaTeX tokens
- Positional encoding
- Masked multi-head attention transformer decoder layers
- Cross-attention to encoder outputs
- **Training Loop**:
- Teacher forcing during training
- Cross-entropy loss on LaTeX predictions
- AdamW optimizer with learning rate scheduling
- Gradient clipping for stability

### 3. Model Export (`export_model.py`)

- Export trained model for inference
- ONNX or TorchScript format for deployment
- Optimize for real-time performance

### 4. Utilities

- **Visualization** (`utils/visualization.py`): 
- Render strokes as images
- Display predictions vs ground truth
- **Evaluation Metrics**:
- Exact match accuracy
- Character error rate (CER)
- LaTeX edit distance

### 5. Dependencies (`requirements.txt`)

Core packages needed:

- `torch>=2.0.0` - PyTorch framework
- `torchvision` - Vision utilities
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `matplotlib` - Visualization
- `lxml` - XML parsing for InkML

## Key Technical Decisions

**Stroke Representation**: Following the paper's approach, we'll tokenize strokes as discrete sequences rather than rendering to images. This preserves temporal information crucial for handwriting understanding.

**Normalization Strategy**: Scale and center each ink sample to a standard coordinate space, then discretize into vocabulary of ~1000 coordinate tokens.

**Model Size**: Start with a compact transformer (6 encoder layers, 6 decoder layers, 512 hidden dim) to train quickly, then scale up if needed.

**Training Strategy**:

- Use combined train + synthetic data (~626k samples)
- Validate on validation set (15k samples)
- Final evaluation on test set (7k samples)

## Files to Modify

- `requirements.txt` - Add dependencies
- `data_preprocessing.py` - Data loading and preprocessing
- `utils/stroke_tokenizer.py` - Stroke tokenization logic
- `utils/visualization.py` - Visualization utilities
- `train.py` - Model architecture and training loop
- `export_model.py` - Model export for deployment

### To-dos

- [x] Create requirements.txt with PyTorch, numpy, lxml, matplotlib, and other dependencies
- [x] Implement InkML parser to extract strokes and labels from XML files
- [x] Build stroke tokenizer with normalization, discretization, and vocabulary
- [x] Create LaTeX tokenizer and build vocabulary from training labels
- [x] Implement PyTorch Dataset and DataLoader for efficient batch processing
- [x] Build encoder-decoder transformer architecture with positional encodings
- [x] Implement training loop with loss, optimizer, and learning rate scheduling
- [x] Add evaluation metrics (accuracy, CER, edit distance) and validation loop
- [x] Create visualization tools to render strokes and display predictions
- [ ] Implement model export to ONNX/TorchScript for deployment