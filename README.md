# LaTeX-OCR: Image-to-LaTeX Model

A PyTorch & Hugging Face Vision-Language Model that converts images of rendered mathematical formulas into LaTeX code.

## 🏗️ Model Architecture

- **Vision Encoder**: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
- **Language Decoder**: [`EleutherAI/pythia-70m`](https://huggingface.co/EleutherAI/pythia-70m)
- **Projector**: MLP projection layer mapping CLIP patch embeddings into Pythia's hidden embedding space.
- **Dataset**: [`lukbl/LaTeX-OCR-dataset`](https://huggingface.co/datasets/lukbl/LaTeX-OCR-dataset)

---

## 📁 Repository Structure

```
LaTeX-OCR/
├── config/
│   └── config.yaml          # Hyperparameters and model configurations
├── scripts/
│   ├── train.sh             # Run model training
│   ├── evaluate.sh          # Run model evaluation on test/val splits
│   └── upload.sh            # Upload trained checkpoint to Hugging Face Hub
├── src/
│   ├── dataset.py           # Dataset loading & PyTorch collate function
│   ├── evaluate.py          # BLEU-4, Exact Match, and Edit Distance metrics
│   ├── model.py             # LaTeXOCRConfig & LaTeXOCRModel implementation
│   ├── train.py             # Hugging Face Trainer pipeline
│   ├── upload_to_hub.py     # Hub repository creator & model card uploader
│   └── utils.py             # Seed setting, config loading, tokenizer setup
├── requirements.txt         # Project dependencies
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityarajsahu/LaTeX-OCR.git
   cd LaTeX-OCR
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start & Usage

### 1. Training
To start fine-tuning the model in the background using `nohup`:
```bash
./scripts/train.sh
```
Logs are automatically saved to `logs/train_<timestamp>.log` (symlinked to `logs/train.log`). You can monitor training logs in real time via:
```bash
tail -f logs/train.log
```
*(Alternatively, run directly in foreground with `python -m src.train --config config/config.yaml`)*

### 2. Evaluation
To compute performance metrics (Exact Match Rate, Normalized Edit Distance, BLEU-4) on the test set:
```bash
./scripts/evaluate.sh config/config.yaml outputs/checkpoint-best test
# or
python -m src.evaluate --config config/config.yaml --checkpoint outputs/checkpoint-best --split test
```

### 3. Upload to Hugging Face Hub
To upload the best checkpoint alongside an automatically generated Model Card:
```bash
./scripts/upload.sh <your-hf-username>/<repo-name>
# or
python -m src.upload_to_hub --checkpoint outputs/checkpoint-best --repo-id <your-hf-username>/<repo-name>
```

---

## 🐍 Python API Inference Example

```python
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from src.model import LaTeXOCRModel

# Load checkpoint
checkpoint_path = "outputs/checkpoint-best"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
image_processor = CLIPImageProcessor.from_pretrained(checkpoint_path)
model = LaTeXOCRModel.from_pretrained(checkpoint_path)

# Prepare image
image = Image.open("formula.png").convert("RGB")
pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]

# Generate LaTeX
generated_ids = model.generate(pixel_values=pixel_values, tokenizer=tokenizer, max_new_tokens=256)
latex_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Predicted LaTeX:", latex_code)
```

---

## 📜 License
[Apache 2.0](LICENSE)