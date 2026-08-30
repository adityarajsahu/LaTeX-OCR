import argparse
import os
import textwrap
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, CLIPImageProcessor

from src.model import LaTeXOCRModel

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
tags:
  - image-to-text
  - latex-ocr
  - clip
  - pythia
  - vision-language-model
datasets:
  - lukbl/LaTeX-OCR-dataset
---

# {repo_id}

A compact image-to-LaTeX model built by merging:

- **Vision encoder**: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
- **Language decoder**: [`EleutherAI/pythia-70m`](https://huggingface.co/EleutherAI/pythia-70m)
- **Projector**: a small MLP mapping CLIP patch embeddings into Pythia's embedding space

Fine-tuned on [`lukbl/LaTeX-OCR-dataset`](https://huggingface.co/datasets/lukbl/LaTeX-OCR-dataset)
to convert images of rendered math formulas into their LaTeX source.

## Usage

This model uses custom (`trust_remote_code`-free) classes registered in this
repo; load it via the project's `LaTeXOCRModel`, or copy `src/model.py`
into your own code:

```python
from transformers import AutoTokenizer, CLIPImageProcessor
from src.model import LaTeXOCRModel
from PIL import Image

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
image_processor = CLIPImageProcessor.from_pretrained(repo_id)
model = LaTeXOCRModel.from_pretrained(repo_id)

image = Image.open("formula.png").convert("RGB")
pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]

generated_ids = model.generate(pixel_values=pixel_values, tokenizer=tokenizer, max_new_tokens=192)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Evaluation

See `eval_report_test.json` in this repo for BLEU-4, exact-match rate, and
normalized edit-distance on a held-out split, plus sample predictions.

## Training details

Trained with `src/train.py` in the accompanying project repository — see
`configs/config.yaml` there for the exact hyperparameters used.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--repo-id", type = str, required = True)
    parser.add_argument("--private", type = lambda x: x.lower() == "true", default = False)
    parser.add_argument("--commit_message", type = str, default = "Upload fine-tuned LaTeXOCR")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    image_processor = CLIPImageProcessor.from_pretrained(args.checkpoint)
    model = LaTeXOCRModel.from_pretrained(args.checkpoint)

    create_repo(args.repo_id, preivate = args.private, exist_ok = True)

    model.push_to_hub(args.repo_id, commit_message = args.commit_message)
    tokenizer.push_to_hub(args.repo_id, commit_message = args.commit_message)
    image_processor.push_to_hub(args.repo_id, commit_message = args.commit_message)

    card_path = os.path.join(args.checkpoint, "README.md")
    with open(card_path, "w") as f:
        f.write(MODEL_CARD_TEMPLATE.format(repo_id = args.repo_id))

    api = HfApi()
    eval_report = os.path.join(args.checkpoint, "eval_report_test.json")
    api.upload_file(
        path_or_fileobj = card_path,
        path_in_repo = "README.md",
        repo_id = args.repo_id,
        commit_message = "Add model card"
    )

    if os.path.exists(eval_report):
        api.upload_file(path_or_fileobj = eval_report, path_in_repo = "eval_report_test.json", repo_id = args.repo_id, commit_message = "Add evaluation report")

    print(
        textwrap.dedent(
            f"""
            [upload] done.
            View it at: https://huggingface.co/{args.repo_id}
            """
        )
    )

if __name__ == "__main__":
    main()