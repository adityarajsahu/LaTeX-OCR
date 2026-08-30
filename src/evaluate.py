import argparse
import json
import os
import Levenshtein
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from src.dataset import LaTeXOCRDataset, collate_fn, load_latex_ocr_splits
from src.model import LaTeXOCRModel
from src.utils import load_config, set_seed

def normalize(s: str) -> str:
    return " ".join(s.strip().split())

@torch.no_grad()
def run_evaluation(model, tokenizer, dataloader, device, max_new_tokens, num_beams):
    model.eval()
    predictions, references = [], []

    for batch in tqdm(dataloader, desc = "evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        generated_ids = model.generate(pixel_values = pixel_values, tokenizer = tokenizer, max_new_tokens = max_new_tokens, num_beams = num_beams)

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
        predictions.extend(normalize(p) for p in decoded)
        references.extend(normalize(t) for t in batch["text"])

    return predictions, references

def compute_metrics(predictions, references):
    exact_matches = sum(p == r for p, r in zip(predictions, references))
    exact_match_rate = exact_matches / len(references)

    edit_distances = [Levenshtein.distance(p, r) / max(len(r), 1) for p, r in zip(predictions, references)]
    mean_norm_edit_distance = sum(edit_distances) / len(edit_distances)

    smoothing = SmoothingFunction().method4
    bleu = corpus_bleu([[r.split()] for r in references], [p.split() for p in predictions], smoothing_function=smoothing)

    return {
        "exact_match_rate": exact_match_rate,
        "mean_normalized_edit_distance": mean_norm_edit_distance,
        "bleu4": bleu,
        "num_examples": len(references)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default = "configs/config.yaml")
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--split", type = str, default = "test", choices = ["validation", "test"])
    parser.add_argument("--batch_size", type = int, default = 16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    image_processor = CLIPImageProcessor.from_pretrained(args.checkpoint)
    model = LaTeXOCRModel.from_pretrained(args.checkpoint).to(device)

    splits = load_latex_ocr_splits(cfg["dataset_name"], cfg["val_fraction"], cfg["test_fraction"], cfg["seed"])
    eval_dataset = LaTeXOCRDataset(splits[args.split], image_processor, tokenizer, cfg["max_text_length"])
    dataloader = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    predictions, references = run_evaluation(model, tokenizer, dataloader, device, cfg["eval_max_new_tokens"], cfg["eval_num_beams"])
    metrics = compute_metrics(predictions, references)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    report_path = os.path.join(args.checkpoint, f"eval_report_{args.split}.json")
    n_samples = cfg["eval_num_samples_to_dump"]
    with open(report_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "samples": [
                {"prediction": p, "reference": r} for p, r in zip(predictions[:n_samples], references[:n_samples])
            ]
        }, f, indent = 4)
    
    print(f"\n[eval] full report + sample predictions written to {report_path}")

if __name__ == "__main__":
    main()