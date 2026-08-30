import argparse
import os
import torch
from transformers import Trainer, TrainingArguments

from src.dataset import LaTeXOCRDataset, collate_fn, load_latex_ocr_splits
from src.model import build_model
from src.utils import build_model_config, build_tokenizer_and_processor, load_config, resize_for_added_tokens, set_seed

class LaTeXOCRTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        inputs = {k: v for k, v in inputs.items() if k != "text"}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default = "config/config.yaml")
    parser.add_argument("--resume_from_checkpoint", type = str, default = None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    tokenizer, image_processor = build_tokenizer_and_processor(cfg)

    print("[data] loading and splitting dataset...")
    splits = load_latex_ocr_splits(cfg["dataset_name"], cfg["val_fraction"], cfg["test_fraction"], cfg["seed"])
    print(f"[data] train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")

    train_dataset = LaTeXOCRDataset(splits["train"], image_processor, tokenizer, cfg["max_text_length"], is_train = True)
    eval_dataset = LaTeXOCRDataset(splits["validation"], image_processor, tokenizer, cfg["max_text_length"], is_train = False)

    model_config = build_model_config(cfg)
    model = build_model(model_config)
    resize_for_added_tokens(model, tokenizer)

    os.makedirs(cfg["output_dir"], exist_ok = True)

    if "warmup_steps" in cfg:
        warmup_steps = cfg["warmup_steps"]
    elif "warmup_ratio" in cfg:
        effective_batch_size = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = steps_per_epoch * cfg["num_train_epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
    else:
        warmup_steps = 0

    training_args = TrainingArguments(
        output_dir = cfg["output_dir"],
        num_train_epochs = cfg["num_train_epochs"],
        per_device_train_batch_size = cfg["per_device_train_batch_size"],
        per_device_eval_batch_size = cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps = cfg["gradient_accumulation_steps"],
        learning_rate = cfg["learning_rate"],
        weight_decay = cfg["weight_decay"],
        warmup_steps = warmup_steps,
        lr_scheduler_type = cfg["lr_scheduler_type"],
        logging_steps = cfg["logging_steps"],
        eval_strategy = "steps",
        eval_steps = cfg["eval_steps"],
        save_strategy = "steps",
        save_steps = cfg["save_steps"],
        save_total_limit = cfg["save_total_limit"],
        fp16 = cfg["fp16"] and torch.cuda.is_available(),
        dataloader_num_workers = cfg["dataloader_num_workers"],
        report_to = cfg["report_to"],
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        remove_unused_columns = False
    )

    trainer = LaTeXOCRTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = collate_fn
    )

    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)

    best_dir = os.path.join(cfg["output_dir"], "checkpoint-best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    image_processor.save_pretrained(best_dir)

    print(f"[train] best model + tokenizer + image processor saved to {best_dir}")

if __name__ == "__main__":
    main()