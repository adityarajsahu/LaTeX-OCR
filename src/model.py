from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel, GPTNeoXConfig, GPTNeoXForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

class LaTeXOCRConfig(PretrainedConfig):
    model_type = "latex_ocr"

    def __init__(self, vision_model_name: str = "openai/clip-vit-base-patch32", language_model_name: str = "EleutherAI/pythia-70m", projector_hidden_dim: int = 1024, freeze_vision_model: bool = True, freeze_language_model: bool = False, vocab_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.projector_hidden_dim = projector_hidden_dim
        self.freeze_vision_model = freeze_vision_model
        self.freeze_language_model = freeze_language_model
        self.vocab_size = vocab_size

@dataclass
class LaTeXOCROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    num_image_tokens: Optional[int] = None

class LaTeXOCRModel(PreTrainedModel):
    config_class = LaTeXOCRConfig

    def __init__(self, config: LaTeXOCRConfig):
        super().__init__(config)

        lm_config = GPTNeoXConfig.from_pretrained(config.language_model_name)
        if getattr(config, "vocab_size", None) is not None:
            lm_config.vocab_size = config.vocab_size

        try:
            self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_model_name, use_safetensors=True)
        except Exception:
            vision_config = CLIPVisionConfig.from_pretrained(config.vision_model_name)
            self.vision_encoder = CLIPVisionModel(vision_config)

        try:
            self.language_model = GPTNeoXForCausalLM.from_pretrained(config.language_model_name, config=lm_config, use_safetensors=True).to(torch.float32)
        except Exception:
            self.language_model = GPTNeoXForCausalLM(lm_config).to(torch.float32)

        vision_hidden_size = self.vision_encoder.config.hidden_size
        lm_hidden_size = self.language_model.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, config.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(config.projector_hidden_dim, lm_hidden_size)
        )

        self._apply_freezing()
        self.post_init()

    def _apply_freezing(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = not self.config.freeze_vision_model
        for p in self.language_model.parameters():
            p.requires_grad = not self.config.freeze_language_model
        for p in self.projector.parameters():
            p.requires_grad = True

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_encoder(pixel_values = pixel_values)
        patch_embeds = vision_outputs.last_hidden_state[:, 1:, :]
        return self.projector(patch_embeds)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs) -> LaTeXOCROutput:
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        image_embeds = image_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim = 1)

        batch_size, num_img_tokens, _ = image_embeds.shape
        device = input_ids.device

        if attention_mask is not None:
            image_attention = torch.ones(batch_size, num_img_tokens, device = device, dtype = attention_mask.dtype)
            full_attention_mask = torch.cat([image_attention, attention_mask], dim = 1)
        else:
            full_attention_mask = None

        full_labels = None
        if labels is not None:
            ignore = torch.full((batch_size, num_img_tokens), -100, device = device, dtype = labels.dtype)
            full_labels = torch.cat([ignore, labels], dim = 1)

        outputs = self.language_model(inputs_embeds = inputs_embeds, attention_mask = full_attention_mask, labels = full_labels, return_dict = True)

        return LaTeXOCROutput(loss = outputs.loss, logits = outputs.logits, past_key_values = outputs.past_key_values, hidden_states = outputs.hidden_states, attentions = outputs.attentions, num_image_tokens = num_img_tokens)

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, tokenizer, max_new_tokens: int = 256, num_beams: int = 4, **gen_kwargs) -> torch.Tensor:
        self.eval()
        image_embeds = self.encode_image(pixel_values)
        batch_size = image_embeds.shape[0]

        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id
        start_ids = torch.full((batch_size, 1), bos_id, device = image_embeds.device, dtype = torch.long)
        start_embeds = self.language_model.get_input_embeddings()(start_ids)
        image_embeds = image_embeds.to(start_embeds.dtype)
        inputs_embeds = torch.cat([image_embeds, start_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=image_embeds.device, dtype=torch.long)

        return self.language_model.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, max_new_tokens = max_new_tokens, num_beams = num_beams, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, **gen_kwargs)

def build_model(config: LaTeXOCRConfig) -> LaTeXOCRModel:
    model = LaTeXOCRModel(config)
    trainable = model.num_trainable_parameters()
    total = sum(p.numel() for p in model.parameters())

    print(f"[model] trainable params: {trainable:,} / {total:,} ({trainable / total:.1%})")

    return model

# if __name__ == "__main__":
#     model_config = LaTeXOCRConfig()
#     model = build_model(model_config)
#     print(model)

#     print("Number of trainable parameters:", model.num_trainable_parameters())
