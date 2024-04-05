import os
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import kornia
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn.functional as F 
from torchvision import transforms
import random
from ldm.util import default, instantiate_from_config
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
import clip

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, return_pool=False):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        if return_pool:
            return z, outputs.pooler_output
        else:
            return z

    def encode(self, text):
        return self(text)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """
        Uses the CLIP image encoder.
        Not actually frozen... If you want that set cond_stage_trainable=False in cfg
        """
    def __init__(
            self,
            model='ViT-L/14',
            jit=False,
            device='cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit,)
        # We don't use the text part so delete it
        del self.model.transformer
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # Expects inputs in the range -1, 1
        # x = kornia.geometry.resize(x, (224, 224),
        #                            interpolation='bicubic',align_corners=True,
        #                            antialias=self.antialias)

        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True)

        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        if isinstance(x, list):
            # [""] denotes condition dropout for ucg
            device = self.model.visual.conv1.weight.device
            return torch.zeros(1, 768, device=device)
        return self.model.encode_image(self.preprocess(x)).float()

    def encode(self, im):
        return self(im).unsqueeze(1)

