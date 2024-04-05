import os
import einops
import torch
import torch as th
import torch.nn as nn
import cv2
from pytorch_lightning.utilities.distributed import rank_zero_only
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_state_dict






class CustomNet(LatentDiffusion):
    def __init__(self, 
                text_encoder_config,
                sd_15_ckpt=None,
                use_cond_concat=False,
                use_bbox_mask=False,
                use_bg_inpainting=False,
                learning_rate_scale=10,
                *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.text_encoder = instantiate_from_config(text_encoder_config)

        if sd_15_ckpt is not None:
            self.load_model_from_ckpt(ckpt=sd_15_ckpt)

        self.use_cond_concat = use_cond_concat
        self.use_bbox_mask = use_bbox_mask
        self.use_bg_inpainting = use_bg_inpainting
        self.learning_rate_scale = learning_rate_scale


    def load_model_from_ckpt(self, ckpt, verbose=True):
        print(" =========================== init Stable Diffusion pretrained checkpoint =========================== ")
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        sd_keys = sd.keys()


        missing = []
        text_encoder_sd = self.text_encoder.state_dict()
        for k in text_encoder_sd.keys():
            sd_k = "cond_stage_model."+ k
            if sd_k in sd_keys:
                text_encoder_sd[k] =  sd[sd_k]
            else:
                missing.append(k)

        self.text_encoder.load_state_dict(text_encoder_sd)




    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        params += list(self.cc_projection.parameters())


        params_dualattn = []
        for k, v in self.model.named_parameters():
            if "to_k_text" in k or "to_v_text" in k:
                params_dualattn.append(v)
                print("training weight: ", k)
            else:
                params.append(v)

        
        opt = torch.optim.AdamW([
                                 {'params':params_dualattn, 'lr': lr*self.learning_rate_scale},
                                 {'params': params, 'lr': lr}
                                 ])


        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss


   

    def shared_step(self, batch, **kwargs):

        if 'txt' in self.ucg_training:
            k = 'txt'
            p = self.ucg_training[k]
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    if isinstance(batch[k], list):     
                        batch[k][i] = ""
        
        with torch.no_grad():
            text = batch['txt']
            text_embedding = self.text_encoder(text)


        x, c = self.get_input(batch, self.first_stage_key)

        c["c_crossattn"].append(text_embedding)
        loss = self(x, c,)
        return loss
    

    def apply_model(self, x_noisy, t, cond, return_ids=False,  **kwargs):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon