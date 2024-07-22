import argparse
import os
import tempfile
from functools import partial
import cv2
import gradio as gr
import imageio
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import copy 
from utils.gradio_utils import load_preprocess_model, preprocess_image
from ldm.util import instantiate_from_config, img2tensor
from customnet.ddim import DDIMSampler
from einops import rearrange
import math



#### Description ####
title = r"""<h1 align="center">CustomNet: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models</h1>"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/TencentARC/CustomNet' target='_blank'><b>CustomNet: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models</b></a>.<br>
🔥 CustomNet is novel unified customization method that can generate harmonious customized images without
test-time optimization. CustomNet supports explicit viewpoint, location, text controls while ensuring
object identity preservation.<br>
🤗 Try to customize the object gneration yourself!<br>
"""
article = r"""
If CustomNet is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/CustomNet' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC%2FCustomNet)](https://github.com/TencentARC/CustomNet)

---

📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@misc{yuan2023customnet,
    title={CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models}, 
    author={Ziyang Yuan and Mingdeng Cao and Xintao Wang and Zhongang Qi and Chun Yuan and Ying Shan},
    year={2023},
    eprint={2310.19784},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>yuanzy22@mails.tsinghua.edu.cn</b>.

"""

# input_img = None
# concat_img = None
# T = None
# prompt = None
negtive_prompt = ""


def send_input_to_concat(input_image):
    W, H = input_image.size
    # image_array[:, 0, :] = image_array[:, 0, :]
    draw = ImageDraw.Draw(input_image)
    draw.rectangle([(0,0),(H-1, W-1)], outline="red", width=8)
    return input_image

def preprocess_input(preprocess_model, input_image):
    # global input_img
    processed_image = preprocess_image(preprocess_model, input_image)
    # input_img = (processed_image / 255.0).astype(np.float32)
    return processed_image
    # return processed_image, processed_image

def adjust_location(x0, y0, x1, y1, input_image):
    x_0 = min(x0, x1)
    x_1 = max(x0, x1)
    y_0 = min(y0, y1)
    y_1 = max(y0, y1)
    print(x0, y0, x1, y1)
    print(x_0, y_0, x_1, y_1)
    new_size = (x_1-x_0, y_1-y_0)
    input_image = input_image.resize(new_size)
    img_array = np.array(input_image)
    white_background = np.zeros((256, 256, 3))
    white_background[y0:y1, x0:x1, :] = img_array
    img_array = white_background.astype(np.uint8)
    concat_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(concat_img)
    draw.rectangle([(x0,y0),(x1,y1)], outline="red", width=5)
    return x_0, y_0, x_1, y_1, concat_img

def prepare_data(device, input_image, x0, y0, x1, y1, polar, azimuth, text):
    if input_image.size[0] != 256 or input_image.size[1] != 256:
        input_image = input_image.resize((256, 256))
    input_image = np.array(input_image)

    img_cond = img2tensor(input_image, bgr2rgb=False, float32=True).unsqueeze(0) / 255.
    img_cond = img_cond*2-1

    img_location = copy.deepcopy(img_cond)
    input_im_padding = torch.ones_like(img_location)

    x_0 = min(x0, x1)
    x_1 = max(x0, x1)
    y_0 = min(y0, y1)
    y_1 = max(y0, y1)
    print(x0, y0, x1, y1)
    print(x_0, y_0, x_1, y_1)
    img_location = torch.nn.functional.interpolate(img_location, (y_1-y_0, x_1-x_0), mode="bilinear")
    input_im_padding[:,:, y_0:y_1, x_0:x_1] = img_location
    img_location = input_im_padding

    T = torch.tensor([[math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), 0.0]]).unsqueeze(1)
    batch = {
            "image_cond": img_cond.to(device),
            "image_location": img_location.to(device),
            'T': T.to(device),
            'text': [text],
            }
    return batch


@torch.no_grad()
def run_generation(sampler, model, device, input_image, x0, y0, x1, y1, polar, azimuth, text, seed):
    seed_everything(seed)
    batch = prepare_data(device, input_image, x0, y0, x1, y1, polar, azimuth, text)

    c = model.get_learned_conditioning(batch["image_cond"])
    c = torch.cat([c, batch["T"]], dim=-1)
    c = model.cc_projection(c)
    
    ## condition
    cond = {}
    cond['c_concat'] = [model.encode_first_stage((batch["image_location"])).mode().detach()]
    cond['c_crossattn'] = [c]
    text_embedding = model.text_encoder(batch["text"])
    cond["c_crossattn"].append(text_embedding)

    ## null-condition
    uc = {}
    neg_prompt = ""

    uc['c_concat'] = [torch.zeros(1, 4, 32, 32).to(c.device)]
    uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
    uc_text_embedding = model.text_encoder([neg_prompt])
    uc['c_crossattn'].append(uc_text_embedding)

    ## sample
    shape = [4, 32, 32]
    samples_latents, _ = sampler.sample(
            S=50, 
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=999,  # useless
            conditioning=cond,
            unconditional_conditioning=uc,
            cfg_type=0,
            cfg_scale_dict={"img": 0., "text":0., "all": 3.0 }
        )
        
    x_samples = model.decode_first_stage(samples_latents)

    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu().numpy()
    x_samples = rearrange(255.0 *x_samples[0], 'c h w -> h w c').astype(np.uint8)
    
    output_image = Image.fromarray(x_samples)

    return output_image


def load_example(input_image, x0, y0, x1, y1, polar, azimuth, prompt):
    # print("AAAA")
    # print(type(x0))
    # print(type(polar))
    return input_image, x0, y0, x1, y1, polar, azimuth, prompt

@torch.no_grad()
def main(args):
    # load model
    device = torch.device("cuda")
    preprocess_model = load_preprocess_model()
    config = OmegaConf.load("configs/config_customnet_inpaint.yaml") 
    model = instantiate_from_config(config.model)
    ckpt = torch.load("pretrain/customnet_inpaint_v1.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    del ckpt
    model = model.to(device)
    sampler = DDIMSampler(model, device=device)
    

    ## prepare data

    img_cv2 = cv2.imread(args.obj_img)
    img_cond = img2tensor(cv2.resize(img_cv2, (256, 256)), bgr2rgb=True, float32=True) / 255.
    img_cond = img_cond*2-1
    img_cond = img_cond.unsqueeze(0)

    img_location = copy.deepcopy(img_cond)
    input_im_padding = torch.ones_like(img_location)
    x_0, y_0, x_1, y_1 = [ int(_) for _ in args.bbox.split(',') ]
    print(x_0, y_0, x_1, y_1 )
    print(img_location.shape)
    img_location = torch.nn.functional.interpolate(img_location, (y_1-y_0, x_1-x_0), mode="bilinear")
    input_im_padding[:, :, y_0:y_1, x_0:x_1] = img_location
    img_location = input_im_padding

    bg = cv2.imread(args.bg_img)
    bg = img2tensor(cv2.resize(bg, (256, 256)), bgr2rgb=True, float32=True) / 255.
    bg = bg*2-1
    bg = bg.unsqueeze(0)
    bg[:,:, y_0:y_1, x_0:x_1] = 1

    T = torch.tensor([[math.radians(args.polar), math.sin(math.radians(args.azimuth)), math.cos(math.radians(args.azimuth)), 0.0]]).unsqueeze(1)


    img_cond = img_cond.to(device)
    img_location = img_location.to(device)
    bg = bg.to(device)
    T = T.to(device)
    text = [args.text]

    ## model input
    c = model.get_learned_conditioning(img_cond)
    c = torch.cat([c, T], dim=-1)
    c = model.cc_projection(c)
    bg_concat = model.encode_first_stage(bg).mode().detach()

    ## condition
    cond = {}
    cond['c_concat'] = [model.encode_first_stage(img_location).mode().detach()]
    cond['c_crossattn'] = [c]
    text_embedding = model.text_encoder(text)
    cond["c_crossattn"].append(text_embedding)
    cond['c_concat'].append(bg_concat)

    ## null-condition
    uc = {}
    neg_prompt = ""

    uc['c_concat'] = [torch.zeros(1, 4, 32, 32).to(c.device)]
    uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
    uc_text_embedding = model.text_encoder([neg_prompt])
    uc['c_crossattn'].append(uc_text_embedding)
    uc["c_concat"].append(bg_concat)

    ## sample
    shape = [4, 32, 32]
    samples_latents, _ = sampler.sample(
            S=50, 
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=999,  # useless
            conditioning=cond,
            unconditional_conditioning=uc,
            cfg_type=0,
            cfg_scale_dict={"img": 0., "text":0., "all": 3.0 }
        )
        
    x_samples = model.decode_first_stage(samples_latents)

    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu().numpy()
    x_samples = rearrange(255.0 *x_samples[0], 'c h w -> h w c').astype(np.uint8)
    
    output_image = Image.fromarray(x_samples)


    os.makedirs(args.save_dir, exist_ok=True)
    output_image.save(os.path.join(args.save_dir, os.path.basename(args.obj_img)))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_img", type=str, default="<path_to_obj_img>")
    parser.add_argument("--bg_img", type=str, default="<path_to_bg_img>")
    parser.add_argument("--bbox", type=str, default='50,50,200,200')
    parser.add_argument("--text", type=str, default='')
    parser.add_argument("--polar", type=float, default=0)
    parser.add_argument("--azimuth", type=float, default=0)
    parser.add_argument("--save_dir", type=str, default='outputs')

    args = parser.parse_args()

    main(args)
    # main(launch_kwargs)
