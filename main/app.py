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
    config = OmegaConf.load("configs/config_customnet.yaml") 
    model = instantiate_from_config(config.model)
    ckpt = torch.load("pretrain/customnet_v1.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    del ckpt
    model = model.to(device)
    sampler = DDIMSampler(model, device=device)

    # load demo
    demo = gr.Blocks()
    with demo:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            ## Left column
            with gr.Column():
                ## step 1. 
                gr.Markdown("## Step 1: Upload an object image and process", show_label=False)
                # with gr.Row(equal_height=True):
                input_image = gr.Image(type="pil", interactive=True, elem_id="input_image", elem_classes='image', visible=True)
                preprocess_botton = gr.Button(value="Need preprocess", visible=True)

                ## step 2. 
                gr.Markdown("## Step 2: Set up different controls ", show_label=False, visible=True)
                gr.Markdown("### 1: Object Location", show_label=False, visible=True)
                with gr.Row():    
                    with gr.Column():
                        with gr.Row():  
                            x0 = gr.Slider(minimum=0, maximum=256, step=1, label="X_0", value=0, interactive=True, visible=True)
                            y0 = gr.Slider(minimum=0, maximum=256, step=1, label="Y_0", value=0, interactive=True, visible=True)
                        with gr.Row():  
                            x1 = gr.Slider(minimum=0, maximum=256, step=1, label="X_1", value=256, interactive=True, visible=True)
                            y1 = gr.Slider(minimum=0, maximum=256, step=1, label="Y_1", value=256, interactive=True, visible=True)
                        location_botton = gr.Button(value="Update Location ", visible=True)

                    location_image = gr.Image(type="pil", interactive=True, elem_id="location", elem_classes='image', visible=True)
                gr.Markdown("### 2: Object Viewpoint", show_label=False, visible=True)
                with gr.Row():    
                    polar = gr.Slider(minimum=-30, maximum=30, step=-0.5, label="Polar Angle", value=0.0, visible=True)
                    azimuth = gr.Slider(minimum=-60, maximum=60, step=-0.5, label="Azimuth angle", value=0.0, visible=True)
                gr.Markdown("### 3: Text", show_label=False, visible=True)
                prompt = gr.Textbox(value="on the seaside", label="Prompt", interactive=True, visible=True)

                ## step 3. 
                gr.Markdown("## Step 3: Run Generation", show_label=False, visible=True)
                seed = gr.Number(value=1234, precision=0, interactive=True, label="Seed", visible=True)
                start = gr.Button(value="Run generation !", visible=True)



            examples_full = [
                ["examples/0.jpg", 50, 50, 256, 256, 0, -30, "a backpack in the office"],
                ["examples/1.jpg", 20, 20, 256, 256, -25, -35, "a pair of shoes on dirt road"],
                ["examples/2.jpg", 0, 0, 256, 256, -15, -20, "a car on the beach"],
                ["examples/3.jpg", 0, 0, 256, 256, 0, 30, "in the jungle"],
                ["examples/4.jpg", 0, 0, 256, 256, 0, -30, "in the snow"],
                ["examples/5.jpg", 20, 20, 240, 240, 10, 20, "with mountain behind"],
            ]

            ## Right column
            with gr.Column():
                gr.Markdown("## Generation Results", show_label=False, visible=True)
                output_image = gr.Image(type="pil", interactive=True, elem_id="output_image", elem_classes='image', visible=True)

                gr.Examples(
                    examples=examples_full,  # NOTE: elements must match inputs list!
                    fn=load_example,
                    inputs=[input_image, x0, y0, x1, y1, polar, azimuth, prompt],
                    outputs=[input_image, x0, y0, x1, y1, polar, azimuth, prompt],
                    cache_examples=False,
                    run_on_click=True,
                )
        gr.Markdown(article)

        ## function


        input_image.change(send_input_to_concat, inputs=input_image, outputs=location_image)
        preprocess_botton.click(partial(preprocess_input, preprocess_model), inputs=input_image, outputs=input_image)
        location_botton.click(adjust_location, 
                                inputs=[x0, y0, x1, y1, input_image], 
                                outputs=[x0, y0, x1, y1, location_image])

        start.click(partial(run_generation, sampler, model, device), 
                                inputs=[input_image, x0, y0, x1, y1, polar, azimuth, prompt, seed], 
                                outputs=output_image)
                                


    demo.launch(server_name='0.0.0.0', share=False, server_port=args.port)
    # demo.queue(concurrency_count=1, max_size=10)
    # demo.launch()
    # demo.queue(max_size=10).launch(**args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12521)

    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    main(args)
    # main(launch_kwargs)
