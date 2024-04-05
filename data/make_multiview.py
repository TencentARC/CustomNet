import os 
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append("examples/FastSAM")
sys.path.append("examples/zero123/zero123")
import json
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice
import cv2
import numpy as np
import argparse
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
from ldm.models.diffusion.ddim import DDIMSampler
from carvekit.api.high import HiInterface
import json
from contextlib import nullcontext
from torchvision import transforms
import math
from einops import rearrange
import io

from ldm.util import instantiate_from_config
from fastsam import FastSAM
from fastsam.prompt import FastSAMPrompt 
import clip


def filter_keys(x):
    try:
        return ("jpg" in x) or ("png" in x)
    except Exception:
        return False  


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:

        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model



def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def load_and_preprocess(interface, input_im):
    '''
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    '''
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert('RGB')

    image_without_background = interface([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, : , -1].astype(np.bool_)
    image[~foreground] = [255., 255., 255.]
    x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    image = image[y:y+h, x:x+w, :]
    image = Image.fromarray(np.array(image))
    
    # resize image such that long edge is 512
    image.thumbnail([200, 200], Image.Resampling.LANCZOS)
    image = add_margin(image, (255, 255, 255), size=256)
    image = np.array(image)
    
    return image, [x, y, w, h]


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            n_samples=2
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)      ## 
            T1 = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
            T1 = T1[None, None, :].repeat(1, 1, 1).to(c.device)
            T2 = torch.tensor([math.radians(-x), math.sin(math.radians(-y)), math.cos(math.radians(-y)), z])
            T2 = T2[None, None, :].repeat(1, 1, 1).to(c.device)
            T = torch.cat([T1,T2])
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)]

            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()



def retrieve_text_prompt(fastsamprompt, text, clip_model, preprocess):
    if fastsamprompt.results == None:
        return []
    format_results = fastsamprompt._format_results(fastsamprompt.results[0], 0)
    cropped_boxes, cropped_images, not_crop, filter_id, annotations = fastsamprompt._crop_image(format_results)
    # clip_model, preprocess = clip.load('ViT-B/32', device=self.device)
    scores = fastsamprompt.retrieve(clip_model, preprocess, cropped_boxes, text, device=fastsamprompt.device)
    max_idx = scores.argsort()
    max_idx = max_idx[-1]
    max_idx += sum(np.array(filter_id) <= int(max_idx))
    return np.array([annotations[max_idx]['segmentation']])

def main(args):
    device = torch.device("cuda")

    ## model 
    # FasterSAM
    fastsam_model = FastSAM("/group/40034/ziyangyuan/projects/diffusion/FastSAM/weights/FastSAM-x.pt") # FastSAM/weights/FastSAM-x.pt
    clip_model, preprocess = clip.load('/group/40034/ziyangyuan/projects/diffusion/pretrained_ckpt/clip/ViT-B-32.pt', device=device)    # ViT-B-32.pt
    # image processor
    img_processor = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device="cuda",
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)
    # Zero123 model
    ckpt='/group/40033/public_datasets/3d_datasets/zero123/zero123-xl.ckpt' # zero123-xl.ckpt
    config='/group/40034/ziyangyuan/projects/diffusion/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml' # configs/sd-objaverse-finetune-c_concat-256.yaml
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)
    sampler = DDIMSampler(model)

    ## dataloader
    img_dir = args.source_folder
    dst_dir = args.dst_folder
    img_list = os.listdir(img_dir)
    img_list.sort()





    ##  process loop 
    fail_case = []
    for idx, img_path in enumerate(tqdm(img_list)):
        img_path = os.path.join(img_dir, img_path)
        img_name = os.path.basename(img_path).split(".")[0]
        print("================================current idx: ", idx, img_path)

        img_pil = Image.open(img_path).convert("RGB")
        with open(os.path.join(args.dst_folder, img_name, "object.json"), "r") as f:
            sbj_txt = json.load(f)["object_salient"][0]    # salient
            # sbj_txt = np.random.choice(json.load(f)["objects"])    # random choice one
        with open(os.path.join(args.dst_folder, img_name, "caption.json"), "r") as f:
            prompt = json.load(f)["caption"][0]    # prompt


        ## fastsam, foreground object segment map
        everything_results = fastsam_model(
            img_pil,
            device=device,
            retina_masks=True,
            imgsz=512,
            conf=0.4,
            iou=0.9    
            )
        prompt_process = FastSAMPrompt(img_pil, everything_results, device=device)
        # mask = prompt_process.text_prompt(text=sbj_txt,).astype(np.uint8)
        mask = retrieve_text_prompt(prompt_process, text=sbj_txt, clip_model=clip_model, preprocess=preprocess).astype(np.uint8)
        img_pil_mask = Image.fromarray(np.array(img_pil)*mask[0,:,:,None])
        



        ## zero123
        input_im, bbox = load_and_preprocess(img_processor, img_pil_mask)
        input_im = (input_im / 255.0).astype(np.float32)

        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)      
        input_im = input_im * 2 - 1                         # BCHW, [-1,1]
        input_im = transforms.functional.resize(input_im, [256, 256])

        x = int(np.random.uniform(0, 60))
        y = int(np.random.uniform(0, 60))
        z = 0
    

        x_samples_ddim = sample_model(input_im, model, sampler, precision='fp32', h=256, w=256,
                                        ddim_steps=50, n_samples=1, scale=3.0, ddim_eta=0.0, 
                                        x=x, y=y, z=z)  
        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))



        ## save 
        save_dir = os.path.join(dst_dir, img_name)
        os.makedirs(save_dir, exist_ok=True)      

        img_pil.save(os.path.join(save_dir, "input.jpg"))            ### input

        mask = mask[0,:,:,None]*255
        cv2.imwrite(os.path.join(save_dir, "mask.jpg"), mask)            ### mask

        for i, img in enumerate(output_ims):
            img.save(os.path.join(save_dir, f"multi_view_{i:02d}.jpg"))            ### multi-view
            
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump({
                "sbj_txt": sbj_txt,
                "caption": prompt, 
                "bbox": bbox,
                "view": [
                    [x, y, z],
                    [-x, -y, z],
                    ]
            } ,f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--source_folder', type=str, default="examples/data")
    parser.add_argument("-d", '--dst_folder', type=str, default="examples/dataset")
    args = parser.parse_args()
    main(args)
