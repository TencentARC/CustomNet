import os
import numpy as np
import torch
import cv2
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from tqdm import tqdm
import argparse
from ldm.models.diffusion.ddim import DDIMSampler
from carvekit.api.high import HiInterface
import json
from contextlib import nullcontext
from torchvision import transforms
import math
from einops import rearrange
import matplotlib.pyplot as plt




class Dataset():
    def __init__(self,
                root="examples/dataset/",
                image_size=256,
                ):

        self.root = root
        self.data = os.listdir(root)
    
        print("Total Image Nums: ", len(self.data))
        self.image_size = image_size

        image_transforms = [transforms.Resize((image_size, image_size))]
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.tform = transforms.Compose(image_transforms)


    def get_data(self, idx):
        img_dir = os.path.join(self.root, self.data[idx])

        ## target img
        target_img_path = os.path.join(img_dir, "input.jpg")
        target_im_pil = Image.open(target_img_path) 
        W, H = target_im_pil.size
        target_im = self.process_im(target_im_pil)

        ## meta
        meta = os.path.join(img_dir, "meta.json")
        with open(os.path.join(img_dir, "meta.json")) as f:
            meta = json.load(f)
        

        ## bbox 
        x0,y0,w,h = meta["bbox"] 
        # x1 = x0 + w
        # y1 = y0 + h
        x0,y0,w,h = int(255*x0/W), int(255*y0/H), int(255*w/W), int(255*h/H)
        
        x00 = x0
        y00 = y0
        x11 = x0 + w
        y11 = y0 + h

        square_l = int(max(w,h)*256/200)
        x0 = x0 - (square_l - w)//2
        w = w + (square_l - w)
        y0 = y0 - (square_l - h)//2
        h = h + (square_l - h)

        x1 = x0 + w
        y1 = y0 + h

        xx0, yy0, xx1, yy1 = 0, 0, 256, 256  ## condtion location
        if x0 < 0:
            xx0 = xx0 + int(-x0*256/square_l)
            x0 = 0
            
        if y0 < 0:
            yy0 = yy0 + int(-y0*256/square_l)
            y0 = 0

        if x1 > 256:
            xx1 = 256 - int((x1-256)*256/square_l)
            x1 = 256 

        if y1 > 256:
            yy1 = 256 - int((y1-256)*256/square_l) 
            y1 = 256 



        ## camera
        camera_list = meta["view"]
        i = np.random.randint(len(camera_list))
        x, y, z = camera_list[i]


        ## cond_img
        cond_im_pil = Image.open(os.path.join(img_dir, f"multi_view_{i:02d}.jpg")) 
        cond_im = self.process_im(cond_im_pil)


        ## caption
        caption = meta["caption"]

        ## image_cond_concat
        try:
            cond_im_pad = torch.ones(256, 256, 3)
            cond_im_cat = cond_im.permute(2,0,1).unsqueeze(0)[:,:,yy0:yy1,xx0:xx1]
            cond_im_cat = torch.nn.functional.interpolate(cond_im_cat, (y1-y0, x1-x0), mode="bilinear")[0].permute(1,2,0)
            cond_im_pad[y0:y1,x0:x1,:] = cond_im_cat
            cond_im_cat = cond_im_pad
        except:
            cond_im_cat = cond_im

        data ={}
        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["image_cond_concat"] = cond_im_cat
        data["T"] = torch.tensor([math.radians(-x), math.sin(math.radians(-y)), math.cos(math.radians(-y)), -z])
        data["txt"] = caption
        # data["bbox_mask"] = bbox_mask

        return data 
    def __getitem__(self, idx):

        try:
            data = self.get_data(idx)
        except:
            data = self.get_data(0)
        
        return data


    def __len__(self, ):
        return len(self.data)

    def process_im(self, im):
        # im = im.convert("RGB")
        return self.tform(im)