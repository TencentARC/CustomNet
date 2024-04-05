import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append("examples/LAVIS")
from PIL import Image
import json
import argparse
import torch
from lavis.models import load_model_and_preprocess
import time 
import csv
from tqdm import tqdm
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--source_folder', type=str, default="examples/data")
    parser.add_argument("-d", '--dst_folder', type=str, default="examples/dataset")
    args = parser.parse_args()

    
    device = torch.device("cuda")

    ## load BLIP-2 OPT-6.7B  
    print("loading model ...")      
    s = time.time()
    BLIP2_model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
    print(f"Loading model finished. Used TIME: {time.time()-s}")

    img_dir = args.source_folder
    img_list = os.listdir(img_dir)
    img_list.sort()

    os.makedirs(args.dst_folder, exist_ok=True)

    for img_path in tqdm(img_list):
        img_path = os.path.join(img_dir, img_path)
        img_name = os.path.basename(img_path).split(".")[0]
        try:
            raw_image = Image.open(img_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            caption = BLIP2_model.generate({"image": image})
            print(caption, os.path.join(args.dst_folder, img_name))
            os.makedirs(os.path.join(args.dst_folder, img_name), exist_ok=True)
            with open(os.path.join(args.dst_folder, img_name, "caption.json"), "w") as f:
                json.dump({"caption":caption}, f)

        except Exception as e:
            print(e)

