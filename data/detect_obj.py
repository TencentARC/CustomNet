import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append("examples/LAVIS")
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import json
import argparse
import time 
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--source_folder', type=str, default="examples/data")
    parser.add_argument("-d", '--dst_folder', type=str, default="examples/dataset")
    args = parser.parse_args()

    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # loads BLIP-2 pre-trained model
    print("loading model ...")      
    s = time.time()
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    model = model.float()
    print(f"Loading model finished. Used TIME: {time.time()-s}")

    # model = model.half()

    img_dir = args.source_folder
    img_list = os.listdir(img_dir)
    img_list.sort()

    for img_path in tqdm(img_list):
        img_path = os.path.join(img_dir, img_path)
        img_name = os.path.basename(img_path).split(".")[0]
        try:
            res = {}
            raw_image = Image.open(img_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            reply = model.generate({"image": image, 
                                "prompt": "Question: What foreground objects are in the image? find them and separate them using commas. Answer:"})
            res["objects"] = reply
            reply = model.generate({"image": image, 
            "prompt": f"Question: What foreground objects are in the image? find them and separate them using commas.  Answer:{reply}. Question: Which object is the most salient?"
            })
            res["object_salient"] = reply

            with open(os.path.join(args.dst_folder, img_name, "object.json"), "w") as f:
                json.dump(res, f)


        except Exception as e:
            print(e)
