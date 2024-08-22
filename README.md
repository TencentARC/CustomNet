# CustomNet: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models.


🤗 [![HF Demo](https://img.shields.io/static/v1?label=Demo&message=CustomNet&color=orange)](https://huggingface.co/spaces/TencentARC/CustomNet) &ensp; 
[![arXiv](https://img.shields.io/badge/arXiv-red)](https://arxiv.org/abs/2310.19784) &ensp;
[![Project Page](https://img.shields.io/badge/Project%20Page-green)](https://jiangyzy.github.io/CustomNet/)

---
Official implementation of CustomNet: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models. ( ACM MM 2024）

<div align="center">
<img src="assets/teaser.png" width="600px"/>  
</div>


## Introduce
Incorporating a customized object into image generation presents an attractive feature in text-to-image (T2I) generation. Some methods finetune T2I models for each object individually at test-time, which tend to be overfitted and time-consuming. Others train an extra encoder to extract object visual information for customization efficiently but struggle to preserve the object’s identity. To address these limitations, we present CustomNet, a unified encoder-based object customization framework that explicitly incorporates 3D novel view synthesis capabilities into the customization process. This integration facilitates the adjustment of spatial positions and viewpoints, producing diverse outputs while effectively preserving the object’s identity. To train our model effectively, we propose a dataset construction pipeline to better handle real-world objects and complex backgrounds. Additionally, we introduce delicate designs that enable location control and flexible background control through textual descriptions or user-defined backgrounds. Our method allows for object customization without the need of test-time optimization, providing simultaneous control over viewpoints, location, and text. Experimental results show that our method outperforms other customization methods regarding identity preservation, diversity, and harmony.


<div align="center">
<img src="assets/pipeline.png" width="600px"/>  
</div>

---

## ⚙️ Environment
    conda create -n customnet python=3.10 -y
    conda activate customnet
    pip install -r requirements.txt

## 💫 Inference

### Run local gradio demo
- Download the weights of Customnet [customnet_v1.pth](https://huggingface.co/TencentARC/CustomNet/tree/main) and put it to `./pretrain`.

- Running scripts:

        sh scripts/run_app.sh

## 🔥Train
### Prepare dataset
- We provide example data in `examples/data` containg images only.
- Clone extra repo in [extralibs](extralibs), and prepare environments.

        cd extralibs
        git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
        git clone https://github.com/salesforce/LAVIS.git
        git clone https://github.com/cvlab-columbia/zero123.git
- Use the script to create datasets:
        
        sh scripts/process_data.sh
### Train
- Check out [config file](configs/config_customnet.yaml), and update related paths.
- Use the script to train:
        
        sh scripts/train.sh

## BibTeX
```
@misc{yuan2023customnet,
    title={CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models}, 
    author={Ziyang Yuan and Mingdeng Cao and Xintao Wang and Zhongang Qi and Chun Yuan and Ying Shan},
    year={2023},
    eprint={2310.19784},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
