# CustomNet


Official implementation of CustomNet: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models.

<div align="center">
<img src="assets/teaser.png" width="600px"/>  
</div>


## Introduce
Incorporating a customized object into image generation presents an attractive feature in text-to-image (T2I) generation. Some methods finetune T2I models for each object individually at test-time, which tend to be overfitted and time-consuming. Others train an extra encoder to extract object visual information for customization efficiently but struggle to preserve the object’s identity. To address these limitations, we present CustomNet, a unified encoder-based object customization framework that explicitly incorporates 3D novel view synthesis capabilities into the customization process. This integration facilitates the adjustment of spatial positions and viewpoints, producing diverse outputs while effectively preserving the object’s identity. To train our model effectively, we propose a dataset construction pipeline to better handle real-world objects and complex backgrounds. Additionally, we introduce delicate designs that enable location control and flexible background control through textual descriptions or user-defined backgrounds. Our method allows for object customization without the need of test-time optimization, providing simultaneous control over viewpoints, location, and text. Experimental results show that our method outperforms other customization methods regarding identity preservation, diversity, and harmony.


<div align="center">
<img src="assets/pipeline.png" width="600px"/>  
</div>


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
