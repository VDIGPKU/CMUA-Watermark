# CMUA-Watermark

The official code for CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes (AAAI2022) [arxiv](https://arxiv.org/abs/2105.10872).  It is based on [disrupting-deepfakes
](https://github.com/natanielruiz/disrupting-deepfakes). 

Contact us with huanghao@stu.pku.edu.cn, wyt@pku.edu.cn.

We will release our code soon (no later than December 31, 2021).

## Introduction

CMUA-Watermark is a **cross-model universal adversarial watermark** that can combat multiple deepfake models while protecting a myriad of facial images. With the proposed perturbation fusion strategies and automatic step size tuning, CMUA-Watermark achieves excellent protection capabilities for facial images against four face modification models (StarGAN, AttGAN, AGGAN, HiSD).

<center>
<img src="./imgs/1.png">
Figure 1. Illustration of our CMUA-Watermark. Once the CMUA-watermark has been generated, we can add it directly to any facial image to generate a protected image that is visually identical to the original image but can distort outputs of deepfake models.
</center>

<center>

<img src="./imgs/2.png" height=400>

Figure 2. The quantitative results of CMUA-Watermark.
</center>

## Usage

### Installation

1. (option1) install the lib by pip (recommend)

    `
    pip install -r requirements.txt
    `

    
2. (option2) We also prepare conda environment for you (if you use CUDA 10.0 and conda 4.8.3) : https://disk.pku.edu.cn:443/link/D613E493EE641184EB52C0C78DD846C8 . You can donwload it and unzip in your '/anaconda3/envs/'. 

### Inference

### Training （attacking multiple deepfake models）

**Will be added soon～**

## Citation

```
@misc{huang2021cmuawatermark,
      title={CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes}, 
      author={Hao Huang and Yongtao Wang and Zhaoyu Chen and Yuze Zhang and Yuheng Li and Zhi Tang and Wei Chu and Jingdong Chen and Weisi Lin and Kai-Kuang Ma},
      year={2021},
      eprint={2105.10872},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.