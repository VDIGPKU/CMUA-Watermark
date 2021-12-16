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


|  Dataset                    | Model   | $L_{mask }^{2} \uparrow$ | $S R_{\text {mask }} \uparrow$ | FID $\uparrow$ | ACS $\downarrow$ | TFHC $\downarrow$               |
|-----------------------------------|---------|---------------------------------|--------------------------------|----------------|------------------|---------------------------------|
|  CelebA     | StarGAN | $0.20$                          | $100.00 \%$                    | $201.003$      | $0.286$          | $66.26 \% \rightarrow 20.61 \%$ |
|  CelebA     | AGGAN   | $0.13$                          | $99.88 \%$                     | $50.959$       | $0.863$          | $65.88 \% \rightarrow 55.52 \%$ |
|  CelebA     | AttGAN  | $0.05$                          | $87.08 \%$                     | $65.063$       | $0.638$          | $55.13 \% \rightarrow 28.05 \%$ |
|  CelebA     | HiSD    | $0.11$                          | $99.87 \%$                     | $92.734$       | $0.153$          | $63.30 \% \rightarrow 3.94 \%$  |
|  LFW        | StarGAN | $0.20$                          | $100.00 \%$                    | $169.329$      | $0.207$          | $43.88 \% \rightarrow 8.20 \%$  |
|  LFW        | AGGAN   | $0.13$                          | $99.99 \%$                     | $37.746$       | $0.806$          | $54.90 \% \rightarrow 46.32 \%$ |
|  LFW        | AttGAN  | $0.06$                          | $94.07 \%$                     | $70.640$       | $0.496$          | $25.86 \% \rightarrow 16.73 \%$ |
|  LFW        | HiSD    | $0.10$                          | $98.13 \%$                     | $88.145$       | $0.314$          | $50.68 \% \rightarrow 16.03 \%$ |
|  Film100    | StarGAN | $0.20$                          | $100.00 \%$                    | $259.716$      | $0.425$          | $61.01 \% \rightarrow 29.09 \%$ |
|  Film100    | AGGAN   | $0.13$                          | $99.88 \%$                     | $129.099$      | $0.832$          | $60.98 \% \rightarrow 55.69 \%$ |
|  Film100    | AttGAN  | $0.07$                          | $95.82 \%$                     | $177.499$      | $0.627$          | $34.56 \% \rightarrow 25.83 \%$ |
|  Film100    | HiSD    | $0.11$                          | $100.00 \%$                    | $220.689$      | $0.207$          | $67.00 \% \rightarrow 14.00 \%$ |

Table 1. The quantitative results of CMUA-Watermark.
</center>

## Usage

### Installation

1. (option1)install the lib (recommend)

    `
    pip install -r requirements.txt
    `

    
2. (option2) We also prepare conda environment for you: **link**. You can donwload it and unzip in your '/anaconda3/envs/'


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