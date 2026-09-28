# Sources and citations

The 73 benchmark figures come from **45 papers on arXiv**. They are not
included in this repository: `scripts/fetch_golden_figures.py` downloads the
papers from arXiv and extracts the figures with
[pdffigures2](https://github.com/allenai/pdffigures2) at 300 DPI. Copyright in
each figure remains with the paper's authors. If you use the benchmark, please cite the papers below as well;
[`CITATIONS.bib`](CITATIONS.bib) has a BibTeX entry for each.

If you are an author and would like your figure removed from the benchmark, open
an issue and it will be taken down.

**Licence** is the licence the paper carries on arXiv, as recorded in arXiv's
metadata (September 2026):

| Licence on arXiv | Papers |
|---|---|
| arXiv default | 22 |
| CC BY 4.0 | 17 |
| CC BY-NC-ND 4.0 | 4 |
| CC BY-SA 4.0 | 1 |
| CC0 1.0 | 1 |

"arXiv default" is the [arXiv non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html):
it lets arXiv distribute the paper and grants no rights to anyone else.

| Paper | Figures used | Licence on arXiv |
|---|---|---|
| Christian Szegedy et al. (2014). *Going Deeper with Convolutions*. [arXiv:1409.4842](https://arxiv.org/abs/1409.4842v1) | Fig 2, Fig 3 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Olaf Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597v1) | Fig 1 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Joseph Redmon et al. (2015). *You Only Look Once: Unified, Real-Time Object Detection*. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640v5) | Fig 3 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Kaiming He et al. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385v1) | Fig 3, Fig 5 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Sasha Targ et al. (2016). *Resnet in Resnet: Generalizing Residual Architectures*. [arXiv:1603.08029](https://arxiv.org/abs/1603.08029v1) | Fig 1 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Gao Huang et al. (2016). *Densely Connected Convolutional Networks*. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993v5) | Fig 1, Fig 2 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Charles R. Qi et al. (2016). *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*. [arXiv:1612.00593v2](https://arxiv.org/abs/1612.00593v2) | Fig 2 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Andrew G. Howard et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. [arXiv:1704.04861](https://arxiv.org/abs/1704.04861v1) | Fig 3 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Ashish Vaswani et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762v2](https://arxiv.org/abs/1706.03762v2) | Fig 1, Fig 2 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Ozan Oktay et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. [arXiv:1804.03999](https://arxiv.org/abs/1804.03999v3) | Fig 1, Fig 2 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| David Donahue et al. (2019). *Injecting Hierarchy with U-Net Transformers*. [arXiv:1910.10488](https://arxiv.org/abs/1910.10488v2) | Fig 1 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Han Shu and Yunhe Wang (2020). *Automatically Searching for U-Net Image Translator Architecture*. [arXiv:2002.11581](https://arxiv.org/abs/2002.11581v1) | Fig 2, Fig 5 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Gusztáv Gaál et al. (2020). *Attention U-Net Based Adversarial Architectures for Chest X-ray Lung Segmentation*. [arXiv:2003.10304](https://arxiv.org/abs/2003.10304v1) | Fig 2 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Changlu Guo et al. (2020). *Channel Attention Residual U-Net for Retinal Vessel Segmentation*. [arXiv:2004.03702](https://arxiv.org/abs/2004.03702v5) | Fig 1, Fig 2 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Yuncheng Zhou et al. (2020). *Anatomy Prior Based U-net for Pathology Segmentation with Attention*. [arXiv:2011.08769](https://arxiv.org/abs/2011.08769v1) | Fig 2 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Michael Holzmann et al. (2021). *Glacier Calving Front Segmentation Using Attention U-Net*. [arXiv:2101.03247](https://arxiv.org/abs/2101.03247v1) | Fig 1 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Olivier Petit et al. (2021). *U-Net Transformer: Self and Cross Attention for Medical Image Segmentation*. [arXiv:2103.06104](https://arxiv.org/abs/2103.06104v2) | Fig 4 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Yao Chang et al. (2021). *TransClaw U-Net: Claw U-Net with Transformers for Medical Image Segmentation*. [arXiv:2107.05188](https://arxiv.org/abs/2107.05188v1) | Fig 1 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Bingzhi Chen et al. (2021). *TransAttUnet: Multi-level Attention-guided U-Net with Transformer for Medical Image Segmentation*. [arXiv:2107.05274](https://arxiv.org/abs/2107.05274v2) | Fig 2 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Yunxiang Li et al. (2021). *GT U-Net: A U-Net Like Group Transformer Network for Tooth Root Segmentation*. [arXiv:2109.14813](https://arxiv.org/abs/2109.14813v1) | Fig 2 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Jinke Wang et al. (2021). *EAR-U-Net: EfficientNet and attention-based residual U-Net for automatic liver segmentation in CT*. [arXiv:2110.01014](https://arxiv.org/abs/2110.01014v1) | Fig 2, Fig 3, Fig 4, Fig 5 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Syeda Furruka Banu et al. (2021). *AWEU-Net: An Attention-Aware Weight Excitation U-Net for Lung Nodule Segmentation*. [arXiv:2110.05144](https://arxiv.org/abs/2110.05144v1) | Fig 4, Fig 5, Fig 6 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Hongyi Wang et al. (2021). *Mixed Transformer U-Net For Medical Image Segmentation*. [arXiv:2111.04734](https://arxiv.org/abs/2111.04734v2) | Fig 1, Fig 2, Fig 3 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Abhishek Srivastava et al. (2021). *AGA-GAN: Attribute Guided Attention Generative Adversarial Network with U-Net for Face Hallucination*. [arXiv:2111.10591](https://arxiv.org/abs/2111.10591v1) | Fig 1, Fig 2, Fig 3, Fig 4, Fig 5 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Baijun Xie et al. (2022). *DXM-TransFuse U-net: Dual Cross-Modal Transformer Fusion U-net for Automated Nerve Identification*. [arXiv:2202.13304](https://arxiv.org/abs/2202.13304v1) | Fig 3, Fig 4 | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| Reza Azad et al. (2022). *Contextual Attention Network: Transformer Meets U-Net*. [arXiv:2203.01932](https://arxiv.org/abs/2203.01932v2) | Fig 1, Fig 3 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Gongping Chen et al. (2022). *AAU-net: An Adaptive Attention U-net for Breast Lesions Segmentation in Ultrasound Images*. [arXiv:2204.12077](https://arxiv.org/abs/2204.12077v3) | Fig 2 | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| Khay Boon Hong (2022). *U-Net with ResNet Backbone for Garment Landmarking Purpose*. [arXiv:2204.12084](https://arxiv.org/abs/2204.12084v1) | Fig 3 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Jiahao Huang et al. (2022). *Swin Deformable Attention U-Net Transformer (SDAUT) for Explainable Fast MRI*. [arXiv:2207.02390](https://arxiv.org/abs/2207.02390v1) | Fig 1, Fig 2 | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| Deepak Raina et al. (2023). *Slim U-Net: Efficient Anatomical Feature Preserving U-net Architecture for Ultrasound Image Segmentation*. [arXiv:2302.11524](https://arxiv.org/abs/2302.11524v1) | Fig 3 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Nicolas Makaroff and Laurent D. Cohen (2023). *Chan-Vese Attention U-Net: An attention mechanism for robust segmentation*. [arXiv:2306.16098](https://arxiv.org/abs/2306.16098v1) | Fig 1 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Junzhou Chen et al. (2023). *Enhancing Nucleus Segmentation with HARU-Net: A Hybrid Attention Based Residual U-Blocks Network*. [arXiv:2308.03382](https://arxiv.org/abs/2308.03382v2) | Fig 5 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Chenyang Si et al. (2023). *FreeU: Free Lunch in Diffusion U-Net*. [arXiv:2309.11497](https://arxiv.org/abs/2309.11497v2) | Fig 4 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Ahmed Albishri et al. (2023). *OCU-Net: A Novel U-Net Architecture for Enhanced Oral Cancer Segmentation*. [arXiv:2310.02486](https://arxiv.org/abs/2310.02486v1) | Fig 3, Fig 6 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Guanqun Sun et al. (2023). *DA-TransUNet: Integrating Spatial and Channel Dual Attention with Transformer U-Net for Medical Image Segmentation*. [arXiv:2310.12570](https://arxiv.org/abs/2310.12570v2) | Fig 1 | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| Lianghui Zhu et al. (2024). *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model*. [arXiv:2401.09417](https://arxiv.org/abs/2401.09417v3) | Fig 2 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Kassaw Abraham Mulat et al. (2024). *SalFAU-Net: Saliency Fusion Attention U-Net for Salient Object Detection*. [arXiv:2405.02906](https://arxiv.org/abs/2405.02906v1) | Fig 1 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Yuta Kaneko et al. (2024). *Multimodal Attention-Enhanced Feature Fusion-based Weekly Supervised Anomaly Violence Detection*. [arXiv:2409.11223](https://arxiv.org/abs/2409.11223v1) | Fig 1, Fig 2, Fig 3, Fig 4 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Greeshma K and Vishnukumar S (2025). *Efficient and Accurate Tuberculosis Diagnosis: Attention Residual U-Net and Vision Transformer Based Detection Framework*. [arXiv:2501.03538](https://arxiv.org/abs/2501.03538v1) | Fig 2, Fig 3, Fig 4 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Yuchuan Tian et al. (2025). *U-REPA: Aligning Diffusion U-Nets to ViTs*. [arXiv:2503.18414](https://arxiv.org/abs/2503.18414v3) | Fig 1 | [arXiv default](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) |
| Syed Haider Ali et al. (2025). *Automated MRI Tumor Segmentation using hybrid U-Net with Transformer and Efficient Attention*. [arXiv:2506.15562](https://arxiv.org/abs/2506.15562v2) | Fig 4 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Lam Pham et al. (2025). *RMAU-NET: A Residual-Multihead-Attention U-Net Architecture for Landslide Segmentation and Detection from Remote Sensing Images*. [arXiv:2507.11143](https://arxiv.org/abs/2507.11143v1) | Fig 6, Fig 7 | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| Akwasi Asare et al. (2025). *TransUNet-GradCAM: A Hybrid Transformer-U-Net with Self-Attention and Explainable Visualizations for Foot Ulcer Segmentation*. [arXiv:2508.03758](https://arxiv.org/abs/2508.03758v4) | Fig 4 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Swaib Ilias Mazumder et al. (2025). *Learning Regional Monsoon Patterns with a Multimodal Attention U-Net*. [arXiv:2509.23267](https://arxiv.org/abs/2509.23267v1) | Fig 1 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Eyad Gad et al. (2025). *Advancing Brain Tumor Segmentation via Attention-based 3D U-Net Architecture and Digital Image Processing*. [arXiv:2510.19109](https://arxiv.org/abs/2510.19109v1) | Fig 3 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
