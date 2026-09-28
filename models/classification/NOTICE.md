# resnet50_vanilla.pth: notices

`scripts/fetch_assets.sh` downloads `resnet50_vanilla.pth` into this folder from
the [`assets-v1` release](https://github.com/DavidHir0/kg-extraction/releases/tag/assets-v1).
It is used only by `main.py classify`, which sorts extracted figures into the 18
classes in `classes.txt`.

## What it is built from

- **Architecture:** ResNet-50, as implemented in
  [torchvision](https://github.com/pytorch/vision) (BSD-3-Clause).
  > Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. 2016. Deep Residual
  > Learning for Image Recognition. In *CVPR 2016*, 770–778.
  > <https://arxiv.org/abs/1512.03385>
- **Starting weights:** torchvision's `ResNet50_Weights.IMAGENET1K_V2`, trained on
  ImageNet-1K. torchvision notes that its pre-trained weights "may have their
  own licenses or terms and conditions derived from the dataset used for
  training" ([torchvision docs](https://docs.pytorch.org/vision/stable/models.html)).
- **Fine-tuning data:** the ACL-Fig dataset.
  > Zeba Karishma, Shaurya Rohatgi, Kavya Shrinivas Puranik, Jian Wu and C. Lee
  > Giles. 2023. ACL-Fig: A Dataset for Scientific Figure Classification. AAAI-23
  > Workshop on Scientific Document Understanding.
  > <https://arxiv.org/abs/2301.12293>

  The dataset's licence is stated differently in two places: the paper says
  **CC BY-NC**, and the
  [dataset page on Hugging Face](https://huggingface.co/datasets/citeseerx/ACL-fig)
  says **CC BY 4.0**. Until that is resolved, treat the checkpoint as
  non-commercial.
