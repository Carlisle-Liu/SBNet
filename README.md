# Learning Structure-Aware Semantic Segmentation with Image-Level Supervision

The implementation of **Learning Structure-Aware Semantic Segmentation with Image-Level Supervision**, Jiawei Liu, Jing Zhang, Yicong Hong and Nick Barnes, IJCNN 2021 [Paper](https://arxiv.org/abs/2104.07216).

## Abstract
Compared with expensive pixel-wise annotations, image-level labels make it possible to learn semantic segmentation in a weakly-supervised manner. Within this pipeline, the class activation map (CAM) is obtained and further processed to serve as a pseudo label to train the semantic segmentation model in a fully-supervised manner. In this paper, we argue that the lost structure information in CAM limits its application in downstream semantic segmentation, leading to deteriorated predictions. Furthermore, the inconsistent class activation scores inside the same object contradicts the common sense that each region of the same object should belong to the same semantic category. To produce sharp prediction with structure information, we introduce an auxiliary semantic boundary detection module, which penalizes the deteriorated predictions. Furthermore, we adopt smoothness loss to encourage prediction inside the object to be consistent. Experimental results on the PASCAL-VOC dataset illustrate the effectiveness of the proposed solution.

Thanks to the work of [Yude Wang](https://github.com/YudeWang) and [jiwoon-ahn](https://github.com/jiwoon-ahn), the code of this repository borrow heavly from their [SEAM]((https://github.com/YudeWang/SEAM)) and [AffinityNet](https://github.com/jiwoon-ahn/psa) repositories, and we follw the same pipeline to verify the effectiveness of our solution.

## Requirements
- Python 3.6
- pytorch 0.4.1, torchvision 0.2.1
- CUDA 9.0
- 4 x GPUs (12GB)
