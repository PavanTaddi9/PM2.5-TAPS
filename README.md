# Contrastive Learning for PM2.5 Prediction
[Link to the paper](https://www.sciencedirect.com/science/article/pii/S2666017222000141)

This repository presents a contrastive learning framework for predicting ground-level PM2.5 from high-resolution micro-satellite imagery. Two contrastive learning frameworks, SimCLR and SimSiam, are tested and then extended to formulate a new framework called Spatiotemporal Contrastive Learning (SCL). The satellite imagery and PM2.5 data is obtained from 1 city: Delhi. 


## Structure of the Repository
The structure of this repository is given below:
- `contrastive_learning`: This module contains scripts for unsupervised pre-training with unlabeled satellite images by using regular contrastive learning (SimCLR and SimSiam) frameworks and SCL frameworks.
- `contrastive_models`: This module contains the backbone architecture of original SimCLR and SimSiam frameworks as well as the corresponding data augmentation functions.
- `model_utils`: This module contains all the models and the utility functions for both contrastive and supervised learning tasks.
- `supervised_learning`: This module contains scripts for training and testing the pre-trained model with satellite images and corresponding PM2.5 labels. 

## Related Work
Contrastive Learning
- T. Chen, S. Kornblith, M. Norouzi, et al., "A simple framework for contrastive learning of visual representations," in International conference on machine learning, PMLR, 2020, pp. 1597-1607.
- T. Chen, S. Kornblith, K. Swersky, et al., "Big self-supervised models are strong semi-supervised learners," arXiv preprint arXiv:2006.10029, 2020. https://github.com/google-research/simclr
- X. Chen and K. He, "Exploring simple siamese representation learning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 15750-15758. https://github.com/facebookresearch/simsiam
- W. Falcon et al., Pytorch lightning, https://github.com/PyTorchLightning/pytorch-lightning, 2019.

PM2.5 Prediction from Satellite Imagery
- T. Zheng, M. H. Bergin, S. Hu, et al., "Estimating ground-level pm2.5 using micro-satellite images by a convolutional neural network and random forest approach," Atmospheric Environment, vol. 230, 2020. doi: 10.1016/j.atmosenv.2020.117451.
- T. Zheng, M. Bergin, G. Wang, et al., "Local pm2.5 hotspot detector at 300 m resolution: A random forest-convolutional neural network joint model jointly trained on satellite images and meteorology," Remote Sensing, vol. 13, 2021. doi: 10.3390/rs13071356.


## Citation

```
@article{jiang2022improving,
  title={Improving spatial variation of ground-level PM2. 5 prediction with contrastive learning from satellite imagery},
  author={Jiang, Ziyang and Zheng, Tongshu and Bergin, Mike and Carlson, David},
  journal={Science of Remote Sensing},
  pages={100052},
  year={2022},
  publisher={Elsevier}
}
```