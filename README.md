# MGID
This repository contains the Pytorch implementation code of the paper "Memory-guided Image De-raining Using Time-Lapse Data." [[Paper]](https://arxiv.org/abs/2201.01883)


<!-- add figure figures/MGID.jpg -->
<p align="center">
  <img src="figures/jpg.png" width="600">


## Abstract

This paper addresses the problem of single image de-raining, that is, the task of recovering clean and rain-free background scenes from a single image obscured by a rainy artifact. Although recent advances adopt real-world time-lapse data to overcome the need for paired rain-clean images, they are limited to fully exploit the time-lapse data. The main cause is that, in terms of network architectures, they could not capture long-term rain streak information in the time-lapse data during training owing to the lack of memory components. To address this problem, we propose a novel network architecture based on a memory network that explicitly helps to capture long-term rain streak information in the time-lapse data. Our network comprises the encoder-decoder networks and a memory network. The features extracted from the encoder are read and updated in the memory network that contains several memory items to store rain streak-aware feature representations. With the read/update operation, the memory network retrieves relevant memory items in terms of the queries, enabling the memory items to represent the various rain streaks included in the time-lapse data. To boost the discriminative power of memory features, we also present a novel background selective whitening (BSW) loss for capturing only rain streak information in the memory network by erasing the background information. Experimental results on standard benchmarks demonstrate the effectiveness and superiority of our approach.



## Dataset

- [TimeLapse Data](https://drive.google.com/file/d/1scs_LN4Rk6M0VEzYYnCPWTfuHISd_8f-/view?usp=drive_link)



## Citation

If MGID is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@article{cho2022memory,
  title={Memory-Guided Image De-Raining Using Time-Lapse Data},
  author={Cho, Jaehoon and Kim, Seungryong and Sohn, Kwanghoon},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4090--4103},
  year={2022},
  publisher={IEEE}
}
```


## Acknowledgement

The work benefits from outstanding prior work and their implementations including:
- [Learning Memory-guided Normality for Anomaly Detection](https://github.com/cvlab-yonsei/MNAD) by Park et al. CVPR 2020.
- [RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening](https://github.com/shachoi/RobustNet) by Choi et al. CVPR 2021.
