# 2022-snu-fastmri
## By Team *DNA: Deep Neuralnet Accelerated MRI Reconstruction*  
🎉2022 [SNU FastMRI Challenge](https://fastmri.snu.ac.kr/) Participant🎉

### Team Members
| Name | Department | Year | GitHub |
|------|------------|-------------------------------|--------|
|Yunseo Kim|[Nuclear Engineering](https://nucleng.snu.ac.kr/)|sophomore|[@yunseo-kim](https://github.com/yunseo-kim)
|Yongho Lee|[Nuclear Engineering](https://nucleng.snu.ac.kr/)|freshman|[@r3ytv333](https://github.com/r3ytv333)

## About [fastMRI](https://fastmri.org/)
> Accelerating Magnetic Resonance Imaging (MRI) by acquiring fewer measurements has the potential to reduce medical costs, minimize stress to patients and make MR imaging possible in applications where it is currently prohibitively slow or expensive.
>
> fastMRI is a collaborative research project between Facebook AI Research (FAIR) and NYU Langone Health. The aim is to investigate the use of AI to make MRI scans up to 10 times faster.
>
> To enable the broader research community to participate in this important project, NYU Langone Health has released fully anonymized raw data and image datasets.

For more information, see the [fastMRI official website](https://fastmri.org/) & [github repository](https://github.com/facebookresearch/fastMRI/), or the [SNU FastMRI Challenge website](https://fastmri.snu.ac.kr/).

## Package & Data Structure
    \root
    ├── fastMRI                              # Project root (this repository)
    │   ├── Code
    │   │   ├── train-unet.py
    │   │   ├── train-varnet.py
    │   │   ├── train-kiunet.py
    │   │   ├── evaluate-unet.py
    │   │   ├── evaluate-varnet.py
    │   │   ├── leaderboard_eval.py
    │   │   ├── eda.py
    │   │   └── plot.py
    │   ├── utils
    │   │   ├── ...
    │   │   └── model
    │   │       ├── kiunet.py                # KiUnet model
    │   │       ├── unet.py                  # Unet module for the Varnet model
    │   │       ├── unet_legacy.py           # Independent Unet model
    │   │       ├── varnet.py                # Varnet model (best performance)
    │   │       └── ...
    │   ├── result
    │   │   ├── test_KiUnet
    │   │   │   ├── checkpoints              # saved KiUnet models and log files
    │   │   │   │   ├── best_model.pt
    │   │   │   │   ├── model.pt
    │   │   │   │   ├── train-log.txt
    │   │   │   │   └── val-log.txt
    │   │   │   ├── reconstructions_forward  # reconstructed images from the val dataset (not included)
    │   │   │   └── reconstructions_val      # reconstructed images from the leaderboard dataset (not included)
    │   │   ├── test_Unet
    │   │   │   ├── checkpoints              # saved Unet models and log files
    │   │   │   │   ├── best_model.pt
    │   │   │   │   ├── model.pt
    │   │   │   │   ├── train-log.txt
    │   │   │   │   └── val-log.txt
    │   │   │   ├── reconstructions_forward  # reconstructed images from the val dataset (not included)
    │   │   │   └── reconstructions_val      # reconstructed images from the leaderboard dataset (not included)
    │   │   └── test_varnet
    │   │       ├── checkpoints              # saved Varnet models and log files
    │   │       │   ├── best_model.pt        # SSIM score: 0.9698, leaderboard ranking: 30/142
    │   │       │   ├── model.pt
    │   │       │   ├── train-log.txt
    │   │       │   └── val-log.txt
    │   │       ├── reconstructions_forward  # reconstructed images from the val dataset (not included)
    │   │       └── reconstructions_val      # reconstructed images from the leaderboard dataset (not included)
    │   ├── LICENSE
    │   └── README.md
    └── input                                # Dataset files (not included)
        ├── leaderboard
        │   ├── image
        │   └── kspace
        ├── train
        │   ├── image
        │   └── kspace
        └── val
            ├── image
            └── kspace

## Performance
### Best Performance Achieved with the Varnet Model
- SSIM score: 0.9698
- Leaderboard Ranking: 30/142

## References
- Zbontar, J., Knoll, F., Sriram, A.*, Murrell, T., Huang, Z., Muckley, M. J., ... & Lui, Y. W. (2018). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv preprint arXiv:1811.08839.
- Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
- Eo, T., Jun, Y., Kim, T., Jang, J., Lee, H. J., & Hwang, D. (2018). KIKI-net: cross-domain convolutional neural networks for reconstructing undersampled magnetic resonance images. Magnetic resonance in medicine, 80(5), 2188–2201. https://doi.org/10.1002/mrm.27201
- Hammernik, Kerstin & Klatzer, Teresa & Kobler, Erich & Recht, Michael & Sodickson, Daniel & Pock, Thomas & Knoll, Florian. (2018). Learning a Variational Network for Reconstruction of Accelerated MRI Data. Magnetic Resonance in Medicine. 79. https://doi.org/10.1002/mrm.26977. 
- Zhang, P., Wang, F., Xu, W., Li, Y. (2018). Multi-channel Generative Adversarial Network for Parallel Magnetic Resonance Image Reconstruction in K-space. In: Frangi, A., Schnabel, J., Davatzikos, C., Alberola-López, C., Fichtinger, G. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2018. MICCAI 2018. Lecture Notes in Computer Science(), vol 11070. Springer, Cham. https://doi.org/10.1007/978-3-030-00928-1_21
- Sriram, Anuroop & Zbontar, Jure & Murrell, Tullie & Zitnick, C. & Defazio, Aaron & Sodickson, Daniel. (2019). GrappaNet: Combining Parallel Imaging with Deep Learning for Multi-Coil MRI Reconstruction. 
- Sriram, A., Zbontar, J., Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., ... & Johnson, P. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. In MICCAI, pages 64-73.
- Valanarasu, J.M.J., Sindagi, V.A., Hacihaliloglu, I., Patel, V.M. (2020). KiU-Net: Towards Accurate Segmentation of Biomedical Images Using Over-Complete Representations. In: , et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. MICCAI 2020. Lecture Notes in Computer Science(), vol 12264. Springer, Cham. https://doi.org/10.1007/978-3-030-59719-1_36
- Muckley, M. J., Riemenschneider, B., Radmanesh, A., Kim, S., Jeong, G., Ko, J., Jun, Y., Shin, H., Hwang, D., Mostapha, M., Arberet, S., Nickel, D., Ramzi, Z., Ciuciu, P., Starck, J. L., Teuwen, J., Karkalousos, D., Zhang, C., Sriram, A., Huang, Z., … Knoll, F. (2021). Results of the 2020 fastMRI Challenge for Machine Learning MR Image Reconstruction. IEEE transactions on medical imaging, 40(9), 2306–2317. https://doi.org/10.1109/TMI.2021.3075856
- Valanarasu, J.M., Sindagi, V.A., Hacihaliloglu, I., & Patel, V.M. (2022). KiU-Net: Overcomplete Convolutional Architectures for Biomedical Image and Volumetric Segmentation. IEEE Transactions on Medical Imaging, 41, 965-976.
- Ramzi, Zaccharie. (2022). Advanced deep neural networks for MRI image reconstruction from highly undersampled data in challenging acquisition settings. 
