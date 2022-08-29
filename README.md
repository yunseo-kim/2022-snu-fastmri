# 2022-snu-fastmri
## By Team *DNA: Deep Neuralnet Accelerated MRI Reconstruction*  
🎉2022 [SNU FastMRI Challenge](https://fastmri.snu.ac.kr/) Participant🎉

### Team Members
| Name | Department | Year | GitHub |
|------|------------|-------------------------------|--------|
|Yunseo Kim|[Nuclear Engineering](https://nucleng.snu.ac.kr/)|sophomore|[@yunseo-qleap](https://github.com/yunseo-qleap)
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
    │   └── result
    │       ├── test_KiUnet
    │       │   ├── checkpoints              # saved KiUnet models and log files
    │       │   │   ├── best_model.pt
    │       │   │   ├── model.pt
    │       │   │   ├── train-log.txt
    │       │   │   └── val-log.txt
    │       │   ├── reconstructions_forward  # reconstructed images from the val dataset (not included)
    │       │   └── reconstructions_val      # reconstructed images from the leaderboard dataset (not included)
    │       ├── test_Unet
    │       │   ├── checkpoints              # saved Unet models and log files
    │       │   │   ├── best_model.pt
    │       │   │   ├── model.pt
    │       │   │   ├── train-log.txt
    │       │   │   └── val-log.txt
    │       │   ├── reconstructions_forward  # reconstructed images from the val dataset (not included)
    │       │   └── reconstructions_val      # reconstructed images from the leaderboard dataset (not included)
    │       └── test_varnet
    │           ├── checkpoints              # saved Varnet models and log files
    │           │   ├── best_model.pt        # SSIM score: 0.9698, leaderboard ranking: 30/142
    │           │   ├── model.pt
    │           │   ├── train-log.txt
    │           │   └── val-log.txt
    │           ├── reconstructions_forward  # reconstructed images from the val dataset (not included)
    │           └── reconstructions_val      # reconstructed images from the leaderboard dataset (not included)
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
