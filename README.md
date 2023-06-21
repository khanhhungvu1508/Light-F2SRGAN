# Light-F2SRGAN
This is an official implementation of paper "Investigation into Perceptual-Aware Optimization for Single-Image Super-Resolution in Embedded Systems".

# Data prepare
You should organize the images layout like this:

```shell
SR_training_dataset
├── DIV2K_train_HR # Include 800 train images
├── Flickr2K
└── DIV2K_valid_HR # Include 100 test images

SR_testing_dataset
├── Set5
├── Set14
├── BSDS100
└── Urban100
```

# Train
There are 5 mode of training:
- **pre**: L1 loss only.
- **per**: Perceptual loss only.
- **gan**: Perceptual loss and gan loss.
- **rgan**: Perceptual loss and rgan loss.
- **full**: The proposed loss in paper.  

To replicate the paper's result, first train model with `pre` mode, followed by `full` mode.

# Pretrain weight structure
```shell
pretrain_weight
├── F2SRGAN_x2.pt           # Pretrain for model F2SRGAN with scale x2
├── F2SRGAN_x4.pt           # Pretrain for model F2SRGAN with scale x4
├── F2SRGAN_x4_QAT.pth.tar  # Pretrain for model F2SRGAN QAT with scale x4
├── F2SRGAN_x4_student.pt   # Pretrain for model F2SRGAN KD with AT method with scale x4
└── LightF2SRGAN_x4.pt      # Pretrain for model light-F2SRGAN with scale x4
```

# Evaluate
- If you want to generate images for Light-F2SRGAN for each optimization method, run `generate_image.py` to produce output images.
- If you want to measure times running some optimization method, run `inference_time.py` to test.
- To measure the Perceptual Index, please refer to this [Repository](https://github.com/roimehrez/PIRM2018) for more information

# Results
**PI results of optimization method**

![PI results of optimization method](https://github.com/khanhhungvu1508/Light-F2SRGAN/assets/69689114/a1a993d3-05f6-4f3b-8f55-7953bc10ea47)

**Time inference in Desktop and Jetson Xavier NX**

![Time inference in Desktop and Jetson Xavier NX](https://github.com/khanhhungvu1508/Light-F2SRGAN/assets/69689114/5756f1a0-a528-4a5c-95c5-67f3be19fc70)


# Citation
```
@article{vu2023investigation,
  title={Investigation into Perceptual-Aware Optimization for Single-Image Super-Resolution in Embedded Systems},
  author={Vu, Khanh Hung and Nguyen, Duc Phuc and Nguyen, Duc Dung and Pham, Hoang-Anh},
  journal={Electronics},
  volume={12},
  number={11},
  pages={2544},
  year={2023},
  publisher={MDPI}
}
```

# References
- [F2SRGAN](https://github.com/bibom108/F2SRGAN)
- [SwiftSRGAN](https://github.com/Koushik0901/Swift-SRGAN)
- [LaMa](https://github.com/advimman/lama)
