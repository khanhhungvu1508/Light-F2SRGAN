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

# Test
Pretrained weight for `x2` and `x4` upscale factor are provided in `pretrain_weight`. Code for load model F2SRGAN:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE_FACTOR = 4 #Setup scale factor
MODEL_PATH = f"./pretrain_weight/F2SRGAN_{SCALE_FACTOR}x.pt"
netG = Generator(upscale_factor = SCALE_FACTOR)
netG.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model'])
model.eval()
```

# Evaluate
- If you want to generate images for Light-F2SRGAN for each optimization method, run `generate_image.py` to produce output images.
- If you want to measure times running some optimization method, run `inference_time.py` to test.
- To measure the Perceptual Index, please refer to this [Repository](https://github.com/roimehrez/PIRM2018) for more information

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
- [SwiftSRGAN](https://github.com/Koushik0901/Swift-SRGAN)
- [LaMa](https://github.com/advimman/lama)
