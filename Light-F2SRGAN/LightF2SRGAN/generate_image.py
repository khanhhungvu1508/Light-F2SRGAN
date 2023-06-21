from LightF2SRGAN.quantize import *
import os
from PIL import Image
from torchvision import transforms
import torch
import math
import numpy as np
import imageio
import cv2

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calc_psnr(sr, hr, scale, rgb_range):
    """calculate psnr"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)
    if hr.size == 1:
        return 0
    if scale != 1:
        shave = scale
    else:
        shave = scale + 6
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
    mse = np.mean(pow(valid, 2))
    return -10 * math.log10(mse)

def calc_ssim(img1, img2, scale):
    """calculate ssim value"""
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    border = 0
    if scale != 1:
        border = scale
    else:
        border = scale + 6
    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]
    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def tensors_to_imgs(x):
    for i in range(len(x)):
        x[i] = x[i].squeeze(0).data.cpu().numpy()
        x[i] = x[i].clip(0, 255)#.round()
        x[i] = x[i].transpose(1, 2, 0).astype(np.uint8)
    return x

def imgs_to_tensors(x):
    for i in range(len(x)):
        x[i] = x[i].transpose(2, 0, 1)
        x[i] = np.expand_dims(x[i], axis=0)
        x[i] = torch.Tensor(x[i].astype(float))
    return x


def rgb2y(rgb):
    return np.dot(rgb[...,:3], [65.738/256, 129.057/256, 25.064/256]) + 16


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def Tensor2img(v):
    normalized = v[0].data.mul(255 / 255)
    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    return ndarr

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def lightweight_F2SRGAN_QAT_psnr_testing(_path, model, ds):
    netG = model.to(DEVICE)
    netG.eval()
    nums = 0
    mean_psnr = 0
    mean_ssim = 0

    if (NUM_BIT_QUANTIZED == 16):
        netG = netG.half()

    with torch.no_grad():
        for name in os.listdir(_path):
            full_path = _path + "/" + name
            hr_image = Image.open(full_path).convert('RGB')
            image_width = (hr_image.width // UPSCALE_FACTOR) * UPSCALE_FACTOR
            image_height = (hr_image.height // UPSCALE_FACTOR) * UPSCALE_FACTOR
            hr_scale = transforms.Resize((image_height, image_width), interpolation=Image.BICUBIC)
            lr_scale = transforms.Resize((image_height // UPSCALE_FACTOR, image_width // UPSCALE_FACTOR), interpolation=Image.BICUBIC)
            lr_image = lr_scale(hr_image)
            hr_image = hr_scale(hr_image)
            lr_image = np.asarray(lr_image)
            hr_image = np.asarray(hr_image)
            
            [lr_image] = imgs_to_tensors([lr_image])
            [hr_image] = imgs_to_tensors([hr_image])
            
            if (NUM_BIT_QUANTIZED == 16):
                out = netG(lr_image.to(DEVICE).half())
            else:
                out = netG(lr_image.to(DEVICE))
            mean_psnr += calc_psnr (out.to(torch.device("cpu")), hr_image.to(torch.device("cpu")), 
                                    scale = UPSCALE_FACTOR, 
                                    rgb_range = RGB_RANGE)
            mean_ssim += calc_ssim (out[0].to(torch.device("cpu")).permute(1, 2, 0).numpy(), 
                                    hr_image[0].to(torch.device("cpu")).permute(1, 2, 0).numpy(), 
                                    scale = UPSCALE_FACTOR)

            [out] = tensors_to_imgs([out])
            [hr_image] = tensors_to_imgs([hr_image])
            
            file_output = "sample_lightweight_F2SRGAN"
            output_folder = os.path.join(file_output + f"_x{UPSCALE_FACTOR}",
                                        ds)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            imageio.imwrite('{}/{}'.format(output_folder, name), out)
            nums += 1
    
    return mean_psnr/nums, mean_ssim/nums

NUM_BIT_QUANTIZED = 16
POSTFIX = "float32" if (NUM_BIT_QUANTIZED == 32) else "fp16" if (NUM_BIT_QUANTIZED == 16) else "int8"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPSCALE_FACTOR = 4
RGB_RANGE = 255
RESOLUTION = [144, 360, 480, 720, 1080]
RGB_RANGE = 255.0
REPLICATION_EXP = 10

PRECISION = torch.float if (NUM_BIT_QUANTIZED == 32) else {torch.half, torch.float}
POSTFIX = "float32" if (NUM_BIT_QUANTIZED == 32) else "fp16"
        
MODEL_NAME = "F2SRGAN"
MODEL_FILEPATH_QAT = "../../pretrain_weight/LightF2SRGAN_x4.pt"

model_quant = torch.load(MODEL_FILEPATH_QAT)['deploy']
model_quant.load_state_dict(torch.load(MODEL_FILEPATH_QAT)['model'])
model_quant.cuda().eval()

dataset = [
    "Set5",
    "Set14",
    "BSDS100",
    "Urban100"
]

for ds in dataset:
    psnr, ssim_val = lightweight_F2SRGAN_QAT_psnr_testing("../../SR_testing_datasets/" + ds,
                                                         model_quant,
                                                         ds)
    print(f'PSNR in {ds} = {psnr}')
    print(f'SSIM in {ds} = {ssim_val}')
