from QAT.data import *
from QAT.model import *
from QAT.quantize import *
import torch
from torch import nn
from torchvision.models import vgg19
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import copy

class GeneratorLoss(nn.Module):
    def __init__(self, mode):
        super(GeneratorLoss, self).__init__()
        self.loss_network = VGG()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        if mode == "pre":
            self.img_to, self.adv_to, self.per_to, self.tv_to = 1, 0, 0, 0
        elif mode == "per":
            self.img_to, self.adv_to, self.per_to, self.tv_to = 0, 0, 1, 0
        elif mode == "gan":
            self.img_to, self.adv_to, self.per_to, self.tv_to = 0, 0.6, 1, 0
        elif mode == "rgan":
            self.img_to, self.adv_to, self.per_to, self.tv_to = 0, 0.6, 1, 0
        elif mode == "full":
            self.img_to, self.adv_to, self.per_to, self.tv_to = 0, 0.6, 1, 2e-8
        print(f"Trade-off params of img, adv, per, tv is: {self.img_to, self.adv_to, self.per_to, self.tv_to}")
        self.mode = mode

    def forward(self, fake_out, real_out, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = nn.BCEWithLogitsLoss()(fake_out, target_real) if self.mode == "gan" \
                        else nn.BCEWithLogitsLoss()(fake_out - real_out, target_real)
        # Perception Loss
        a, b = self.loss_network(out_images, target_images)
        perception_loss = self.mse_loss(a, b)
        # Image Loss
        image_loss = self.mae_loss(out_images, target_images)
        # TV Loss
        tv_loss = TVLoss(out_images)
        return image_loss * self.img_to + adversarial_loss * self.adv_to + perception_loss * self.per_to + tv_loss * self.tv_to


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:35]) #VGG 5_4

        rgb_range = 255
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        return vgg_sr, vgg_hr


def TVLoss(y):
    loss_var = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
               torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    return loss_var

CROP_SIZE = 48
UPSCALE_FACTOR = 4
CUDA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
BATCH_SIZE = 16
LR = 1e-4
MODEL_FILEPATH = "../../pretrain_weight/F2SRGAN/F2SRGAN_4x.pt"

RESUME = 0
MODE = "full"  # MODE is pre/ per/ gan/ rgan/ full

train_path = ["./SR_training_dataset/DIV2K_train_HR", "./SR_training_dataset/Flickr2K"]
test_path = ["./SR_training_dataset/DIV2K_valid_HR"]
RESOLUTION = [144, 360, 480, 720, 1080]
RGB_RANGE = 255.0
REPLICATION_EXP = 10

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(42)

train_set = TrainWholeDataset(train_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValDataset(test_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(
    dataset=train_set,
    num_workers=2,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=1, shuffle=False)

def create_model(upscale_factor):
    netG = Generator(upscale_factor=upscale_factor)
    return netG

def load_model(model, model_filepath, device):
    if (model_filepath != ""):
        model.load_state_dict(torch.load(model_filepath, map_location=device)['model'])
    return model

def train_model(model,
                model_path,
                train_loader,
                val_loader,
                device,
                num_epoches = 200,
                model_quantized = False):
    netG = model.to(device)
    print(
        "# generator parameters:", sum(param.numel() for param in netG.parameters())
    )

    netD = Discriminator().to(device)
    print(
        "# discriminator parameters:", sum(param.numel() for param in netD.parameters())
    )

    generator_criterion = GeneratorLoss(MODE).to(device)
    optimizerG = torch.optim.AdamW(netG.parameters(), lr=LR)
    optimizerG.load_state_dict(torch.load(model_path)['opti'])
    optimizerD = torch.optim.AdamW(netD.parameters(), lr=LR)

    scheduler_G = lr_scheduler.StepLR(optimizerG, step_size=25, gamma=0.5)
    scheduler_D = lr_scheduler.StepLR(optimizerD, step_size=25, gamma=0.5)

    results = {
        "d_loss": [],
        "g_loss": [],
        "psnr": [],
        "ssim": []
    }
    
    if model_quantized:
        netG = unfreeze_model(netG)

    for epoch in range(1, num_epoches + 1):
        scheduler_G.step()
        scheduler_D.step()
        cur_lr = optimizerG.param_groups[0]['lr']

        train_bar = tqdm(train_loader, total=len(train_loader))
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "learning_rate": cur_lr
        }
        
        if model_quantized:
            netG = freeze_model(netG)

        netG.train()
        netD.train()
        for lr_img, hr_img in train_bar:
            batch_size = lr_img.size(0)
            running_results["batch_sizes"] += batch_size
            
            if model_quantized:
                netG = unfreeze_model(netG)
                
            hr_img = hr_img.to(device)
            lr_img = lr_img.to(device)
            target_real = torch.Tensor(batch_size, 1).fill_(1.0).to(device)
            target_fake = torch.Tensor(batch_size, 1).fill_(0.0).to(device)
            ############################
            # (1) Update D network
            ###########################
            if MODE != "pre":
                sr_img = netG(lr_img)

                netD.zero_grad()
                real_out = netD(hr_img)
                fake_out = netD(sr_img)

                d_loss = nn.BCEWithLogitsLoss()(real_out, target_real) + nn.BCEWithLogitsLoss()(fake_out, target_fake) if MODE == "gan" \
                        else nn.BCEWithLogitsLoss()(real_out - fake_out, target_real)
                d_loss.backward(retain_graph=True)
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()

            sr_img = netG(lr_img)
            fake_out = netD(sr_img)
            real_out = netD(hr_img)

            g_loss = generator_criterion(fake_out, real_out, sr_img, hr_img, target_real)
            # g_loss = generator_criterion(fake_out, sr_img, hr_img)
            g_loss.backward()

            optimizerG.step()

            # loss for current after before optimization
            running_results["g_loss"] += g_loss.item() * batch_size
            if MODE != "pre":
                running_results["d_loss"] += d_loss.item() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss_D: %f Loss_G: %f Learning_rate: %f"
                % (
                    epoch,
                    num_epoches,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["learning_rate"]
                )
            )
            
            if model_quantized:
                netG = freeze_model(netG)

        netG.eval()

        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader))
            valing_results = {
                "mse": 0,
                "ssims": 0,
                "psnr": 0,
                "ssim": 0,
                "batch_sizes": 0,
            }
            val_images = []
            for val_lr, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results["batch_sizes"] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                # Forward
                sr = netG(lr)
                # Loss & metrics
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size

                valing_results["ssims"] += 0
                valing_results["psnr"] = 10 * math.log10(
                    (hr.max() ** 2)
                    / (valing_results["mse"] / valing_results["batch_sizes"])
                )
                valing_results["ssim"] = (
                    valing_results["ssims"] / valing_results["batch_sizes"]
                )
                val_bar.set_description(
                    desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                    % (valing_results["psnr"], valing_results["ssim"])
                )

        # save model parameters
        netG.train()
        netD.train()
        
        #########################
        torch.save(
            {"model": netG.state_dict()},
            f"./netG_{UPSCALE_FACTOR}x_epoch{epoch}.pth.tar",
        )
        #########################

        results["d_loss"].append(
            running_results["d_loss"] / running_results["batch_sizes"]
        )
        results["g_loss"].append(
            running_results["g_loss"] / running_results["batch_sizes"]
        )

        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

## Run model before QAT
buffer_model = copy.deepcopy(Generator(upscale_factor=UPSCALE_FACTOR))
print(buffer_model)
model_before_QAT = load_model(model = buffer_model,
                              model_filepath = MODEL_FILEPATH,
                              device = CUDA_DEVICE)
model_before_QAT.cuda().eval()

## Apply quantization
NUM_BIT_QUANTIZED = 16
NUM_EPOCHS_FINETUNE = 1

PRECISION = torch.float if (NUM_BIT_QUANTIZED == 32) else {torch.half, torch.float}
POSTFIX = "float32" if (NUM_BIT_QUANTIZED == 32) else "fp16"
        
MODEL_NAME = "F2SRGAN"
FILE_QAT_NAME = f"trained_{MODEL_NAME}_qat_{POSTFIX}"

model_quant = quantize_model(model_before_QAT, NUM_BIT_QUANTIZED, NUM_BIT_QUANTIZED, full_precision_flag=False).to(CUDA_DEVICE)

## QAT
train_model(model=model_quant,
            model_path=MODEL_FILEPATH,
            train_loader=train_loader,
            val_loader=val_loader,
            device=CUDA_DEVICE,
            num_epoches=NUM_EPOCHS_FINETUNE,
            model_quantized = True)