from KD.data import *
from KD.model import *
import torch
from torch import nn
from torchvision.models import vgg19
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

class GeneratorLoss(nn.Module):
    def __init__(self, mode):
        super(GeneratorLoss, self).__init__()
        self.loss_network = VGG()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        self.ce = nn.CrossEntropyLoss()
        self.kl = DistillKL(4)
        self.kd = AFD(args)

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
        self.ce_to = 1
        self.kl_to = 0.9
        self.kd_to = 200
        print(f"Trade-off params of img, adv, per, tv is: {self.img_to, self.adv_to, self.per_to, self.tv_to}")
        self.mode = mode

    def forward(self, fake_out, real_out, out_imagesT, out_images, target_images, f_t, f_s):
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
        # CE loss
        ce_loss = self.ce(out_images, target_images)
        # DistillKL loss
        kl_loss = self.kl(out_images, out_imagesT)
        # AFD loss
        kd_loss = self.kd(f_s, f_t)
#         return image_loss
        return image_loss * self.img_to + \
                adversarial_loss * self.adv_to + perception_loss * self.per_to + tv_loss * self.tv_to + \
                ce_loss * self.ce_to + kl_loss * self.kl_to + kd_loss * self.kd_to


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


def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(42)

CROP_SIZE = 48
UPSCALE_FACTOR = 4
NUM_EPOCHS = 46
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LR = 5e-5

RESUME = 0
MODE = "full"  # MODE is pre/ per/ gan/ rgan/ full

train_path = ["../input/df2k-ost/train/DIV2K/DIV2K_train_HR", "/kaggle/input/df2k-ost/train/Flickr2K"]
test_path = ["/kaggle/input/super-resolution-benchmarks/Set5/Set5/original"]

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

netGT = Generator_T(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
print("# generator parameters:", sum(param.numel() for param in netG.parameters()))
netG = Generator_S(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
print("# generator parameters:", sum(param.numel() for param in netG.parameters()))
netD = Discriminator().to(DEVICE)
print("# discriminator parameters:", sum(param.numel() for param in netD.parameters()))

##########################
data = torch.randn(2, 3, CROP_SIZE, CROP_SIZE).to(DEVICE)
netGT.eval()
netG.eval()
with torch.no_grad():
    _, feat_t = netGT(data, is_feat=True)
    _, feat_s = netG(data, is_feat=True)
args = {
    "qk_dim":128,
    "guide_layers":4,
    "hint_layers":4
}
args['t_shapes'] = [feat_t[i].size() for i in range(args['guide_layers'])]
args['s_shapes'] = [feat_s[i].size() for i in range(args['hint_layers'])]
args['n_t'], args['unique_t_shapes'] = unique_shape(args['t_shapes'])
##########################

generator_criterion = GeneratorLoss(MODE).to(DEVICE)
optimizerG = torch.optim.AdamW(netG.parameters(), lr=LR)
optimizerD = torch.optim.AdamW(netD.parameters(), lr=LR)

if MODE != "pre" and RESUME == 0:
    print(f"Loading from PRE")
    netGT.load_state_dict(torch.load(f"netG_{UPSCALE_FACTOR}x_epoch46.pt")['model'])
    # netG.load_state_dict(torch.load(f"netG_{UPSCALE_FACTOR}x_epoch46.pt")['model'])
#     optimizerG.load_state_dict(torch.load(f"netG_{UPSCALE_FACTOR}x_epoch100.pt")['opti'])

scheduler_G = lr_scheduler.StepLR(optimizerG, step_size=25, gamma=0.5)
scheduler_D = lr_scheduler.StepLR(optimizerD, step_size=25, gamma=0.5)

results = {
    "d_loss": [],
    "g_loss": [],
    "psnr": [],
    "ssim": []
}

for epoch in range(max(RESUME, 1), NUM_EPOCHS + 1):
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
    netGT.eval()
    netG.train()
    netD.train()
    for lr_img, hr_img in train_bar:
        batch_size = lr_img.size(0)
        running_results["batch_sizes"] += batch_size
        hr_img = hr_img.to(DEVICE)
        lr_img = lr_img.to(DEVICE)
        target_real = torch.Tensor(batch_size, 1).fill_(1.0).to(DEVICE)
        target_fake = torch.Tensor(batch_size, 1).fill_(0.0).to(DEVICE)
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

        sr_imgT, f_t = netGT(lr_img, is_feat = True)
        sr_imgS, f_s = netG(lr_img, is_feat = True)

        fake_out = netD(sr_imgS)
        real_out = netD(hr_img)

        g_loss = generator_criterion(fake_out, real_out, sr_imgT, sr_imgS, hr_img, f_t, f_s)
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
                NUM_EPOCHS,
                running_results["d_loss"] / running_results["batch_sizes"],
                running_results["g_loss"] / running_results["batch_sizes"],
                running_results["learning_rate"]
            )
        )

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
        {"model": netG.state_dict(),
         "opti": optimizerG.state_dict()},
        f"./netG_{UPSCALE_FACTOR}x_epoch{epoch}.pt",
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