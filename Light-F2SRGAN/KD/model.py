import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeperableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(SeperableConv2d, self).__init__(
            nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels, groups=in_channels, 
                kernel_size=kernel_size, padding='same', dilation=dilation,
                bias=bias, padding_mode=padding_mode
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__(
            SeperableConv2d(in_channels, in_channels * scale_factor**2, kernel_size=3),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(num_parameters=in_channels),
            SeperableConv2d(in_channels, in_channels * scale_factor**2, kernel_size=3),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(num_parameters=in_channels)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_bn=False, use_ffc=False, use_act=True, discriminator=False, **kwargs):
        if use_ffc: conv = FFC(in_channels, out_channels, kernel_size=3, 
                ratio_gin=0.5, ratio_gout=0.5, inline = True
            )
        else: conv = SeperableConv2d(in_channels, out_channels, **kwargs)
        m = [conv]
        
        if use_bn: m.append(nn.BatchNorm2d(out_channels))
        if use_act: m.append(nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels))
        super(ConvBlock, self).__init__(*m)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, index):
        super(ResidualBlock, self).__init__()
        
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_ffc=True if index % 2 == 0 else False
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False
        )
        
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
#         else:
#             features = []
            
        out = self.block1(x)
        out = self.block2(out)
        out = out.mul(0.1)
        out += x
        return out, [out]


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Generator_T(nn.Module):
    def __init__(self, in_channels: int = 3, num_channels: int = 64, num_blocks: int = 16, upscale_factor: int = 4):
        super(Generator_T, self).__init__()
        
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=3, use_act=False)

        self.residual1 = nn.Sequential(
            *[ResidualBlock(num_channels, _) for _ in range(num_blocks//4)]
        )
        self.residual2 = nn.Sequential(
            *[ResidualBlock(num_channels, _) for _ in range(num_blocks//4)]
        )
        self.residual3 = nn.Sequential(
            *[ResidualBlock(num_channels, _) for _ in range(num_blocks//4)]
        )
        self.residual4 = nn.Sequential(
            *[ResidualBlock(num_channels, _) for _ in range(num_blocks//4)]
        )

        self.upsampler = UpsampleBlock(num_channels, scale_factor=2)
        self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=3)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 255
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        
    def forward(self, x, is_feat = False):
        x = self.sub_mean(x)
        initial = self.initial(x)

        x, f1 = self.residual1(initial) # len(f1) = 4
        x, f2 = self.residual2(x)
        x, f3 = self.residual2(x)
        x, f4 = self.residual2(x)
        x = x + initial

        x = self.upsampler(x)
        out = self.final_conv(x)
        out = self.add_mean(out)

        if is_feat:
            return out, f1+ f2+ f3+ f4
        else:
            return out


class Generator_S(nn.Module):
    def __init__(self, in_channels: int = 3, num_channels: int = 64, num_blocks: int = 8, upscale_factor: int = 4):
        super(Generator_S, self).__init__()
        
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=3, use_act=False)

        self.residual1 = nn.Sequential(
            *[ResidualBlock(num_channels, 1) for _ in range(num_blocks//4)]
        )
        self.residual2 = nn.Sequential(
            *[ResidualBlock(num_channels, 1) for _ in range(num_blocks//4)]
        )
        self.residual3 = nn.Sequential(
            *[ResidualBlock(num_channels, 1) for _ in range(num_blocks//4)]
        )
        self.residual4 = nn.Sequential(
            *[ResidualBlock(num_channels, 1) for _ in range(num_blocks//4)]
        )

        self.upsampler = UpsampleBlock(num_channels, scale_factor=2)
        self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=3)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 255
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        
    def forward(self, x, is_feat = False):
        x = self.sub_mean(x)
        initial = self.initial(x)

        x, f1 = self.residual1(initial)
        x, f2 = self.residual2(x)
        x, f3 = self.residual2(x)
        x, f4 = self.residual2(x)
        x = x + initial

        x = self.upsampler(x)
        out = self.final_conv(x)
        out = self.add_mean(out)
        if is_feat:
            return out, f1+ f2+ f3+ f4
        else:
            return out


def ComplexConv(x, weight):
    real = F.conv2d(x.real, weight.real, None, stride=1, padding=0, dilation=1, groups=1) - \
           F.conv2d(x.imag, weight.imag, None, stride=1, padding=0, dilation=1, groups=1)
    imag = F.conv2d(x.real, weight.imag, None, stride=1, padding=0, dilation=1, groups=1) + \
           F.conv2d(x.imag, weight.real, None, stride=1, padding=0, dilation=1, groups=1)
    x = torch.complex(real, imag)
    return x


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, 2, dtype=torch.float32) * 0.02)
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        
    def forward(self, x):
        B, C, H, W = x.shape
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        y = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        
        # FFT Shift
        y = torch.fft.fftshift(y)

        weight = torch.view_as_complex(self.complex_weight)
        y = ComplexConv(y, weight)

        # FFT IShift
        y = torch.fft.ifftshift(y)
        
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        y = torch.fft.irfftn(y, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        return y
    

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, enable_lfu=True, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            SeperableConv2d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, **fu_kwargs)

        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = SeperableConv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_h = h // split_no
            split_w = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_h, dim=-2)[0:2], dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_w, dim=-1)[0:2], dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()

            if h % 2 == 1:
                h_zeros = torch.zeros(xs.shape[0], xs.shape[1], 1, xs.shape[3]).to(DEVICE)
                xs = torch.cat((xs, h_zeros), dim=2)
            if w % 2 == 1:
                w_zeros = torch.zeros(xs.shape[0], xs.shape[1], xs.shape[2], 1).to(DEVICE)
                xs = torch.cat((xs, w_zeros), dim=3)
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, inline=True, stride=1, padding=0,
                 dilation=1, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.inline = inline

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else SeperableConv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else SeperableConv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else SeperableConv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, enable_lfu, **spectral_kwargs)
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else SeperableConv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.global_in_num], x[:, -self.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
            
        out = out_xl, out_xg
        if self.inline:
            out = torch.cat(out, dim=1)

        return out

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss
    
    
class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class AFD(nn.Module):
    def __init__(self, args):
        super(AFD, self).__init__()
        self.guide_layers = args['guide_layers']
        self.hint_layers = args['hint_layers']
        self.attention = Attention(args).to(DEVICE)

    def forward(self, g_s, g_t):
        g_t = [g_t[i] for i in range(self.guide_layers)]
        g_s = [g_s[i] for i in range(self.hint_layers)]
        loss = self.attention(g_s, g_t)
        return sum(loss)


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args['qk_dim']
        self.n_t = args['n_t']
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        self.p_t = nn.Parameter(torch.Tensor(len(args['t_shapes']), args['qk_dim']))
        self.p_s = nn.Parameter(torch.Tensor(len(args['s_shapes']), args['qk_dim']))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t): 
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args['qk_dim']) for t_shape in args['t_shapes']])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args['t_shapes'])
        self.s = len(args['s_shapes'])
        self.qk_dim = args['qk_dim']
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args['unique_t_shapes']]) # [2, 64, 48, 48]

        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args['s_shapes']])
        self.bilinear = nn_bn_relu(args['qk_dim'], args['qk_dim'] * len(args['t_shapes']))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1).view(bs * self.s, -1)  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
#         print(g_s[0].pow(2).mean(1, keepdim=True).shape) # torch.Size([16, 1, 12, 12])
#         print(self.sample(g_s[0].pow(2).mean(1, keepdim=True)).shape) # torch.Size([16, 1, 48, 48])
#         print(self.sample(g_s[0].pow(2).mean(1, keepdim=True)).view(bs, -1).shape) # torch.Size([16, 2304])
        g_s = torch.stack([(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s
    
class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        features: tuple = (64, 64, 128, 128, 256, 256, 512, 512),
    ) -> None:
        super(Discriminator, self).__init__()

        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return torch.sigmoid(self.classifier(x))