import torch
from torch import nn
import torch.nn.functional as F
import copy
from torch.autograd import Function
from collections import OrderedDict

def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    input: integer input tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point

class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """

        # if x_min is None or x_max is None or (sum(x_min == x_max) == 1
        #                                       and x_min.numel() == 1):
        #     x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
    
class QuantAct(nn.Module):
    """
    Class to quantize given activations
    """

    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True,
                 beta=0.9):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('beta', torch.Tensor([beta]))
        self.register_buffer('beta_t', torch.ones(1))
        self.act_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = True
    
    def forward(self, x):
        """
        quantize given activation x
        """

        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

            #self.beta_t = self.beta_t * self.beta
            #self.x_min = (self.x_min * self.beta + x_min * (1 - self.beta))/(1 - self.beta_t)
            #self.x_max = (self.x_max * self.beta + x_max * (1 - self.beta)) / (1 - self.beta_t)

            #self.x_min += -self.x_min + min(self.x_min, x_min)
            #self.x_max += -self.x_max + max(self.x_max, x_max)

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min,
                                          self.x_max)
            return quant_act
        else:
            return x


class QuantActPreLu(nn.Module):
    """
    Class to quantize given activations
    """

    def __init__(self,
                 act_bit,
                 full_precision_flag=False,
                 running_stat=True):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantActPreLu, self).__init__()
        self.activation_bit = act_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.act_function = AsymmetricQuantFunction.apply
        self.quantAct=QuantAct(activation_bit=act_bit,running_stat=True,full_precision_flag=full_precision_flag)

    def __repr__(self):
        s = super(QuantActPreLu, self).__repr__()
        s = "(" + s + " activation_bit={}, full_precision_flag={})".format(
            self.activation_bit, self.full_precision_flag)
        return s

    def set_param(self, prelu):
        self.weight = nn.Parameter(prelu.weight.data.clone())


    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x):

        w = self.weight
        x_transform = w.data.detach()
        a_min = x_transform.min(dim=0).values
        a_max = x_transform.max(dim=0).values
        if not self.full_precision_flag:
            w = self.act_function(self.weight, self.activation_bit, a_min,
                                     a_max)
        else:
            w = self.weight

        #inputs = max(0, inputs) + alpha * min(0, inputs)

        #w_min = torch.mul( F.relu(-x),-w)
        #x= F.relu(x) + w_min
        #inputs = self.quantized_op.add(torch.relu(x), weight_min_res)
        x = F.prelu(x, weight = w)
        x = self.quantAct(x)
        return x

class Quant_Linear(nn.Module):
    """
    Class to quantize given linear layer weights
    """
    
    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply
    
    def __repr__(self):
        s = super(Quant_Linear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s
    
    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
    
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        else:
            w = self.weight
        return F.linear(x, weight=w, bias=self.bias)


class Quant_Conv2d(nn.Module):
    """
    Class to quantize given convolutional layer weights
    """
    
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply
    
    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s
    
    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None
    
    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
    
def quantize_model( model, weight_bit = None, act_bit = None, full_precision_flag = False ):
        """
        Recursively quantize a pretrained single-precision model to int8 quantized model
        model: pretrained single-precision model
        """
        #if not (weight_bit) and not (act_bit ):
        #    weight_bit = self.settings.qw
        #    act_bit = self.settings.qa
        # quantize convolutional and linear layers
        if type(model) == nn.Conv2d:
            quant_mod = Quant_Conv2d(weight_bit=weight_bit,full_precision_flag=full_precision_flag)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.Linear:
            quant_mod = Quant_Linear(weight_bit=weight_bit,full_precision_flag=full_precision_flag)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.PReLU:
            quant_mod = QuantActPreLu(act_bit=act_bit,full_precision_flag=full_precision_flag)
            quant_mod.set_param(model)
            return quant_mod
        # quantize all the activation
        elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model)==nn.PReLU:
            return nn.Sequential(*[model, QuantAct(activation_bit=act_bit,full_precision_flag=full_precision_flag)])
        # recursively use the quantized module to replace the single-precision module
        elif type(model) == nn.Sequential or isinstance(model,nn.Sequential):
                mods = OrderedDict()
                for n, m in model.named_children():
                    mods[n] = quantize_model(m, weight_bit=weight_bit, act_bit=act_bit,full_precision_flag=full_precision_flag)
                return nn.Sequential(mods)
        else:
            q_model = copy.deepcopy(model)
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    setattr(q_model, attr, quantize_model(mod,weight_bit=weight_bit, act_bit=act_bit,full_precision_flag=full_precision_flag))
            return q_model

def freeze_model( model):
        """
        freeze the activation range
        """
        if type(model) == QuantAct:
            model.fix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                freeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    freeze_model(mod)
            return model

def unfreeze_model( model):
        """
        unfreeze the activation range
        """
        if type(model) == QuantAct:
            model.unfix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                unfreeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    unfreeze_model(mod)
            return model