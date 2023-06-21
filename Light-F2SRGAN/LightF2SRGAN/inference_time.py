from LightF2SRGAN.quantize import *
import torch
import numpy as np
import copy
import time

def load_model(model, model_filepath, device):
    if (model_filepath != ""):
        model.load_state_dict(torch.load(model_filepath, map_location=device)['model'])
    return model

def F2SRGAN_inference_time_check_JIT(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=500):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)

    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%100==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

NUM_BIT_QUANTIZED = 16
POSTFIX = "float32" if (NUM_BIT_QUANTIZED == 32) else "fp16" if (NUM_BIT_QUANTIZED == 16) else "int8"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPSCALE_FACTOR = 4
RESOLUTION = [144, 360, 480, 720, 1080]
RGB_RANGE = 255.0
REPLICATION_EXP = 10

PRECISION = torch.float if (NUM_BIT_QUANTIZED == 32) else {torch.half, torch.float}
POSTFIX = "float32" if (NUM_BIT_QUANTIZED == 32) else "fp16"
        
MODEL_NAME = "F2SRGAN"
FILE_QAT_NAME_OUTPUT = f"./trained_{MODEL_NAME}_{POSTFIX}"
MODEL_FILEPATH_QAT = "../../pretrain_weight/F2SRGAN_4x_QAT.pth.tar"

model_quant = torch.load(MODEL_FILEPATH_QAT)['deploy']
model_quant.load_state_dict(torch.load(MODEL_FILEPATH_QAT)['model'])
model_quant.cuda().eval()

for resolution in RESOLUTION:
    with torch.no_grad():
        jit_model = torch.jit.trace(model_quant.cuda().half(), torch.rand(1, 3, resolution, resolution).to("cuda").half())
        torch.jit.save(jit_model, FILE_QAT_NAME_OUTPUT + f"_{resolution}.jit.pt")
    qat_model = torch.jit.load(FILE_QAT_NAME_OUTPUT + f"_{resolution}.jit.pt").eval().to("cuda")
    print(f'--------------------Resolution = {resolution} x {resolution}--------------------')
    F2SRGAN_inference_time_check_JIT(qat_model, input_shape=(1, 3, resolution, resolution), dtype=POSTFIX)