import warnings
import os

import numpy
import torch

import networks as networks
from empty_cache import empty_cache


class Interpolator:
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    def __init__(self, model_directory: str, sf: int, height: int, width: int, batch_size=1, **dain):
        # args
        self.batch_size = batch_size
        # Model
        model = networks.__dict__[dain['net_name']](
            channel=3, filter_size=4, timestep=1 / sf, training=False).cuda()
        empty_cache()

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(model_directory).items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del pretrained_dict, model_dict
        self.model = model.eval()

        # pader
        if width != ((width >> 7) << 7):
            intWidth_pad = (((width >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - width) / 2)
            intPaddingRight = intWidth_pad - width - intPaddingLeft
        else:
            intPaddingLeft = 32
            intPaddingRight = 32
        if height != ((height >> 7) << 7):
            intHeight_pad = (((height >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - height) / 2)
            intPaddingBottom = intHeight_pad - height - intPaddingTop
        else:
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])
        self.hs = intPaddingLeft  # Horizontal Start
        self.he = intPaddingLeft + width
        self.vs = intPaddingTop
        self.ve = intPaddingTop + height  # Vertical End
        if 'CUDA_EMPTY_CACHE' in os.environ and int(os.environ['CUDA_EMPTY_CACHE']):
            self.ndarray2tensor = lambda frames: [torch.squeeze(pader(torch.unsqueeze((torch.ByteTensor(frame)[:, :, :3].permute(2, 0, 1).float() / 255), 0))) for frame in frames]
            self.batch = torch.FloatTensor(batch_size + 1, 3, intPaddingTop + height + intPaddingBottom, intPaddingLeft + width + intPaddingRight)
            self.torch_stack = lambda X0, X1: torch.stack((X0, X1), dim=0).cuda()
        else:
            self.ndarray2tensor = lambda frames: [torch.squeeze(pader(torch.unsqueeze((torch.cuda.ByteTensor(frame)[:, :, :3].permute(2, 0, 1).float() / 255), 0))) for frame in frames]
            self.batch = torch.cuda.FloatTensor(batch_size + 1, 3, intPaddingTop + height + intPaddingBottom, intPaddingLeft + width + intPaddingRight)
            self.torch_stack = lambda X0, X1: torch.stack((X0, X1), dim=0)
        if dain['net_name'] == 'DAIN_slowmotion':
            self.tensor2ndarray = lambda y_: [[(255*item).clamp(0.0, 255.0).byte()[0, :, self.vs:self.ve,self.hs:self.he].permute(1, 2, 0).cpu().numpy()] for item in y_]
        elif dain['net_name'] == 'DAIN':
            self.tensor2ndarray = lambda y_: [[(255*item).clamp(0.0, 255.0).byte()[:, self.vs:self.ve,self.hs:self.he].permute(1, 2, 0).cpu().numpy()] for item in y_]

    def interpolate(self, frames):
        f = self.ndarray2tensor(frames)
        for i in range(self.batch_size):
            self.batch[i + 1] = f[i]
        # print(self.batch)
        X0 = self.batch[:-1]
        X1 = self.batch[1:]
        empty_cache()
        y_ = self.tensor2ndarray(self.model(self.torch_stack(X0, X1)))
        empty_cache()
        self.batch[0] = self.batch[1]
        return y_
