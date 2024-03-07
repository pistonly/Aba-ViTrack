import torch
import numpy as np
import thop
from copy import deepcopy
import sys
sys.path.insert(0, "..")

from model.AbaViTrack import AbaViTrack
from model.head import CenterPredictor
from model.AbaViT import abavit_patch16_224
from Tracker import Tracker




def get_flops(model, template_size, search_size):
    """Return a model's FLOPs."""
    try:
        p = next(model.parameters())
        template = torch.empty((1, 3, template_size, template_size)).to(p.device)
        search = torch.empty((1, 3, search_size, search_size)).to(p.device)
        flops = thop.profile(deepcopy(model), inputs=(template, search), verbose=False)[0] / 1E9 * 2 if thop else 0  # stride GFLOPs
        return flops
    except Exception as e:
        print(e)
        return 0

def get_num_params(model):
    """Return  lwthe total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())

def export_onnx(model, file_path, template_size, search_size, dynamic=False):
    try:
        output_names = ['output']
        p = next(model.parameters())
        template = torch.empty((1, 3, template_size, template_size)).to(p.device)
        search = torch.empty((1, 3, search_size, search_size)).to(p.device)
        model = model.eval()
        torch.onnx.export(
            model.cpu() if dynamic else model,  # dynamic=True only compatible with cpu
            (template, search),
            file_path,
            verbose=False,
            opset_version=13,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['template', 'search'],
            output_names=output_names,
            dynamic_axes=dynamic or None)
    except Exception as e:
        print(e)

def build_box_head(in_channel, out_channel, search_size, stride):
    feat_sz = search_size / stride
    center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                  feat_sz=feat_sz, stride=stride)
    return center_head

def build_model():
    search_size = 256
    stride = 16
    backbone = abavit_patch16_224()
    box_head = build_box_head(backbone.embed_dim, 256, search_size, stride)

    model = AbaViTrack(
        backbone,
        box_head,
        output_tuple=True
    )

    return model

model = build_model()
tracker = Tracker(model)
template_size = tracker.template_size
search_size = tracker.search_size

param_num = get_num_params(model)
flops = get_flops(model, template_size, search_size)

print(f"param num: {param_num}, flops: {flops} Gflops")

export_onnx(model, "./aba_track.onnx", template_size, search_size)
