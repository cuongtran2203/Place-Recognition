from ConTNet import *
import torch.nn as nn
from metrics import ArcMarginProduct
from ConTNet import build_model,create_ConTNet_M
import numpy as np
import torch
class ConvTmixArcFace(nn.Module):
    def __init__(self,model_pretrained=None) -> None:
        super(ConvTmixArcFace,self).__init__()
        self.model=build_model(arch='ConT-M', use_avgdown=True, relative=True, qkv_bias=True, pre_norm=True)
        self.model.load_state_dict(torch.load(model_pretrained),strict=False)
        # Chọn 9 khối đầu tiên
        selected_layers = list(self.model.children())[:-2]
        # Tạo một mô hình mới chỉ chứa 9 khối đầu tiên
        self.feats = torch.nn.Sequential(*selected_layers)
        self.GAP=nn.AvgPool2d(5)
        self.arcface=ArcMarginProduct(in_features=512,out_features=512)
    def forward(self,x):
        feature=self.feats(x)
        emb=self.GAP(feature)
        print(emb.shape)
        # emb_v=self.arcface(emb)
        return emb
if __name__=="__main__":
    device = torch.device("cuda")
    model=ConvTmixArcFace(model_pretrained="checkpoint_M.pth")
    input = torch.Tensor(4, 3, 224, 224)
    out=model(input)
    print(out.shape)

