import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.hrnet import hrnet18, hrnet32
from models.heatmapmodel import heatmap2coord


class BinaryHeadBlock(nn.Module):
    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        super(BinaryHeadBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias = False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(proj_channels, out_channels, 1), )
    
    def forward(self, input):
        N, C, H, W = input.shape  
        binary_heats = self.layers(input)  # (N, 270, 64, 64) -> (N, 106, 64, 64)
        return binary_heats


class BinaryHeatmap2Coordinate(nn.Module):
    def __init__(self, stride = 4.0, topk = 9, **kwargs):
        super(BinaryHeatmap2Coordinate, self).__init__()
        self.topk = topk
        self.stride = stride
    
    def forward(self, input):

        return self.stride * heatmap2coord(input, self.topk)  # (N, 106, 2), 取得最后小数形式的特征点


class HeatmapHead(nn.Module):
    def __init__(self):
        super(HeatmapHead, self).__init__()
        
        self.head = BinaryHeadBlock(in_channels = 480, proj_channels = 270, out_channels = 106)
        self.decoder = BinaryHeatmap2Coordinate(topk = 9, stride = 4)
    
    def forward(self, input):
        binary_heats = self.head(input)  # (N, 270, 64, 64) -> (N, 106, 64, 64)
        lmks = self.decoder(binary_heats)

        return binary_heats, lmks


class HeatMapLandmarker(nn.Module):
    def __init__(self, pretrained = True):
        super(HeatMapLandmarker, self).__init__()
        self.backbone = hrnet32(pretrained = pretrained)  # 定义backbone, 输出形状为 N, 270, 64, 64
        self.heatmap_head = HeatmapHead()  # 定义H3Rnet的头, 
        
    def forward(self, x):
        heatmaps, landmark = self.heatmap_head(self.backbone(x))
        return heatmaps, landmark