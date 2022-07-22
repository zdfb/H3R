import torch
import numpy as np
import torch.nn.functional as F



def heatmap2coord(heatmap, topk = 9):
    N, C, H, W = heatmap.shape  # (N, 106, 64, 64)
    score, index = heatmap.view(N, C, 1, -1).topk(topk, dim = -1)  # 在106个特征图上进行运算，取每张特征图上最大的k个点,

    # index (N, 106, 1, K)
    coord = torch.cat([index % W, index // W], dim = 2) # (N, 106, 2, K)
    return (coord * F.softmax(score, dim = -1)).sum(-1)  # 取得最后的小数形式的特征点, (N, C, 2)


def generate_gaussian(t, x, y, sigma = 10):

    _gaussians = {}


    h,w = t.shape
    
    # Heatmap pixel per output pixel
    mu_x = int(0.5 * (x + 1.) * w)
    mu_y = int(0.5 * (y + 1.) * h)
    
    tmp_size = sigma * 3
    
    # Top-left
    x1,y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)
    
    # Bottom right
    x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
    if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
        return t
    
    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    x0 = y0 = size // 2
    
    # The gaussian is not normalized, we want the center value to equal 1
    g = _gaussians[sigma] if sigma in _gaussians \
                else torch.Tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))
    _gaussians[sigma] = g
    
    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1
    
    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)
    
    t[img_y_min:img_y_max, img_x_min:img_x_max] = \
      g[g_y_min:g_y_max, g_x_min:g_x_max]
    
    return t


def coord2heatmap(w, h, ow, oh, x, y, random_round = False, random_round_with_gaussian=  False):

    # Get scale
    sx = ow / w
    sy = oh / h
    
    # Unrounded target points
    px = x * sx
    py = y * sy
    
    # Truncated coordinates  整数部分
    nx, ny = int(px), int(py)
    
    # Coordinate error  小数部分
    ex, ey = px - nx, py - ny

    # Heatmap    
    heatmap = torch.zeros(ow, oh)

    if random_round_with_gaussian:
        xyr = torch.rand(2)
        xx = (ex >= xyr[0]).long()
        yy = (ey >= xyr[1]).long()
        row = min(ny + yy, heatmap.shape[0] - 1)
        col = min(nx + xx, heatmap.shape[1] - 1)

        # Normalize into - 1, 1
        col = (col / float(ow)) * (2) + (-1)
        row = (row / float(oh)) * (2) + (-1)
        heatmap = generate_gaussian(heatmap, col, row, sigma = 1.5) # sigma = 1.5?


    elif random_round:
        xyr = torch.rand(2)
        xx = (ex >= xyr[0]).long()
        yy = (ey >= xyr[1]).long()
        heatmap[min(ny + yy, heatmap.shape[0] - 1), 
                min(nx + xx, heatmap.shape[1] - 1)] = 1
    
    return heatmap


def lmks2heatmap(lmks, random_round = False, random_round_with_gaussian = False):
    w, h, ow, oh = 128, 128, 32, 32
    heatmap = torch.rand((lmks.shape[0], lmks.shape[1], ow, oh))  # (B, 106, 64, 64)
    for i in range(lmks.shape[0]):  # num_lmks
        for j in range(lmks.shape[1]):
            heatmap[i][j] = coord2heatmap(w, h, ow, oh, lmks[i][j][0], lmks[i][j][1], random_round = random_round, random_round_with_gaussian = random_round_with_gaussian)
    
    return heatmap


def clip_by_tensor(t, t_min, t_max):

    t = t.float()  # 转化为float

    # 小于极小值的部分用极小值代替
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def binary_heatmap_loss(preds, targs, pos_weight = None):
    
    preds = preds.float()
    
    if pos_weight is not None:
        _, p, h, w = preds.shape
        
        pos_weight = torch.tensor(pos_weight, device = preds.device).expand(p, h, w)

    return F.binary_cross_entropy_with_logits(preds, targs)

def mse_loss(preds, targs):

    criterion = torch.nn.MSELoss(reduction = 'mean')

    return criterion(preds, targs)

def l1_loss(preds, targs):
    
    criterion = torch.nn.L1Loss(reduction = 'mean')

    return criterion(preds, targs)






