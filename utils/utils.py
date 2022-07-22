import numpy as np

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_nme(preds, target):
    preds = preds.reshape(preds.shape[0], -1, 2).detach().cpu().numpy() # landmark 
    target = target.reshape(target.shape[0], -1, 2).detach().cpu().numpy() # landmark_gt

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    l, r = 35, 93

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]

        eye_distant = np.linalg.norm(pts_gt[l ] - pts_gt[r])
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis = 1)) / (eye_distant * 106)
    
    return rmse