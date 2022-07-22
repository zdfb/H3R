import os
import cv2
import numpy as np
from torch.utils import data


class H3R_Datasets(data.Dataset):
    def __init__(self, file_list, transforms = None, img_root = None, img_size = 128):
        assert img_root is not None
        self.line = None
        self.path = None
        self.img_size = img_size

        self.landmarks = None
        self.filenames = None
        self.img_root = img_root
        self.transforms = transforms

        with open(file_list, 'r') as f:
            self.lines = f.readlines()
    
    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        image_name = self.line[0]
        image_name = image_name.replace('\\', '/')
        image_name = os.path.join(self.img_root, image_name)
        self.img = cv2.imread(image_name)
        self.img = cv2.resize(self.img, (self.img_size, self.img_size))
        self.landmarks = np.asarray(self.line[1:213], dtype = np.float32)
        
        if np.random.randn() < 0.5:
            self.img, self.landmarks = self.random_rotate(self.img, self.landmarks)
        
        if self.transforms:
            self.img = self.transforms(self.img)
        return self.img, self.landmarks
    
    def random_rotate(self, img, landmarks):
        angle = np.random.randint(-10, 10)  # 随机旋转角度

        cx = self.img_size // 2  # 中心x坐标
        cy = self.img_size // 2  # 中心y坐标

        cx = cx + int(np.random.randint(-self.img_size * 0.1, self.img_size * 0.1))
        cy = cy + int(np.random.randint(-self.img_size * 0.1, self.img_size * 0.1))

        M, landmarks = self.rotate(angle, (cx, cy), landmarks)

        imgT = cv2.warpAffine(img, M, (self.img_size, self.img_size))

        return imgT, landmarks

    def rotate(self, angle, center, landmarks):
        rad = angle * np.pi / 180.0
        alpha = np.cos(rad)
        beta = np.sin(rad)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0, 0] = alpha
        M[0, 1] = beta
        M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
        M[1, 0] = -beta
        M[1, 1] = alpha
        M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

        landmarks = landmarks.reshape(106, 2)
        landmarks = landmarks * 128
        landmarks = np.asarray([(M[0, 0] * x + M[0, 1] * y +  M[0, 2], M[1, 0] * x 
                                                   + M[1, 1] * y + M[1, 2]) for (x, y) in landmarks])
        
        landmarks = landmarks.reshape(-1)
        landmarks = landmarks / 128

        return M, landmarks
        
    def __len__(self):
        return len(self.lines)