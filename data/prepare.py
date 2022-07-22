import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm


class ImageDate:
    def __init__(self, line, imgDir, image_size = 128):
        self.image_size = image_size
        line = line.strip().split()

        assert(len(line) == 217) # 1 + 4 + 106 * 2

        self.list = line
        self.landmark = np.asarray(list(map(float, line[5:])), dtype = np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, line[1:5])), dtype = np.int32).reshape(-1, 2)
        self.path = os.path.join(imgDir, line[0])

        self.img = None

        self.imgs = []
        self.landmarks = []
        self.boxes = []
    

    def load_data(self):
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh / 2).astype(np.int32)

        img = cv2.imread(self.path)
        boxsize = int(np.max(wh) * 1.2)  # 框区域扩张1.2倍

        xy = center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            imgT = cv2.copyMakeBorder(
                imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy) / boxsize  # 裁剪后 归一化
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

    def save_data(self, path, prefix):

        labels = []

        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (106, 2)
            save_path = os.path.join(path, prefix + '_' + str(i) + '.jpg')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            landmark_str = ' '.join(
                list(map(str, lanmark.reshape(-1).tolist())))

            label = '{} {} \n'.format(save_path, landmark_str)
            labels.append(label)
        
        return labels


def get_dataset_list(imgDir, outDir, landmarkFile, is_train, test_ratio = 0.1):
    with open(landmarkFile, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(os.path.split(outDir)[-1], 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)
        num_train = int(len(lines) * (1 - test_ratio))
        if is_train:
            lines = lines[:num_train]
        else:
            lines = lines[num_train:]

        for i, line in enumerate(tqdm(lines)):
            Img = ImageDate(line, imgDir, image_size = crop_img_size)
            img_name = Img.path
            Img.load_data()
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i) + '_' + filename)
            labels.append(label_txt)

    with open(os.path.join(outDir, 'list.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)  # 一次写入多行字符串

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    imageDirs = './imgs'
    crop_img_size = 128
    landmarkFile = os.path.join(root_dir, "bbox_landmark.txt")

    outDirs = ['test_data', 'train_data']
    for outDir in outDirs:
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if 'test' in outDir:
            is_train = False
        else:
            is_train = True
        get_dataset_list(imageDirs, outDir, landmarkFile, is_train, test_ratio=0.1)
    print('end')
