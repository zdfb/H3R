# H3R 
## 数据集地址
+ 数据集来源于飞桨社区
>- 链接：https://aistudio.baidu.com/aistudio/datasetdetail/55093/0

## 数据集准备

  ```bash
  # 下载数据集到data/imgs下
  cd data
  python prepare.py
  ```
  ```bash
  # data 文件夹结构
  data/
    imgs/
    train_data/
      imgs/
      list.txt
    test_data/
      imgs/
      list.txt
  ```
## 开始训练
  ```
  cd ..
  python train.py
  ```
## Reference
https://github.com/baoshengyu/H3R
https://github.com/vuthede/heatmap-based-landmarker
https://elte.me/2021-03-10-keypoint-regression-fastai#random-rounding-and-high-resolution-net
