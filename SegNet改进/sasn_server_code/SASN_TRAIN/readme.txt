使用https://github.com/hlwang1124/SNE-RoadSeg框架
环境Python 3.7, CUDA 10.0, cuDNN 7 and PyTorch 1.1.
使用了VKITTI2数据集
使用时请修改./scripts/train.sh 中的--split_scheme ./datasets/all_120，此处对应为dataset list，请按照实际目录生成。
训练的tensorboard在./runs下，权重在./save_models/sasn下

