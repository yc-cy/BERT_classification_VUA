# coding: UTF-8

import os
import torch

class Config(object):
    def __init__(self, data_dir):

        # 获取数据集
        assert os.path.exists(data_dir)
        self.train_file = os.path.join(data_dir, "VUA_formatted_train.csv")
        self.dev_file = os.path.join(data_dir, "VUA_formatted_test.csv")
        assert os.path.isfile(self.train_file)
        assert os.path.isfile(self.dev_file)

        # 设置模型保存目录
        self.saved_model_dir = os.path.join(data_dir, "model")
        self.saved_model = os.path.join(self.saved_model_dir, "bert_model.pth")
        if not os.path.exists(self.saved_model_dir):
            os.mkdir(self.saved_model_dir)

        # 获取文本输出标签
        self.num_labels = 2
        self.label_list = ["yes", "no"]

        # 微调训练配置
        self.num_epochs = 3
        # 日志输出频率，batch数/次
        self.log_batch = 100
        self.batch_size = 128
        self.max_seq_len = 100
        # 误差1000次后未更新，则停止训练
        self.require_improvement = 1000

        # 预热阶段包含epoch数
        self.warmup_steps = 0
        # 正则化权重系数
        self.weight_decay = 0.01
        # 梯度裁剪比率（防止梯度爆炸）
        self.max_grad_norm = 1.0
        self.learning_rate = 5e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

