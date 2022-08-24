# coding: UTF-8

import time
import torch
import random
from tqdm import tqdm
import csv
from datetime import timedelta

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class DataProcessor(object):
    def __init__(self, path, device, tokenizer, batch_size, max_seq_len, seed):
        # 随机种子
        self.seed = seed
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.data = self.load(path)

        self.index = 0
        # 除batch后是否残余
        self.residue = False
        # 总数据数
        self.num_samples = len(self.data[0])
        # 总batch数
        self.num_batches = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.residue = True

    def load(self, path):
        contents = []
        labels = []
        with open(path, encoding='latin-1') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                sentence = line[3]
                label = line[5]
                contents.append(sentence)
                labels.append(int(label))
        # 数据打散
        index = list(range(len(labels)))
        random.seed(self.seed)
        random.shuffle(index)
        contents = [contents[_] for _ in index]
        labels = [labels[_] for _ in index]
        return (contents, labels)

    # 重写迭代器函数
    def __next__(self):
        # 最后一个batch，需要满足residue=True
        # 否则省略
        if self.residue and self.index == self.num_batches:
            batch_x = self.data[0][self.index * self.batch_size: self.num_samples]
            batch_y = self.data[1][self.index * self.batch_size: self.num_samples]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch
        # 如果索引大于最大batch，则停止
        elif self.index >= self.num_batches:
            self.index = 0
            raise StopIteration
        # 索引在最大batch内
        else:
            batch_x = self.data[0][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch_y = self.data[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch

    def _to_tensor(self, batch_x, batch_y):
        inputs = self.tokenizer.batch_encode_plus(
            batch_x,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation="longest_first",
            return_tensors="pt")
        inputs = inputs.to(self.device)
        labels = torch.LongTensor(batch_y).to(self.device)
        return (inputs, labels)

    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches
