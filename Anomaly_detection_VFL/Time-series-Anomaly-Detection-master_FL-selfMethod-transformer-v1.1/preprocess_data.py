import os
import torch
from torch import device
import glob
import datetime
import numpy as np
import shutil
from pathlib import Path
import pickle

"""
0.数据的最后一个维度是label
1.train和label的处理过程一样，所以这里需要train的最后一个维度也是label才行
2.有数据的标准化，standardization() ->  preprocessing() ->  __init__()  这是调用过程
3.注意，用的数据 在 labeled 文件夹里面
"""

def normalization(seqData,max,min):  # 最简单的正则化 数据在0-1之间 暂时没用到
    return (seqData -min)/(max-min)

def standardization(seqData,mean,std):  # 标准化，注意和上面的区别
                                        # 数据0均值，1方差
    return (seqData-mean)/(std+1e-7)

def reconstruct(seqData,mean,std):  # 有归一数据变回正常尺度的数据
    return seqData*std+mean

class PickleDataLoad(object):
    def __init__(self, data_type, filename, augment_test_data=True):
        # data_type ：ecg等
        # filename：chfdb_chf13_45590.pkl 数据集里面具体那个小数据集部分
        # augment：测试集是否需要数据增强，首先训练集肯定数据增强的
        
        self.augment_test_data=augment_test_data
        self.trainData, self.trainLabel = self.preprocessing(Path('dataset',data_type,'labeled','train',filename),train=True)
                                                        # 这个path应该是 dataset/ecg/labeled/train/chfdb
        self.testData, self.testLabel = self.preprocessing(Path('dataset',data_type,'labeled','test',filename),train=False)

    #def augmentation(self,data,label,noise_ratio=0.05,noise_interval=0.0005,max_length=100000): # 原始版本
    def augmentation(self, data, label, noise_ratio=0.05, noise_interval=0.0005, max_length=200000):
        noiseSeq = torch.randn(data.size())  # 生成和data同样尺度的噪声
        augmentedData = data.clone()
        augmentedLabel = label.clone()
        for i in np.arange(0, noise_ratio, noise_interval):
            scaled_noiseSeq = noise_ratio * self.std.expand_as(data) * noiseSeq
                                            # 比例*标准差*随机数
            augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq], dim=0)
                                            # 增强后的数据是原数据连接上 加上噪声后的数据
            augmentedLabel = torch.cat([augmentedLabel, label])
                                            # 数据变了但是label不变，这才叫数据增强！
            if len(augmentedData) > max_length:  # 太长就截断
                augmentedData = augmentedData[:max_length]
                augmentedLabel = augmentedLabel[:max_length]
                break

        return augmentedData, augmentedLabel

    def preprocessing(self, path, train=True):
        """ Read, Standardize, Augment """

        with open(str(path), 'rb') as f:
            data = torch.FloatTensor(pickle.load(f))  # 打开的文件可以直接pickle.load
            label = data[:,-1]  # label是最后一列
            data = data[:,:-1]  # 数据是其余列
        if train:
            self.mean = data.mean(dim=0)  # dim=0，每个维度生成一个均值
            self.std= data.std(dim=0)  # 生成标准差
            self.length = len(data)
            data,label = self.augmentation(data,label)  # 数据增强
        else:  # 测试集是否需要数据增强呢
            if self.augment_test_data:
                data, label = self.augmentation(data, label)

        data = standardization(data,self.mean,self.std)

        return data,label
        # 上面这个data是真实的朴素的数据 torch.Size([100000, 2])
        # label是朴素的label torch.Size([100000])


    def batchify(self,args,data, bsz):  # 
        """

        data        :trainData 
        """
        nbatch = data.size(0) // bsz  # data shape: [time,dim]
        trimmed_data = data.narrow(0,0,nbatch * bsz)  # 为了规整数据的数量
        batched_data = trimmed_data.contiguous().view(bsz, -1, trimmed_data.size(-1)).transpose(0,1)
                                # [nbatch,bsz,dim]
                                # 上面先分成bsz个块，这样保证最后剔除batch维度，时间还是连续的
                                # 时间连续已经得到了证明，在K80机器里__test__2里面
                                # 实际上bsz如果是3，那么nbatch就是length/3
                                # 第一个维度 nbatch 其实还是 时间seq的那个角度
        batched_data = batched_data.to(device(args.device))
        return batched_data  # 直接返回数据！

