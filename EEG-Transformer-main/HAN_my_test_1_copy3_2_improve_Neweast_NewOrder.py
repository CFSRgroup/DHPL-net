"""
Transformer for EEG classification
"""
# 2分类
# 留一 的训练部分 不再做划分  直接把除一个被试之外的其他被试数据全做直接训练

# （1）：没有权重的归一化，没有先验的位置相关性编码，但是有原来的加的可学习位置编码position
# 共耗时:10.5256 min
# 最终的平均测试精度为: test_avg_acc:0.6444  test_avg_loss:1.9850  test_avg_F1_score:0.5250

# （2）：在（1）基础上加入权重归一化
# 共耗时:11.1647 min
# 最终的平均测试精度为: test_avg_acc:0.6176  test_avg_loss:2.1060  test_avg_F1_score:0.4987

# （3）：在（1）基础上加上5个时刻的先验的位置相关性编码
# 共耗时:10.5037 min
# 最终的平均测试精度为: test_avg_acc:0.6021  test_avg_loss:2.1155  test_avg_F1_score:0.4818

# （4）：在（3）的基础上去掉5个时刻的可学习位置编码



import os
import numpy as np
import math
import random
import time
import scipy.io as io
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
# from confusion_matrix1 import *
from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

# from confusion_matrix import plot_confusion_matrix
# from cm_no_normal import plot_confusion_matrix_nn
# from torchsummary import summary

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from sklearn.model_selection import LeaveOneOut
import adabound
cudnn.benchmark = False
cudnn.deterministic = True
from feature_reprocess import Datasets
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
# writer = SummaryWriter('/home/syh/Documents/MI/code/Trans/TensorBoardX/')

# torch.cuda.set_device(6)  设置多个GPU并行，下面三行我给注释掉了，只有一个GPU
'''
gpus = [6]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
'''

total_train_step = 0
total_test_step = 0
# best_test_acc = 0
scaler = None

weight_spectral = 0
weight_temporal = 0
weight_channel_before = 0
weight_channel_after = 0
weight_timeInstance_before = 0
weight_timeInstance_after = 0

# 相对位置先验编码
# 5个连续时刻编码位置：相对距离
w_t = np.array([[0, 1, 2, 3, 4],
     [1, 0, 1, 2, 3],
     [2, 1, 0, 1, 2],
     [3, 2, 1, 0, 1],
     [4, 3, 2, 1, 0]])
# w = w + np.spacing(1)
w1 = np.ones(25).reshape((5, 5)) * 5
w_t = w1 - w_t
w_t = F.softmax(torch.from_numpy(w_t), dim=-1)
w_t = Variable(w_t.cuda().type(torch.cuda.FloatTensor))
# 通道位置编码
AF3 = [76.8371, 33.7007, 21.227]
F7 = [42.4743, 70.2629, -11.42]
F3 = [53.1112, 50.2438, 42.192]
FC5 = [18.6433, 77.2149, 24.46]
T7 = [-16.0187, 84.1611, -9.346]
P7 = [-73.4527, 72.4343, -2.487]
O1 = [-112.449, 29.4134, 8.839]
O2 = [-112.156, -29.8426, 8.8]
P8 = [-73.0683, -73.0557, -2.54]
T8 = [-15.0203, -85.0799, -9.49]
FC6 = [19.9357, -79.5341, 24.438]
F4 = [54.3048, -51.8362, 40.814]
F8 = [44.4217, -73.0431, -12]
AF4 = [77.7259, -35.7123, 21.956]
channel_cartesian = [AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4]

d = np.empty((14, 14))
def channel_d():
    # 计算坐标c1和c2之间的距离
    for i in range(14):
        for j in range(14):
            if i != j:
                d[i][j] = ((channel_cartesian[i][0]-channel_cartesian[j][0])**2 + (channel_cartesian[i][1]-channel_cartesian[j][1])**2 + (channel_cartesian[i][2]-channel_cartesian[j][2])**2)**(1/2)
            else:
                d[i][j] = 1
    return d
w_c = 1./channel_d()
w_c = F.softmax(torch.from_numpy(w_c), dim=-1)
w_c = Variable(w_c.cuda().type(torch.cuda.FloatTensor))


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            # Reduce('b n e -> b e', reduction='mean'),  # (b,3,32)->(b,32)
            nn.LayerNorm(emb_size),  # (b,32)
            nn.Linear(emb_size, n_classes),  # (128,9)->(128,3)
        )
    def forward(self, x):
        out = self.clshead(x)
        return x, out

class PositionalEmbedding1DTimeSeq(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding

class PatchEmbeddingTimeSeq(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.positional_embedding = PositionalEmbedding1DTimeSeq(3, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.positional_embedding(x)  # 和一个可学习的位置向量相加  (128,6,9)
        return x

class ViT(nn.Sequential):
    def __init__(self, feature_forward_expansion=1, channel_forward_expansion=4, emb_size=12, gru_size=12, n_layers=1, timeSeq_forward_expansion=8, n_classes=2, **kwargs):  # depth网络重复结构的深度
        super().__init__()
        self.spectralAttention = FeatureAttention(14, 4, feature_forward_expansion)
        self.temporalAttention = FeatureAttention(14, 5, feature_forward_expansion)
        self.channelPosition = ChannelEmbedding(9)
        self.channelAttention = ChannelAttention(9, channel_forward_expansion)
        self.linear = nn.Linear(252, emb_size)
        self.gru = GRU2(emb_size, gru_size, n_layers)
        self.temporalPosition = TimeSeqEmbedding(gru_size*2)
        self.temporalAttentionUnion = TimeSeqAttention(gru_size*2, timeSeq_forward_expansion)
        self.classificationHead = ClassificationHead(gru_size*2, n_classes)


    def forward(self, x):

        spectroTemporal = x.permute(0, 1, 3, 2)  # (b,5,14,9)->(b,5,9,14)
        spectral = spectroTemporal[:, :, :4, :]  # (b,5,4,14)
        temporal = spectroTemporal[:, :, 4:, :]  # (b,5,5,14)
        spectralLogits = self.spectralAttention(spectral)  # (b,5,4,14)
        spectralLogits = spectralLogits.reshape(len(spectralLogits), 5, -1)  # (b,5,56)
        temporalLogits = self.temporalAttention(temporal)  # (b,5,5,14)
        temporalLogits = temporalLogits.reshape(len(temporalLogits), 5, -1)  # (b,5,70)

        channelPosition = self.channelPosition(x)  # (b,5,14,9)
        channelLogits = self.channelAttention(channelPosition)  # (b,3,4,9)
        channelLogits = channelLogits.reshape(len(channelLogits), 5, -1)

        union = torch.cat([spectralLogits, temporalLogits, channelLogits], dim=2)  # (b,5,14+14+9=37)  (b,5,37)
        union = self.linear(union)  # (b,5,37)->(b,5,32)

        gruLogits = self.gru(union)  # (b,5,64)

        temporalData = self.temporalPosition(gruLogits)  # (b,3,64)
        temporalData = self.temporalAttentionUnion(temporalData)  # (b,64)

        out = self.classificationHead(temporalData)
        return out

class TemporalSpectralSplit(nn.Module):
    def __init__(self):
        super(TemporalSpectralSplit, self).__init__()

    def forward(self, x):
        spectral = x[:, :4, :]  # (3b,9,14)
        temporal = x[:, 4:, :]
        return spectral, temporal

class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(x)
        return x

class FeatureAttention(nn.Module):
    def __init__(self, input_dim, fea_dim, feature_forward_expansion, drop_p=0.5, forward_drop_p=0.5):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            FeatureEncoderBlock(input_dim, fea_dim, feature_forward_expansion),
        )

    def forward(self, x):
        x = self.attention(x)  # (3b,9,14)
        # x = x.view(-1, 14, x.size(1))  # (42b,9)->(3b,14,9)
        return x

class FeatureEncoderBlock(nn.Sequential):
    def __init__(self,
                 input_dim,
                 fea_dim,
                 feature_forward_expansion,
                 drop_p=0.5,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(input_dim),  # 输入(42b,9)
                    FeatureMultiHeadAttention(input_dim, fea_dim, drop_p),
                    nn.Dropout(drop_p)  # (42b,9)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(input_dim),  # (42b,9)
                    FeedForwardBlock(
                        input_dim, expansion=feature_forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)  # (42b,9)
                )
            ),
            # Reduce('b n d e -> b n e', reduction='mean'),  # (b,5,9,14)->(b,5,14)
        )

class FeatureMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, fea_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.fea_dim = fea_dim
        self.queries = nn.Linear(input_dim, input_dim)
        self.keys = nn.Linear(input_dim, input_dim)
        self.values = nn.Linear(input_dim, input_dim)
        self.weight_norm = nn.LayerNorm(fea_dim)
        self.att_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:  # (b,5,4,14)
        queries = self.queries(x)  # (b,3,4,56)->(b,3,h,4,56/h)
        keys = self.keys(x)  # (128,5,15,9)->(128,5,3,15,3)
        values = self.values(x)  # (128,5,15,9)->(128,5,3,15,3)
        energy = torch.einsum('beqd, bekd -> beqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min  # 构造一个浮点数
            energy.mask_fill(~mask, fill_value)
        # scaling = 1 ** (1 / self.input_dim)
        scaling = self.input_dim ** (1 / 2)

        if self.fea_dim == 4:
            global weight_spectral
            weight_spectral = energy
        else:
            global weight_temporal
            weight_temporal = energy

        energy = self.weight_norm(energy)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('beal, belv -> beav', att, values)

        return out

class ChannelEmbedding(nn.Module):
    def __init__(self, input_dim):
        super(ChannelEmbedding, self).__init__()
        self.positional_embedding = PositionalEmbedding1D(14, input_dim)

    def forward(self, x):  # (3b,14,9)
        x = self.positional_embedding(x)
        return x

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding

class ChannelAttention(nn.Sequential):  # 输入(b,3,4,14,9)
    def __init__(self, input_dim, channel_forward_expansion, drop_p=0.5, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    ChannelMultiHeadAttention(input_dim, drop_p),  # (3b,14,16)
                    nn.Dropout(drop_p)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(input_dim),  # (3b,14,16)
                    FeedForwardBlock(
                        input_dim, expansion=channel_forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )
            ),
        )

class ChannelMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.weight_norm = nn.LayerNorm(14)
        self.att_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:  # (b,3,4,14,9)
        global weight_channel_before, weight_channel_after
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        energy = torch.einsum('bcqd, bckd -> bcqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min  # 构造一个浮点数
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        energy1 = torch.einsum('al, bclv -> bcav', w_c, energy)

        # energy = self.weight_norm(energy)  # 邻接矩阵之前的权重矩阵
        weight_channel_before = energy
        weight_channel_after = energy1

        energy1 = self.weight_norm(energy1)
        att = F.softmax(energy1 / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bfal, bflv -> bfav', att, values)  # (128,5,3,15,3)


        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),  # (128,15,9)->(128,15,4*9)=(128,15,36)
            nn.Dropout(drop_p),  # (128,15,36)
            #nn.GELU(),  # (128,15,36)
            nn.Linear(expansion * emb_size, emb_size),  # (128,15,9)
        )

class GRU2(nn.Module):
    def __init__(self, emb_size, gru_size, n_layers):
        super(GRU2, self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.gru = torch.nn.GRU(emb_size, gru_size, n_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # (3b,9)
        x = self.norm(x)
        output, hidden = self.gru(x)  # (b,3,16)->(b,3,16*2=32)
        output = self.dropout(output)
        output = torch.tanh(output)
        # return output[:, -1, :]
        return output

class TimeSeqEmbedding(nn.Module):
    def __init__(self, input_dim):
        super(TimeSeqEmbedding, self).__init__()
        self.positional_embedding = PositionalEmbedding1D_temporal(5, input_dim)

    def forward(self, x):
        x = self.positional_embedding(x)
        return x

class PositionalEmbedding1D_temporal(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding

class TimeSeqAttention(nn.Sequential):
    def __init__(self,
                 emb_size,
                 timeSeq_forward_expansion,
                 drop_p=0.5,
                 forward_drop_p=0.5):
        super().__init__(

            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),  # 输入(b,3,32)
                    TimeSeqMultiHeadAttention(emb_size, drop_p),  # (128,6,9)
                    nn.Dropout(drop_p)  # (64,15,9)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),  # (64,15,9)
                    FeedForwardBlock(
                        emb_size, expansion=timeSeq_forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)  # (64,15,9)
                )
            ),
            Reduce('b n e -> b e', reduction='mean'),  # (b,3,64)->(b,64)
        )

class TimeSeqMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.weight_norm = nn.LayerNorm(5)
        self.att_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:  # (b,3,64)
        global weight_timeInstance_before, weight_timeInstance_after
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        energy = torch.einsum('bqd, bkd -> bqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min  # 构造一个浮点数
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        energy1 = torch.einsum('al, blv -> bav', w_t, energy)

        # energy = self.weight_norm(energy)
        weight_timeInstance_before = energy
        weight_timeInstance_after = energy1

        energy1 = self.weight_norm(energy1)
        att = F.softmax(energy1 / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bal, blv -> bav', att, values)  # (128,5,3,15,3)



        return out

class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """
    def __init__(self):
        super(RDrop, self).__init__()
        # self.ce = nn.CrossEntropyLoss(reduction='none')
        # self.kld = nn.KLDivLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss()

    def forward(self, logits1, logits2, target, kl_weight=0.9):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.

        Returns:
            loss: Losses with the size of the batch size.
        """
        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + kl_weight * kl_loss
        return loss

class RDopoutLoss(nn.Module):
    def __init__(self):
        super(RDopoutLoss, self).__init__()
    def forward(self, outut1, output2, target):
        criterion = RDrop()
        loss = criterion(outut1, output2, target)
        return loss

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') #加上虚线对角线



## 超参数
# embedding_dim:原始特征维度=9
# hidden_size:隐藏层维度=9 or 大于9的数，可以丰富特征
# num_layers = 5  ，这里我先设定成1 避免过于复杂的网络
# batchsize = 128
# output_dim = 1  输入的维度，如果是2分类或者回归问题，输出维度为1，如果是n类（n>2)分类问题，则应该是n。
class Trans():
    def __init__(self, writer):  # nsub指的是被试编号，即下面main中调用时传入的i
        super(Trans, self).__init__()
        self.total_train_step = 0
        self.total_test_step = 0
        # self.batch_size = 50
        self.batch_size = 32
        self.n_epochs = 15  ########
        self.k_fold = 10
        # self.img_height = 22   # 我改的，我不做图像
        # self.img_width = 600
        # self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        # self.nSub = nsub  # nsub被试编号
        self.start_epoch = 0
        # self.root = '...'  # the path of data
        self.root = '.'  # 我改的：数据文件的根路径
        self.writer = writer


        self.pretrain = False  # 该模型预训练设置为False

        # self.log_write = open("results/log_subject%d.txt" % self.nSub, "w")

        # self.img_shape = (self.channels, self.img_height, self.img_width)  # something no use  这行我注释掉了

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        # 三个损失函数，转移到GPU上
        self.criterion_l1 = torch.nn.L1Loss().cuda()  # LLoss():平均绝对误差
        self.criterion_l2 = torch.nn.MSELoss().cuda()  # MSELoss():均方差
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()  # 交叉熵
        self.criterion_cls = RDopoutLoss().cuda()

        self.model = ViT().cuda()  # 把VIT模型转移到GPU上
        # summary(self.model, (5, 14, 9))
        # 下面一行我给注释掉了(因为只有一个GPU)
        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        # self.model = self.model.cuda()
        # print(self.model)
        # summary(self.model, (1, 16, 1000))  # input_size:(1, 16, 1000)


        # input = torch.ones((27, 12, 257)).cuda()
        # # input = b = torch.Tensor([2,3])
        # print("input模型测试，模拟数据大小：\n", np.shape(input))
        # label, output = self.model(input)
        # print("测试输出：", output.size())


        # summary(self.model, (90, 14, 257))  # torchsummary: 查看网络层形状、参数
        self.centers = {}


    def train(self, data_train, label_train,  data_val, label_val, data_test, label_test, test_result_acc, test_result_F1_score):
        global total_train_step, total_test_step, scaler
        epoch_all = 0

        data_train = data_train.reshape(len(data_train), 5, -1).reshape(-1, 126)  # (b,5,126)
        data_val = data_val.reshape(len(data_val), 5, -1).reshape(-1, 126)
        data_test = data_test.reshape(len(data_test), 5, -1).reshape(-1, 126)
        scaler = StandardScaler()
        train_scaler = scaler.fit_transform(data_train)
        val_scaler = scaler.transform(data_val)
        test_scaler = scaler.transform(data_test)
        data_train = np.array(train_scaler).reshape(-1, 5, 126)  # (280,3,4,14,9)
        data_train = data_train.reshape(len(data_train), 5, 14, 9)
        data_val = np.array(val_scaler).reshape(-1, 5, 126)
        data_val = data_val.reshape(len(data_val), 5, 14, 9)
        data_test = np.array(test_scaler).reshape(-1, 5, 126)  # (1152, 11, 9)
        data_test = data_test.reshape(len(data_test), 5, 14, 9)

        data_val = torch.from_numpy(data_val).cuda()
        label_val = torch.from_numpy(label_val - 1).cuda()
        dataset_val = torch.utils.data.TensorDataset(data_val, label_val)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        label_val = Variable(label_val.type(self.LongTensor))

        data_test = torch.from_numpy(data_test).cuda()
        label_test = torch.from_numpy(label_test - 1).cuda()
        dataset_test = torch.utils.data.TensorDataset(data_test, label_test)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        label_test = Variable(label_test.type(self.LongTensor))

        # 4、加载器
        data_train = torch.from_numpy(data_train).cuda()
        label_train = torch.from_numpy(label_train - 1).cuda()  # 使得label从0开始，表示下标
        dataset_train = torch.utils.data.TensorDataset(data_train, label_train)  # 将特征向量和标签打包成列表
        self.train_dataloader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.batch_size,
                                                            shuffle=True)
        # 5、优化器 Optimizers
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        self.optimizer = adabound.AdaBound(self.model.parameters(), lr=1e-3, final_lr=0.1)

        train_ls, test_ls = [], []  ##存储train_loss、train_acc; test_loss、test_acc


        for e in range(self.n_epochs):

            self.model.train()
            for data_i, label_i in self.train_dataloader:  ###分批训练
                data_i = Variable(data_i.cuda().type(self.Tensor))  # (128,5,14,9)
                label_i = Variable(label_i.cuda().type(self.LongTensor))  # (128,)
                tok1, output1 = self.model(data_i)  # outputs:(64,3)
                tok2, output2 = self.model(data_i)
                loss = self.criterion_cls(output1, output2, label_i)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_train_step = total_train_step + 1
            train_pred1 = torch.max(output1, 1)[1]
            train_pred2 = torch.max(output2, 1)[1]
            train_acc1 = float((train_pred1 == label_i).cpu().numpy().astype(int).sum()) / float(label_i.size(0))
            train_acc2 = float((train_pred2 == label_i).cpu().numpy().astype(int).sum()) / float(label_i.size(0))
            train_acc = (train_acc1 + train_acc2) / 2
            train_ls.append((loss.item(), train_acc))

            epoch_all = epoch_all + 1

            # 每训练1轮epoch，在验证集上都测试一次

            self.model.eval()
            with torch.no_grad():
                # for j, (data_j, label_j) in enumerate(self.test_dataloader):
                #     data_j = Variable(data_j.type(self.Tensor))
                #     Tok1, Cls1 = self.model(data_j)
                #     Tok2, Cls2 = self.model(data_j)
                #     if j == 0:
                #         tmp_label1 = Cls1
                #         tmp_label2 = Cls2
                #     else:
                #         tmp_label1 = torch.cat([tmp_label1, Cls1], dim=0)
                #         tmp_label2 = torch.cat([tmp_label2, Cls2], dim=0)
                # loss_test = self.criterion_cls(tmp_label2, tmp_label2, label_test)
                # y_pred1 = torch.max(tmp_label1, 1)[1]
                # y_pred2 = torch.max(tmp_label2, 1)[1]
                # acc1 = float((y_pred1 == label_test).cpu().numpy().sum()) / float(label_test.size(0))
                # acc2 = float((y_pred2 == label_test).cpu().numpy().sum()) / float(label_test.size(0))
                # acc = (acc1 + acc2)/2
                ## test_ls.append((loss_test.item(), acc))
                for j, (data_j, label_j) in enumerate(self.val_dataloader):
                    data_j = Variable(data_j.type(self.Tensor))
                    Tok1, Cls1 = self.model(data_j)
                    if j == 0:
                        tmp_label1 = Cls1
                    else:
                        tmp_label1 = torch.cat([tmp_label1, Cls1], dim=0)
                y_pred1 = torch.max(tmp_label1, 1)[1]
                acc = float((y_pred1 == label_val).cpu().numpy().sum()) / float(label_val.size(0))

            if epoch_all % 5 == 0:
                print('Epoch:', epoch_all,
                      '  Train acc:%.4f' % (train_acc),
                      '  Train loss:%.4f' % (loss.item()),
                      '  val acc:%.4f' % (acc))


        self.model.eval()
        # self.model.load_state_dict(torch.load('nBack_class2_LiuyiDevide{}.pth'.format(ith)))
        # 评估模型
        pred_all = []  # 记录每个batch的模型预测输出的标签
        label_all = []  # 对应的真实标签
        with torch.no_grad():
            for j, (data_k, label_k) in enumerate(self.test_dataloader):
                data_k = Variable(data_k.type(self.Tensor))
                Tok1, Cls1 = self.model(data_k)
                if j == 0:
                    tmp_label1 = Cls1
                else:
                    tmp_label1 = torch.cat([tmp_label1, Cls1], dim=0)

                pred1 = Cls1.cpu().detach().numpy()  # 先把pred1转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
                label1 = label_k.cpu().detach().numpy()
                pred_all.extend(np.argmax(pred1, axis=1))  # 求每一行的最大值索引
                label_all.extend(label1)

            acc_test_lastone = float((torch.max(tmp_label1, 1)[1] == label_test).cpu().numpy().sum()) / float(
                label_test.size(0))
            F1_score = f1_score(label_all, pred_all)
            print("#"*30, " 最终在测试集上的测试结果(最优的测试精度结果)： ", "#"*30, "\n", "acc:%.4f" % (acc_test_lastone))
            print("F1-Score:{:.4f}".format(F1_score))
            test_result_acc.append(acc_test_lastone)
            test_result_F1_score.append(F1_score)
            print("for_yet: acc:%.4f, F1:%.4f" % (np.mean(test_result_acc), np.mean(test_result_F1_score)))

            # score_AUC = torch.nn.Softmax(dim=1)(tmp_label1)
            # auc_prob_DHPLNet = np.array(score_AUC.cpu())
            # io.savemat('./ROC/prob_DHPLNet.mat', {'prob': auc_prob_DHPLNet})  # 保存网络在测试集上的预测结果的二分类概率，用于计算ROC曲线
            # prob_label = np.array(label_test.cpu())
            # io.savemat('./ROC/prob_DHPLNet_trueLabel.mat', {'label': prob_label})  # 保存对应的真实标签
        return label_all, pred_all




def main():

    # acc_KL = []
    # f1_KL = []
    # for KL_we in [0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5]:

    global weight_spectral, weight_temporal, weight_channel_before, weight_channel_after, weight_timeInstance_before, weight_timeInstance_after

    # print('KL_weight={}\n'.format(KL_we))

    print("有几个GPU：", torch.cuda.device_count())  # 有几个GPU
    print("GPU是否可用：", torch.cuda.is_available())  # GPU是否可用
    writer = SummaryWriter('./logs_train/cross_sub')

    # seed_n = np.random.randint(500)
    seed_n = 100  # 100:
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    # print('\n******************************************* Subject %d *********************************************\n' % (i+1))

    # (2840,5,14,9) (2840,) (2840,)
    data_all, label_all, groups = Datasets().get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_forCopy3_2_Attention_slipWindow()
    # data_all, label_all, groups = Datasets().get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2_for_slipWindow_independent()
    time_begin = time.time()

    ith = 1
    logo = LeaveOneGroupOut()
    test_result_acc = []
    test_result_F1_score = []

    label_allSub = []  # 整个数据集上的真实标签（拼接了每个被试的）
    pred_allSub = []  # 整个数据集上的预测标签


    for trainIndex, testIndex in logo.split(data_all, label_all, groups=groups):
        print(f'\n\nthe {ith}th_LiuYi start ...')
        trans = Trans(writer)   # 留一法 每个测试集重新初始化模型
        # 1、获得训练数据和测试数据，及相应的标签数据
        data_train_all = data_all[trainIndex]  # (60*15,3,14,9)=(900,3,14,9)
        label_train_all = label_all[trainIndex]
        groups1 = np.array(groups)[trainIndex]
        # 训练集和验证集的划分
        logo1 = LeaveOneGroupOut()
        for trainIndex1, testIndex1 in logo1.split(data_train_all, label_train_all, groups=groups1):
            data_train = data_train_all[trainIndex1]  # (50*46,3,14,9)=(2300,3,14,9)
            label_train = label_train_all[trainIndex1]
            data_val = data_train_all[testIndex1]
            label_val = label_train_all[testIndex1]
            break
        data_train = np.array(data_train)
        label_train = np.array(label_train)
        data_val = np.array(data_val)
        label_val = np.array(label_val)
        data_test = data_all[testIndex]
        label_test = label_all[testIndex]
        # 分别打乱训练集和测试集的顺序
        all_shuff_num = np.random.permutation(len(data_train))
        data_train = data_train[all_shuff_num]
        label_train = label_train[all_shuff_num]
        all_shuff_num = np.random.permutation(len(data_val))
        data_val = data_val[all_shuff_num]
        label_val = label_val[all_shuff_num]
        all_shuff_num = np.random.permutation(len(data_test))
        data_test = data_test[all_shuff_num]
        label_test = label_test[all_shuff_num]
        print(np.shape(data_train), np.shape(label_train), np.shape(data_val), np.shape(label_val), np.shape(data_test), np.shape(label_test))

        label_i, pred_i = trans.train(data_train, label_train, data_val, label_val, data_test, label_test, test_result_acc, test_result_F1_score)  #
        label_allSub.append(label_i)
        pred_allSub.append(pred_i)

        # 权重的热力图
        weight_spectral = np.array(weight_spectral.cpu())
        io.savemat('./SCI_N-back_result_data/s{}_weight_spectral.mat'.format(ith), {'weight': weight_spectral})
        weight_temporal = np.array(weight_temporal.cpu())
        io.savemat('./SCI_N-back_result_data/s{}_weight_temporal.mat'.format(ith), {'weight': weight_temporal})
        weight_channel_before = np.array(weight_channel_before.cpu())
        io.savemat('./SCI_N-back_result_data/s{}_weight_channel_before.mat'.format(ith),
                   {'weight': weight_channel_before})
        weight_channel_after = np.array(weight_channel_after.cpu())
        io.savemat('./SCI_N-back_result_data/s{}_weight_channel_after.mat'.format(ith),
                   {'weight': weight_channel_after})
        weight_timeInstance_before = np.array(weight_timeInstance_before.cpu())
        io.savemat('./SCI_N-back_result_data/s{}_weight_timeInstance_before.mat'.format(ith),
                   {'weight': weight_timeInstance_before})
        weight_timeInstance_after = np.array(weight_timeInstance_after.cpu())
        io.savemat('./SCI_N-back_result_data/s{}_weight_timeInstance_after.mat'.format(ith),
                   {'weight': weight_timeInstance_after})

        ith = ith + 1


    avg_test_acc, test_avg_F1_score = np.mean(test_result_acc), np.mean(test_result_F1_score)
    time_end = time.time()
    torch.save(trans.model.state_dict(), 'nBack_class2_sub_independent.pth')

    # 保存每个被试的测试结果数据：ACC、F1、pred_label、real_label
    io.savemat('./SCI_N-back_result_data/acc_allSub.mat', {'acc': np.array(test_result_acc)})
    io.savemat('./SCI_N-back_result_data/f1_allSub.mat', {'f1': np.array(test_result_F1_score)})
    io.savemat('./SCI_N-back_result_data/label_pred_allSub.mat', {'label': np.array(pred_allSub)})
    io.savemat('./SCI_N-back_result_data/label_real_allSub.mat', {'label': np.array(label_allSub)})


    print("共耗时:%.4f" % ((time_end-time_begin)/60), "min")
    print("最终的平均测试精度为: test_avg_acc:%.4f" % avg_test_acc, " test_avg_F1_score:%.4f" % test_avg_F1_score)
    writer.close()

    #     acc_KL.append(avg_test_acc)
    #     f1_KL.append(test_avg_F1_score)
    #
    # io.savemat('./000_acc.mat', {'acc': np.array(acc_KL)})
    # io.savemat('./000_f1.mat', {'f1': np.array(f1_KL)})


    # # 跨库测试
    # data_all_nback, label_all_nback, groups_nback = Datasets().get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2()
    # test_acc, test_loss = trans.test(data_all, data_all_nback, label_all_nback)  # 测试
    # print("在n-back上的测试结果：acc:%.4f" % test_acc, "loss:%.4f" % test_loss)


if __name__ == "__main__":

    main()
