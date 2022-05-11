import numpy as np
import scipy.io
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pandas import Series, DataFrame

class Datasets():
    def __init__(self):
        pass


    # n-back 3分类
    def get_features3D_forNBack_class3(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72].reshape(72,14,-1)  # (72,14,4)
            a1 = tmp['eegraw'][72:].reshape(18,14,-1)  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72].reshape(72,14,-1)  # (72,14,5)
            b1 = tmp['eegraw'][72:].reshape(18,14,-1)  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_mi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72].reshape(72,14,-1)
            a1 = tmp['eegraw'][72:].reshape(18,14,-1)
            tmp = scipy.io.loadmat('./feature_n_back/time/s{}_mi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72].reshape(72,14,-1)
            b1 = tmp['eegraw'][72:].reshape(18,14,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72].reshape(72,14,-1)
            a1 = tmp['eegraw'][72:].reshape(18,14,-1)
            tmp = scipy.io.loadmat('./feature_n_back/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72].reshape(72,14,-1)
            b1 = tmp['eegraw'][72:].reshape(18,14,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (216, 14,9) 训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (54,14,9)  测试集
            c = 3 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (216, ) 训练集标签
            c1 = 3 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (54,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_features3D_forNBack_class3_afterICA(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9) lo训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9) lo测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_mi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_mi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9) mi训练集
            x1_test = np.concatenate((a1, b1), axis=2)  # (18,14,9) mi测试集
            y1 = 2 * np.ones(72)
            y1_test = 2 * np.ones(18)

            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (36,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (144,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (36,)  测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x1_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            y1 = 3 * np.ones(72)
            y1_test = 3 * np.ones(18)

            x = np.concatenate((x, x1), axis=0)  # (216, 14,9) 训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (54,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (216, ) 训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (54,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    # EEGLAB处理的（用了Adjust）with baseline - 效果不好：精度0.4-0.5
    def get_features3D_forNBack_class3_afterICA_EEGLAB(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/psd/s{}_lo.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/time/s{}_lo.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/psd/s{}_mi.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/time/s{}_mi.mat'.format(sub + 1))
            b = tmp['eegraw'][:72].reshape(72,14,-1)
            b1 = tmp['eegraw'][72:].reshape(18,14,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/psd/s{}_hi.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/time/s{}_hi.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (216, 14,9) 训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (54,14,9)  测试集
            c = 3 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (216, ) 训练集标签
            c1 = 3 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (54,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    # EEGLAB处理的（用了Adjust）no baseline - 效果不好：精度0.4-0.5
    def get_features3D_forNBack_class3_afterICA_EEGLAB_noBaseline(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/noBaselineICA/psd/s{}_lo.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/noBaselineICA/time/s{}_lo.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/noBaselineICA/psd/s{}_mi.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/noBaselineICA/time/s{}_mi.mat'.format(sub + 1))
            b = tmp['eegraw'][:72].reshape(72,14,-1)
            b1 = tmp['eegraw'][72:].reshape(18,14,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/noBaselineICA/psd/s{}_hi.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/noBaselineICA/time/s{}_hi.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (216, 14,9) 训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (54,14,9)  测试集
            c = 3 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (216, ) 训练集标签
            c1 = 3 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (54,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test


    # n-back 2分类
    def get_features3D_forNBack_class2(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72].reshape(72,14,-1)  # (72,14,4)
            a1 = tmp['eegraw'][72:].reshape(18,14,-1)  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72].reshape(72,14,-1)  # (72,14,5)
            b1 = tmp['eegraw'][72:].reshape(18,14,-1)  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72].reshape(72,14,-1)
            a1 = tmp['eegraw'][72:].reshape(18,14,-1)
            tmp = scipy.io.loadmat('./feature_n_back/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72].reshape(72,14,-1)
            b1 = tmp['eegraw'][72:].reshape(18,14,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forNBack_class2_afterICA(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forNBack_class2_afterICA_improve2(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (90,14,4)
            tmp = tmp.reshape(3,-1,14,4).transpose(1,0,2,3)  # (30,3,14,4)
            a = tmp[:24]  # (24,3,14,4)
            a1 = tmp[24:]  # 测试集 (6,3,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (18,5,14,4)
            tmp = tmp.reshape(3,-1,14,5).transpose(1,0,2,3)  # (30,3,14,5)
            b = tmp[:24]  # (24,3,14,5)
            b1 = tmp[24:]  # (24,3,14,5)
            x = np.concatenate((a, b), axis=3)  # (24,3,14,9)训练集
            x_test = np.concatenate((a1, b1), axis=3)  # (24,3,14,9)测试集
            y = np.ones(24)  # 训练集标签  (14,)
            y_test = np.ones(6)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']  # (18,5,14,4)
            tmp = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # (30,3,14,4)
            a = tmp[:24]  # (14,5,14,4)
            a1 = tmp[24:]  # 测试集 (4,5,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']  # (18,5,14,4)
            tmp = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)  # (30,3,14,4)
            b = tmp[:24]  # (14,5,14,5)
            b1 = tmp[24:]  # (4,5,14,5)
            x1 = np.concatenate((a, b), axis=3)  # (24,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (48,3,14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=3)  # (6,3,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (12,3,14,9)  测试集
            c = 2 * np.ones(24)
            y = np.concatenate((y, c), axis=0)  # (48,)  训练集标签
            c1 = 2 * np.ones(6)
            y_test = np.concatenate((y_test, c1), axis=0)  # (12,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (448,5,14,9) (448,) (128,5,14,9) (128,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forSTEW_afterICA_matlabHandTest_improve2(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,4)
            tmp = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # (25,3,14,4)
            a = tmp[:20]  # lo阶段的训练集(20,3,14,4)
            a1 = tmp[20:]  # lo阶段的测试集 (5,3,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,5)
            tmp = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)  # (25,3,14,5)
            b = tmp[:20]  # lo阶段的训练集(20,3,14,5)
            b1 = tmp[20:]  # lo阶段的测试集 (5,3,14,5)

            x = np.concatenate((a, b), axis=3)  # (60,14,9) lo训练集
            x_test = np.concatenate((a1, b1), axis=3)  # (15,14,9) lo测试集
            y = np.ones(20)  # lo训练集标签
            y_test = np.ones(5)  # lo测试集标签

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'] # (75,14,4)
            tmp = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # (25,3,14,4)
            a = tmp[:20]  # hi阶段的训练集(60,14,4)
            a1 = tmp[20:]  # hi阶段的测试集 (15,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'] # (75,14,5)
            tmp = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)  # (25,3,14,4)
            b = tmp[:20]  # hi阶段的训练集(60,14,5)
            b1 = tmp[20:]  # hi阶段的测试集 (15,14,5)

            x1 = np.concatenate((a, b), axis=3)  # (60,14,9) hi训练集
            x1_test = np.concatenate((a1, b1), axis=3)  # (15,14,9) hi测试集
            y1 = 2 * np.ones(20)  # hi训练集标签
            y1_test = 2 * np.ones(5)  # hi测试集标签

            x = np.concatenate((x, x1), axis=0)  # (40,3,14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (5,3,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (40,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (10,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        return self.data_train, self.label_train, self.data_test, self.label_test



    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_forCopy3(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(20):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_5/psd/s{}_sit-first_epoch.mat'.format(sub + 1))[
                'eegraw']  # (75,14,4)
            a = tmp.reshape(3, -1, 14, 4).reshape(3, -1, 5, 14, 4).transpose(1, 0, 2, 3, 4)  #
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_5/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']
            b = tmp.reshape(3, -1, 14, 5).reshape(3, -1, 5, 14, 5).transpose(1, 0, 2, 3, 4)  #
            x = np.concatenate((a, b), axis=4)  #
            y = np.ones(5)  # 标签  (90,)

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_5/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            a = tmp.reshape(3, -1, 14, 4).reshape(3, -1, 5, 14, 4).transpose(1, 0, 2, 3, 4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_5/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            b = tmp.reshape(3, -1, 14, 5).reshape(3, -1, 5, 14, 5).transpose(1, 0, 2, 3, 4)
            x1 = np.concatenate((a, b), axis=4)  # (28,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (56,3,14,9)
            c = 2 * np.ones(5)
            y = np.concatenate((y, c), axis=0)  # (56,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(10)*(sub+1)).astype(int)
            groups.extend(bbb)
        return data, label, groups

    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_forCopy3_2(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(20):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,4)
            all1 = np.empty([0, 21, 5, 56])
            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3,25,-1)[0]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i+5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            a = all1.transpose(1, 0, 2, 3).reshape(21,3,5,14,4)  # (21,3,5,14,4)


            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']
            all1 = np.empty([0, 21, 5, 70])
            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[0]  # (25,70)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            b = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 5)  # (21,3,5,14,4)



            x = np.concatenate((a, b), axis=4)  #
            y = np.ones(21)  # 标签  (90,)



            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            all1 = np.empty([0, 21, 5, 56])
            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[0]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            a = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 4)  # (21,3,5,14,4)


            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            all1 = np.empty([0, 21, 5, 70])
            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[0]  # (25,70)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            b = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 5)  # (21,3,5,14,4)


            x1 = np.concatenate((a, b), axis=4)  # (28,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (56,3,14,9)
            c = 2 * np.ones(21)
            y = np.concatenate((y, c), axis=0)  # (56,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(42)*(sub+1)).astype(int)
            groups.extend(bbb)
        return data, label, groups



    def get_3Dfeatures_BiDNN_Liuyi(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(20):
            a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,4)
            b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']
            x = np.concatenate((a, b), axis=2)
            y = np.ones(75)  # 标签

            a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            x1 = np.concatenate((a, b), axis=2)

            x = np.concatenate((x, x1), axis=0)  # (150,14,9)
            c = 2 * np.ones(75)
            y = np.concatenate((y, c), axis=0)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(150)*(sub+1)).astype(int)
            groups.extend(bbb)
        return data, label, groups

    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_forCopy3_2_Attention_slipWindow(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(20):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,4)
            tmp = tmp.reshape(75, -1)  # (75,56)
            a = []
            for i in range(71):
                a.append(tmp[i:i+5])
            a = np.array(a).reshape(71, 5, 14, 4)

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(75, -1)  # (75,70)
            b = []
            for i in range(71):
                b.append(tmp[i:i + 5])
            b = np.array(b).reshape(71, 5, 14, 5)

            x = np.concatenate((a, b), axis=3)  # (75,5,14,9)
            y = np.ones(71)  # 标签  (90,)


            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(75, -1)  # (75,56)
            a = []
            for i in range(71):
                a.append(tmp[i:i + 5])
            a = np.array(a).reshape(71, 5, 14, 4)


            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(75, -1)  # (75,70)
            b = []
            for i in range(71):
                b.append(tmp[i:i + 5])
            b = np.array(b).reshape(71, 5, 14, 5)


            x1 = np.concatenate((a, b), axis=3)
            x = np.concatenate((x, x1), axis=0)  # (142,5,14,9)
            c = 2 * np.ones(71)
            y = np.concatenate((y, c), axis=0)  # (142,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(142)*(sub+1)).astype(int)
            groups.extend(bbb)
        return data, label, groups

    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(20):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (90,14,4)
            tmp = tmp.reshape(75, -1)  # (75,56)
            a = []
            for i in range(71):
                a.append(tmp[i:i + 5])
            a = np.array(a).reshape(71, 5, 14, 4)

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(75, -1)  # (75,70)
            b = []
            for i in range(71):
                b.append(tmp[i:i + 5])
            b = np.array(b).reshape(71, 5, 14, 5)

            x = np.concatenate((a, b), axis=3)  # (75,5,14,9)
            y = np.ones(71)  # 标签  (90,)


            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            a = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            b = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            x1 = np.concatenate((a, b), axis=3)  # (30,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (60,3,14,9)
            c = 2 * np.ones(25)
            y = np.concatenate((y, c), axis=0)  # (60,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(50)*(sub+1)).astype(int)
            groups.extend(bbb)
        return data, label, groups



    def get_3Dfeatures_forNBack_class2_afterICA_improve(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(20):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA5/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (90,14,4)
            tmp = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # (3,30,14,4)->(30,3,14,4)
            a = tmp[:24]  # (24,3,14,4)
            a1 = tmp[24:]  # 测试集 (6,3,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA5/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)  # (3,30,14,5)->(30,3,14,5)
            b = tmp[:24]  # (24,3,14,5)
            b1 = tmp[24:]  # (6,3,14,5)
            x = np.concatenate((a, b), axis=3)  # (24,3,14,9)训练集
            x_test = np.concatenate((a1, b1), axis=3)  # (6,3,14,9)测试集
            y = np.ones(24)  # 训练集标签  (14,)
            y_test = np.ones(6)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA5/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'] # (18,5,14,4)
            tmp = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)
            a = tmp[:24]
            a1 = tmp[24:]  # 测试集
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA5/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']  # (18,5,14,4)
            tmp = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            b = tmp[:24]  #
            b1 = tmp[24:]  #
            x1 = np.concatenate((a, b), axis=3)  # (24,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (48,3,14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=3)  # (6,3,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (12,3,14,9)  测试集
            c = 2 * np.ones(24)
            y = np.concatenate((y, c), axis=0)  # (48,)  训练集标签
            c1 = 2 * np.ones(6)
            y_test = np.concatenate((y_test, c1), axis=0)  # (12,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (448,5,14,9) (448,) (128,5,14,9) (128,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_forCopy3_2_Attention_slipWindow_Dependent(self, sub):

        tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub))['eegraw']  # (75,14,4)
        tmp = tmp.reshape(75, -1)  # (75,56)
        a = []
        for i in range(71):
            a.append(tmp[i:i+5])
        a = np.array(a).reshape(71, 5, 14, 4)

        tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub))['eegraw']
        tmp = tmp.reshape(75, -1)  # (75,70)
        b = []
        for i in range(71):
            b.append(tmp[i:i + 5])
        b = np.array(b).reshape(71, 5, 14, 5)


        x = np.concatenate((a, b), axis=3)  # (71,5,14,9)
        a_train = x[:int(0.7*len(x))]  # 训练集
        a_test = x[int(0.7*len(x)):]  # 测试集
        y1_train = np.ones(int(0.7*len(x)))  # 标签
        y1_test = np.ones(len(x)-int(0.7*len(x)))


        tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub))['eegraw']
        tmp = tmp.reshape(75, -1)  # (75,56)
        a = []
        for i in range(71):
            a.append(tmp[i:i + 5])
        a = np.array(a).reshape(71, 5, 14, 4)

        tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub))['eegraw']
        tmp = tmp.reshape(75, -1)  # (75,70)
        b = []
        for i in range(71):
            b.append(tmp[i:i + 5])
        b = np.array(b).reshape(71, 5, 14, 5)


        x1 = np.concatenate((a, b), axis=3)
        b_train = x1[:int(0.7*len(x1))]
        b_test = x1[int(0.7*len(x1)):]
        y2_train = 2 * np.ones(int(0.7*len(x1)))
        y2_test = 2 * np.ones(len(x1)-int(0.7*len(x1)))

        x_train = np.concatenate((a_train, b_train), axis=0)  # (98,5,14,9)
        x_test = np.concatenate((a_test, b_test), axis=0)  # (44,5,14,9)
        y_train = np.concatenate((y1_train, y2_train), axis=0)  # (98,)
        y_test = np.concatenate((y1_test, y2_test), axis=0)  # (44,)

        return x_train, y_train, x_test, y_test

    def get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2_for_slipWindow_independent(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  #
            tmp = tmp.reshape(75, -1)  # (75,56)
            a = []
            for i in range(71):
                a.append(tmp[i:i + 5])
            a = np.array(a).reshape(71, 5, 14, 4)

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(75, -1)  # (75,70)
            b = []
            for i in range(71):
                b.append(tmp[i:i + 5])
            b = np.array(b).reshape(71, 5, 14, 5)

            x = np.concatenate((a, b), axis=3)  # (75,5,14,9)
            y = np.ones(71)  # 标签  (90,)



            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(75, -1)  # (75,56)
            a = []
            for i in range(71):
                a.append(tmp[i:i + 5])
            a = np.array(a).reshape(71, 5, 14, 4)

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(75, -1)  # (75,70)
            b = []
            for i in range(71):
                b.append(tmp[i:i + 5])
            b = np.array(b).reshape(71, 5, 14, 5)

            x1 = np.concatenate((a, b), axis=3)
            x = np.concatenate((x, x1), axis=0)  # (142,5,14,9)
            c = 2 * np.ones(71)
            y = np.concatenate((y, c), axis=0)  # (142,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(142) * (sub + 1)).astype(int)
            groups.extend(bbb)
        return data, label, groups


    def get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2_for_slipWindow_dependent(self, sub):

        tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub))['eegraw']  #
        tmp = tmp.reshape(75, -1)  # (75,56)
        a = []
        for i in range(71):
            a.append(tmp[i:i + 5])
        a = np.array(a).reshape(71, 5, 14, 4)

        tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub))['eegraw']
        tmp = tmp.reshape(75, -1)  # (75,70)
        b = []
        for i in range(71):
            b.append(tmp[i:i + 5])
        b = np.array(b).reshape(71, 5, 14, 5)

        x = np.concatenate((a, b), axis=3)  # (71,5,14,9)
        x1_train = x[:int(0.7*len(x))]
        x1_test = x[int(0.7*len(x)):]
        y1_train = np.ones(int(0.7*len(x)))
        y1_test = np.ones(len(x)-int(0.7*len(x)))


        tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub))['eegraw']
        tmp = tmp.reshape(75, -1)  # (75,56)
        a = []
        for i in range(71):
            a.append(tmp[i:i + 5])
        a = np.array(a).reshape(71, 5, 14, 4)

        tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub))['eegraw']
        tmp = tmp.reshape(75, -1)  # (75,70)
        b = []
        for i in range(71):
            b.append(tmp[i:i + 5])
        b = np.array(b).reshape(71, 5, 14, 5)

        x1 = np.concatenate((a, b), axis=3)
        x2_train = x1[:int(0.7*len(x1))]
        x2_test = x1[int(0.7*len(x1)):]
        y2_train = 2 * np.ones(int(0.7*len(x1)))
        y2_test = 2 * np.ones(len(x1)-int(0.7*len(x1)))

        x_train = np.concatenate((x1_train, x2_train), axis=0)  # (142,5,14,9)
        x_test = np.concatenate((x1_test, x2_test), axis=0)
        y_train = np.concatenate((y1_train, y2_train), axis=0)
        y_test = np.concatenate((y1_test, y2_test), axis=0)

        return x_train, y_train, x_test, y_test


    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_forCopy3_2_Attention(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(20):
            a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,4)
            # all1 = np.empty([0, 21, 5, 56])
            # tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3,25,-1)[0]  # (25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i+5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[1]  # (25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[2]  # (3,25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)
            # a = all1.transpose(1, 0, 2, 3).reshape(21,3,5,14,4)  # (21,3,5,14,4)
            a = a.reshape(-1,5,14,4)  # (15,5,14,4)
            b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,5)
            # all1 = np.empty([0, 21, 5, 70])
            # tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[0]  # (25,70)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[1]  # (25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[2]  # (3,25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)
            # x = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 5)  # (21,3,5,14,4)
            b = b.reshape(-1,5,14,5)  # (15,5,14,5)
            x = np.concatenate((a, b), axis=3)  # (15,5,14,9)
            y = np.ones(15)  # 标签  (15,)

            a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            # all1 = np.empty([0, 21, 5, 56])
            # tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[0]  # (25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[1]  # (25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[2]  # (3,25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)
            #
            # a = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 4)  # (21,3,5,14,4)
            a = a.reshape(-1, 5, 14, 4)  # (15,5,14,4)
            b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            # all1 = np.empty([0, 21, 5, 70])
            # tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[0]  # (25,70)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[1]  # (25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)
            #
            # tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[2]  # (3,25,56)
            # all_tmp = []
            # for i in range(21):
            #     all_tmp.append(tmp1[i:i + 5])
            # all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)
            #
            # b = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 5)  # (21,3,5,14,4)
            b = b.reshape(-1, 5, 14, 5)  # (15,5,14,5)
            x1 = np.concatenate((a, b), axis=3)  # (15,5,14,9)
            x = np.concatenate((x, x1), axis=0)  # (30,5,14,9)
            c = 2 * np.ones(15)
            y = np.concatenate((y, c), axis=0)  # (30,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(30)*(sub+1)).astype(int)
            groups.extend(bbb)
        return data, label, groups

    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_forCopy3_2_ccc(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(20):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,4)
            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3,25,-1)[0]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i+5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            a = all1.transpose(1, 0, 2, 3).reshape(21,3,5,14,4)  # (21,3,5,14,4)


            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']
            all1 = np.empty([0, 21, 5, 70])
            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[0]  # (25,70)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            b = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 5)  # (21,3,5,14,4)



            x = np.concatenate((a, b), axis=4)  #
            y = np.ones(21)  # 标签  (90,)



            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            all1 = np.empty([0, 21, 5, 56])
            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[0]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 4).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            a = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 4)  # (21,3,5,14,4)


            tmp = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            all1 = np.empty([0, 21, 5, 70])
            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[0]  # (25,70)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[1]  # (25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)

            tmp1 = tmp.reshape(3, -1, 14, 5).reshape(3, 25, -1)[2]  # (3,25,56)
            all_tmp = []
            for i in range(21):
                all_tmp.append(tmp1[i:i + 5])
            all1 = np.append(all1, [all_tmp], axis=0)  # (3,21,5,56)

            b = all1.transpose(1, 0, 2, 3).reshape(21, 3, 5, 14, 5)  # (21,3,5,14,4)


            x1 = np.concatenate((a, b), axis=4)  # (28,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (56,3,14,9)
            c = 2 * np.ones(21)
            y = np.concatenate((y, c), axis=0)  # (56,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(42)*(sub+1)).astype(int)
            groups.extend(bbb)
        return data, label, groups

    def get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2_CCC(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        groups = []
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))[
                'eegraw']  # (75,14,4)
            a = tmp.reshape(-1, 5, 14, 4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))[
                'eegraw']
            b = tmp.reshape(-1, 5, 14, 5)
            x = np.concatenate((a, b), axis=3)  # (15,5,14,9)
            y = np.ones(15)  # 标签  (90,)

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            a = tmp.reshape(-1, 5, 14, 4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            b = tmp.reshape(-1, 5, 14, 5)
            x1 = np.concatenate((a, b), axis=3)  # (15,5,14,9)
            x = np.concatenate((x, x1), axis=0)  # (30,5,14,9)
            c = 2 * np.ones(15)
            y = np.concatenate((y, c), axis=0)  # (60,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(30) * (sub + 1)).astype(int)
            groups.extend(bbb)
        return data, label, groups


    def get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2_forCopy3(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))  #
            a = tmp['eegraw'].reshape(3, -1, 14, 4).reshape(3, -1, 5, 14, 4).transpose(1, 0, 2, 3, 4)  # (75,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'].reshape(3, -1, 14, 5).reshape(3, -1, 5, 14, 5).transpose(1, 0, 2, 3, 4) # (75,14,5)
            x = np.concatenate((a, b), axis=4)  # (75,14,9)
            y = np.ones(5)  # 标签  (75,)

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'].reshape(3, -1, 14, 4).reshape(3, -1, 5, 14, 4).transpose(1, 0, 2, 3, 4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'].reshape(3, -1, 14, 5).reshape(3, -1, 5, 14, 5).transpose(1, 0, 2, 3, 4)
            x1 = np.concatenate((a, b), axis=4)  # (25,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (50,3,14,9)
            c = 2 * np.ones(5)
            y = np.concatenate((y, c), axis=0)  # (50,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(10)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups




    def get_3Dfeatures_forNBack_class2_afterICA_subDependent_improve2_for_Contrast_timeSeq(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(20):
            a = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # (90,14,4)
            a1 = a[:int(len(a)*0.8)]  # (24,3,14,4)
            a2 = a[int(len(a)*0.8):]
            b = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            b1 = b[:int(len(b)*0.8)]
            b2 = b[int(len(b)*0.8):]
            x = np.concatenate((a1, b1), axis=3)  # (24,14,9)
            x_test = np.concatenate((a2, b2), axis=3)  # (6,14,9)
            y = np.ones(24)  # 标签  (90,)
            y_test = np.ones(6)

            a = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)
            a1 = a[:int(len(a)*0.8)]  # (24,14,4)
            a2 = a[int(len(a)*0.8):]
            b = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            b1 = b[:int(len(b)*0.8)]
            b2 = b[int(len(b)*0.8):]
            x1 = np.concatenate((a1, b1), axis=3)  # (24,14,9)
            x1_test = np.concatenate((a2, b2), axis=3)  # (6,14,9)
            x = np.concatenate((x, x1), axis=0)  # (48,14,9)
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (12,14,9)
            c = 2 * np.ones(24)
            c_test = 2 * np.ones(6)
            y = np.concatenate((y, c), axis=0)  # (48,)
            y_test = np.concatenate((y_test, c_test), axis=0)  # (12,)

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)
        return data_train, label_train, data_test, label_test

    # (b,14,9)
    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve2_for_Contrast(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(20):
            # a = scipy.io.loadmat('./feature_n_back/after_ICA21/psd/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']  # (90,14,4)
            a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub + 1))[
                'eegraw']  # (90,14,4)
            # a = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # ->(3,30,14,4)->(30,3,14,4)
            b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub + 1))['eegraw']
            # b = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)  # (30,3,14,4)
            x = np.concatenate((a, b), axis=2)  # (90,14,9)
            y = np.ones(75)  # 标签  (90,)

            a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            # a = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)
            b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            # b = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            x1 = np.concatenate((a, b), axis=2)  # (90,14,9)
            x = np.concatenate((x, x1), axis=0)  # (180,14,9)
            c = 2 * np.ones(75)
            y = np.concatenate((y, c), axis=0)  # (180,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(150)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups


    # (b,14,9)
    def get_3Dfeatures_forNBack_class2_afterICA_subDependent_improve2_for_Contrast(self, sub):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        # for sub in range(20):
        a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_sit-first_epoch.mat'.format(sub))['eegraw']  # (90,14,4)
        a1 = a[:int(len(a)*(2/3))]  # (72,14,4)
        a2 = a[int(len(a)*(2/3)):]
        b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_sit-first_epoch.mat'.format(sub))['eegraw']
        b1 = b[:int(len(b)*(2/3))]
        b2 = b[int(len(b)*(2/3)):]
        x = np.concatenate((a1, b1), axis=2)  # (72,14,9)
        x_test = np.concatenate((a2, b2), axis=2)  # (18,14,9)
        y = np.ones(int(len(b)*(2/3)))  # 标签  (90,)
        y_test = np.ones(len(b)-int(len(b)*(2/3)))

        a = scipy.io.loadmat('./feature_n_back/after_ICA15_6/psd/s{}_hi_epoch.mat'.format(sub))['eegraw']
        a1 = a[:int(len(a)*(2/3))]  # (72,14,4)
        a2 = a[int(len(a)*(2/3)):]
        b = scipy.io.loadmat('./feature_n_back/after_ICA15_6/time/s{}_hi_epoch.mat'.format(sub))['eegraw']
        b1 = b[:int(len(b)*(2/3))]
        b2 = b[int(len(b)*(2/3)):]
        x1 = np.concatenate((a1, b1), axis=2)  # (72,14,9)
        x1_test = np.concatenate((a2, b2), axis=2)  # (18,14,9)
        x = np.concatenate((x, x1), axis=0)  # (144,14,9)
        x_test = np.concatenate((x_test, x1_test), axis=0)  # (36,14,9)
        c = 2 * np.ones(int(len(b)*(2/3)))
        c_test = 2 * np.ones(len(b)-int(len(b)*(2/3)))
        y = np.concatenate((y, c), axis=0)  # (144,)
        y_test = np.concatenate((y_test, c_test), axis=0)  # (36,)

        # if sub == 0:
        data_train = x
        data_test = x_test
        label_train = y
        label_test = y_test
            # else:
            #     data_train = np.concatenate((data_train, x), axis=0)
            #     data_test = np.concatenate((data_test, x_test), axis=0)
            #     label_train = np.concatenate((label_train, y), axis=0)
            #     label_test = np.concatenate((label_test, y_test), axis=0)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return data_train, label_train, data_test, label_test


    def get_3Dfeatures_forSTEW_afterICA_matlabHandTest_improve(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,4)
            tmp = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # (3,25,14,4)->(25,3,14,4)
            a = tmp[:20]  # lo阶段的训练集
            a1 = tmp[20:]  # lo阶段的测试集
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (75,14,5)
            tmp = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            b = tmp[:20]  # lo阶段的训练集
            b1 = tmp[20:]  # lo阶段的测试集

            x = np.concatenate((a, b), axis=3)  # (25,3,14,9) lo训练集
            x_test = np.concatenate((a1, b1), axis=3)  # (5,3,14,9) lo测试集
            y = np.ones(20)  # lo训练集标签
            y_test = np.ones(5)  # lo测试集标签

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)
            a = tmp[:20]  # hi阶段的训练集
            a1 = tmp[20:]  # hi阶段的测试集
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            tmp = tmp.reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            b = tmp[:20]  # hi阶段的训练集
            b1 = tmp[20:]  # hi阶段的测试集

            x1 = np.concatenate((a, b), axis=3)  # hi训练集
            x1_test = np.concatenate((a1, b1), axis=3)  #  hi测试集
            y1 = 2 * np.ones(20)  # hi训练集标签
            y1_test = 2 * np.ones(5)  # hi测试集标签

            x = np.concatenate((x, x1), axis=0)  # (40,3,14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (10,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (40,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (10,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))  #
            a = tmp['eegraw'].reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)  # (75,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'].reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)  # (75,14,5)
            x = np.concatenate((a, b), axis=3)  # (75,14,9)
            y = np.ones(25)  # 标签  (75,)

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'].reshape(3, -1, 14, 4).transpose(1, 0, 2, 3)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'].reshape(3, -1, 14, 5).transpose(1, 0, 2, 3)
            x1 = np.concatenate((a, b), axis=3)  # (25,3,14,9)
            x = np.concatenate((x, x1), axis=0)  # (50,3,14,9)
            c = 2 * np.ones(25)
            y = np.concatenate((y, c), axis=0)  # (50,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(50)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups


    def get_3Dfeatures_forSTEW_class2_afterICA_Liuyi_improve2_for_Contrast(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(48):
            a = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (90,14,4)
            b = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']
            x = np.concatenate((a, b), axis=2)  # (90,14,9)
            y = np.ones(75)  # 标签  (90,)

            a = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            b = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']
            x1 = np.concatenate((a, b), axis=2)  # (90,14,9)
            x = np.concatenate((x, x1), axis=0)  # (180,14,9)
            c = 2 * np.ones(75)
            y = np.concatenate((y, c), axis=0)  # (180,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(150)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups


    def get_3Dfeatures_forSTEW_class2_afterICA_subDependent_improve2_for_Contrast(self, sub):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        # for sub in range(48):
        a = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub))['eegraw']  # (90,14,4)
        a1 = a[:int(len(a)*0.7)]  # (60,14,4)
        a2 = a[int(len(a)*0.7):]  # (15,14,4)
        b = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub))['eegraw']
        b1 = b[:int(len(b)*0.7)]
        b2 = b[int(len(b)*0.7):]
        x = np.concatenate((a1, b1), axis=2)  # (60,14,9)
        x_test = np.concatenate((a2, b2), axis=2)  # (15,14,9)
        y = np.ones(int(len(b)*0.7))  # 标签  (90,)
        y_test = np.ones(len(b)-int(len(b)*0.7))

        a = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub))['eegraw']
        a1 = a[:int(len(a)*0.7)]  # (60,14,5)
        a2 = a[int(len(a)*0.7):]  # (15,14,5)
        b = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub))['eegraw']
        b1 = b[:int(len(b)*0.7)]
        b2 = b[int(len(b)*0.7):]
        x1 = np.concatenate((a1, b1), axis=2)  # (60,14,9)
        x1_test = np.concatenate((a2, b2), axis=2)  # (15,14,9)
        x = np.concatenate((x, x1), axis=0)  # (120,14,9)
        x_test = np.concatenate((x_test, x1_test), axis=0)  # (30,14,9)
        c = 2 * np.ones(int(len(b)*0.7))
        c_test = 2 * np.ones(len(b)-int(len(b)*0.7))
        y = np.concatenate((y, c), axis=0)  # (120,)
        y_test = np.concatenate((y_test, c_test), axis=0)  # (30,)

        # if sub == 0:
        data_train = x
        data_test = x_test
        label_train = y
        label_test = y_test
        # else:
        #     data_train = np.concatenate((data_train, x), axis=0)
        #     label_train = np.concatenate((label_train, y), axis=0)
        #     data_test = np.concatenate((data_test, x_test), axis=0)
        #     label_test = np.concatenate((label_test, y_test), axis=0)

        return data_train, label_train, data_test, label_test

    def get_3Dfeatures_forNBack_class3_afterICA_improve(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)  # (18,5,14,4)
            a = tmp[:14]  # (14,5,14,4)
            a1 = tmp[14:]  # 测试集 (4,5,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)  # (18,5,14,4)
            b = tmp[:14]  # (14,5,14,5)
            b1 = tmp[14:]  # (4,5,14,5)
            x = np.concatenate((a, b), axis=3)  # (14,5,14,9)训练集
            x_test = np.concatenate((a1, b1), axis=3)  # (4,5,14,9)测试集
            y = np.ones(14)  # 训练集标签  (14,)
            y_test = np.ones(4)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_mi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)  # (18,5,14,4)
            a = tmp[:14]  # (14,5,14,4)
            a1 = tmp[14:]  # 测试集 (4,5,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_mi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)  # (18,5,14,4)
            b = tmp[:14]  # (14,5,14,5)
            b1 = tmp[14:]  # (4,5,14,5)
            x1 = np.concatenate((a, b), axis=3)  # (14,5,14,9)
            x = np.concatenate((x, x1), axis=0)  # (28,5,14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=3)  # (4,5,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (8,5,14,9)  测试集
            c = 2 * np.ones(14)
            y = np.concatenate((y, c), axis=0)  # (28,)  训练集标签
            c1 = 2 * np.ones(4)
            y_test = np.concatenate((y_test, c1), axis=0)  # (8,)  测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)  # (18,5,14,4)
            a = tmp[:14]  # (14,5,14,4)
            a1 = tmp[14:]  # 测试集 (4,5,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)  # (18,5,14,4)
            b = tmp[:14]  # (14,5,14,5)
            b1 = tmp[14:]  # (4,5,14,5)
            x1 = np.concatenate((a, b), axis=3)  # (14,5,14,9)
            x = np.concatenate((x, x1), axis=0)  # (28,5,14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=3)  # (4,5,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (8,5,14,9)  测试集
            c = 3 * np.ones(14)
            y = np.concatenate((y, c), axis=0)  # (28,)  训练集标签
            c1 = 3 * np.ones(4)
            y_test = np.concatenate((y_test, c1), axis=0)  # (8,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forSTEW_class2_afterICA_improve(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)  # (18,5,14,4)
            a = tmp[:14]  # (14,5,14,4)
            a1 = tmp[14:]  # 测试集 (4,5,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)  # (18,5,14,4)
            b = tmp[:14]  # (14,5,14,5)
            b1 = tmp[14:]  # (4,5,14,5)
            x = np.concatenate((a, b), axis=3)  # (14,5,14,9)训练集
            x_test = np.concatenate((a1, b1), axis=3)  # (4,5,14,9)测试集
            y = np.ones(14)  # 训练集标签  (14,)
            y_test = np.ones(4)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)  # (18,5,14,4)
            a = tmp[:14]  # (14,5,14,4)
            a1 = tmp[14:]  # 测试集 (4,5,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)  # (18,5,14,4)
            b = tmp[:14]  # (14,5,14,5)
            b1 = tmp[14:]  # (4,5,14,5)
            x1 = np.concatenate((a, b), axis=3)  # (14,5,14,9)
            x = np.concatenate((x, x1), axis=0)  # (28,5,14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=3)  # (4,5,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (8,5,14,9)  测试集
            c = 2 * np.ones(14)
            y = np.concatenate((y, c), axis=0)  # (28,)  训练集标签
            c1 = 2 * np.ones(4)
            y_test = np.concatenate((y_test, c1), axis=0)  # (8,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (448,5,14,9) (448,) (128,5,14,9) (128,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forSTEW_no_ICA_matlabHandTest_improve(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)  # (15,5,14,4)
            a = tmp[:12]  # lo阶段的训练集(60,5,14,4)
            a1 = tmp[12:]  # lo阶段的测试集 (15,5,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)  # (15,5,14,5)
            b = tmp[:12]  # lo阶段的训练集(60,5,14,5)
            b1 = tmp[12:]  # lo阶段的测试集 (15,5,14,5)
            x = np.concatenate((a, b), axis=3)  # (60,5,14,9) lo训练集
            x_test = np.concatenate((a1, b1), axis=3)  # (15,5,14,9) lo测试集
            y = np.ones(12)  # lo训练集标签
            y_test = np.ones(3)  # lo测试集标签

            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)  # (75,14,4)
            a = tmp[:12]  # hi阶段的训练集(60,5,14,4)
            a1 = tmp[12:]  # hi阶段的测试集 (15,5,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)  # (75,14,5)
            b = tmp[:12]  # hi阶段的训练集(60,5,14,5)
            b1 = tmp[12:]  # hi阶段的测试集 (15,5,14,5)
            x1 = np.concatenate((a, b), axis=3)  # (60,5,14,9) hi训练集
            x1_test = np.concatenate((a1, b1), axis=3)  # (15,5,14,9) hi测试集
            y1 = 2 * np.ones(12)  # hi训练集标签
            y1_test = 2 * np.ones(3)  # hi测试集标签
            x = np.concatenate((x, x1), axis=0)  # (120,5,14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (30,5,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (120,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (30,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        return self.data_train, self.label_train, self.data_test, self.label_test



    def get_3Dfeatures_forNBack_class2_no_ICA_by_hand(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/no_ICA_by_hand/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/no_ICA_by_hand/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)lo训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)lo测试集
            y = np.ones(72)  # lo训练集标签  (72,)
            y_test = np.ones(18)  # lo测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/no_ICA_by_hand/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/no_ICA_by_hand/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9) hi训练集
            x1_test = np.concatenate((a1, b1), axis=2)  # (18,14,9) hi测试集
            y1 = 2 * np.ones(72)
            y1_test = 2 * np.ones(18)

            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (36,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (144,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forNBack_class2_ICA_by_hand(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)lo训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)lo测试集
            y = np.ones(72)  # lo训练集标签  (72,)
            y_test = np.ones(18)  # lo测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9) hi训练集
            x1_test = np.concatenate((a1, b1), axis=2)  # (18,14,9) hi测试集
            y1 = 2 * np.ones(72)
            y1_test = 2 * np.ones(18)

            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (36,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (144,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forNBack_class2_afterICA_matlabHandTest(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand_coding/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand_coding/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand_coding/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand_coding/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test = data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw']  # (90,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']  # (90,14,5)
            x = np.concatenate((a, b), axis=2)  # (90,14, 9)
            y = np.ones(90)  # 标签  (90,)

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw']
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']
            x1 = np.concatenate((a, b), axis=2)  # (90,14,9)
            x = np.concatenate((x, x1), axis=0)  # (180, 14,9)
            c = 2 * np.ones(90)
            y = np.concatenate((y, c), axis=0)  # (180,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(180)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups

    def get_3Dfeatures_forNBack_class2_afterICA_Liuyi_improve(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)
            a = tmp  # (18,5,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)
            b = tmp  # (18,5,14,5)
            x = np.concatenate((a, b), axis=3)  # (90,5,14,9)
            y = np.ones(18)  # 标签  (90,)

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,4)
            a = tmp
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw'].reshape(-1,5,14,5)
            b = tmp
            x1 = np.concatenate((a, b), axis=3)  # (90,5,14,9)
            x = np.concatenate((x, x1), axis=0)  # (180,5,14,9)
            c = 2 * np.ones(18)
            y = np.concatenate((y, c), axis=0)  # (180,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(36)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups


    def get_3Dfeatures_forNBack_class3_afterICA_Liuyi(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw']  # (90,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']  # (90,14,5)
            x = np.concatenate((a, b), axis=2)  # (90,14, 9)
            y = np.ones(90)  # 标签  (90,)

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_mi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw']
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_mi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']
            x1 = np.concatenate((a, b), axis=2)  # (90,14,9)
            x = np.concatenate((x, x1), axis=0)  # (180, 14,9)
            c = 2 * np.ones(90)
            y = np.concatenate((y, c), axis=0)  # (180,)

            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw']
            tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']
            x1 = np.concatenate((a, b), axis=2)  # (90,14,9)
            x = np.concatenate((x, x1), axis=0)  # (270, 14,9)
            c = 3 * np.ones(90)
            y = np.concatenate((y, c), axis=0)  # (270,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(270)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (4050, 14, 9)
        self.label = label  # (4050,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups

    def get_3Dfeatures_forNBack_class3_afterICA_matlabHandTest(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand_coding/psd/s{}_3stage_epoch.mat'.format(sub + 1))  # (270,14,4)
            a = tmp['eegraw'][:72]  # lo阶段的训练集(72,14,4)
            a1 = tmp['eegraw'][72:90]  # lo阶段的测试集 (18,14,4)
            e = tmp['eegraw'][90:162]
            e1 = tmp['eegraw'][162:180]
            c = tmp['eegraw'][180:252]  # hi阶段的训练集(72,14,4)
            c1 = tmp['eegraw'][252:]  # hi阶段的测试集(18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/ICA_by_hand_coding/time/s{}_3stage_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # lo阶段的训练集(72,14,5)
            b1 = tmp['eegraw'][72:90]  # lo阶段的测试集 (18,14,5)
            f = tmp['eegraw'][90:162]
            f1 = tmp['eegraw'][162:180]
            d = tmp['eegraw'][180:252]  # hi阶段的训练集(72,14,5)
            d1 = tmp['eegraw'][252:]  # hi阶段的测试集(18,14,5)

            x = np.concatenate((a, b), axis=2)  # (72,14, 9)lo训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)lo测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            x1 = np.concatenate((e, f), axis=2)  # (72,14, 9)mi训练集
            x1_test = np.concatenate((e1, f1), axis=2)  # (18,14,9)mi测试集
            y1 = 2 * np.ones(72)  # 训练集标签  (72,)
            y1_test = 2 * np.ones(18)  # 测试集标签

            x = np.concatenate((x, x1), axis=0)  # (144,14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (36,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (144,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (36,)  测试集标签

            x1 = np.concatenate((c, d), axis=2)  # (72,14, 9)hi训练集
            x1_test = np.concatenate((c1, d1), axis=2)  # (18,14,9)hi测试集
            y1 = 3 * np.ones(72)
            y1_test = 3 * np.ones(18)

            x = np.concatenate((x, x1), axis=0)  # (216,14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (54,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (216,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (54,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forNBack_class2_afterICA_EEGLAB(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/psd/s{}_lo.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]  # (72,14,4)
            a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/time/s{}_lo.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]  # (72,14,5)
            b1 = tmp['eegraw'][72:]  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/psd/s{}_hi.mat'.format(sub + 1))
            a = tmp['eegraw'][:72]
            a1 = tmp['eegraw'][72:]
            tmp = scipy.io.loadmat('./feature_n_back/ICA_EEGLAB/time/s{}_hi.mat'.format(sub + 1))
            b = tmp['eegraw'][:72]
            b1 = tmp['eegraw'][72:]
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    # n_back 2分类  特征按被试进行归一化
    def get_features3D_forNBack_class2_zscore(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(16):
            tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (90,56)
            tmp1 = scipy.io.loadmat('./feature_n_back/time/s{}_lo_epoch.mat'.format(sub + 1))['eegraw']  # (90,70)
            data = np.concatenate((tmp, tmp1), axis=1)  # (90,126)
            tmp2 = scipy.io.loadmat('./feature_n_back/psd/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']  # (90,56)
            tmp3 = scipy.io.loadmat('./feature_n_back/time/s{}_hi_epoch.mat'.format(sub + 1))['eegraw']  # (90,70)
            data1 = np.concatenate((tmp2, tmp3), axis=1)  # (90,126)
            data = np.concatenate((data, data1), axis=0)  # (180,126)  一个被试总的特征矩阵
            # z-score
            scaler = StandardScaler()
            data = scaler.fit_transform(data)  # (126,180) 对每个样本的126维特征向量进行归一化
            # data = data.transpose()  # (180,126)
            data_lo = data[:90, :]  # lo (90,126)
            data_hi = data[90:, :]  # hi (90,126)
            data_lo_psd = data_lo[:, :56]  # (90,56)
            data_lo_time = data_lo[:, 56:]  # (90,70)
            data_hi_psd = data_hi[:, :56]  # (90,56)
            data_hi_time = data_hi[:, 56:]  # (90,70)

            # tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = data_lo_psd[:72].reshape(72,14,-1)  # (72,14,4)
            a1 = data_lo_psd[72:].reshape(18,14,-1)  # 测试集 (18,14,4)
            # tmp = scipy.io.loadmat('./feature_n_back/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = data_lo_time[:72].reshape(72,14,-1)  # (72,14,5)
            b1 = data_lo_time[72:].reshape(18,14,-1)  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(72)  # 训练集标签  (72,)
            y_test = np.ones(18)  # 测试集标签

            # tmp = scipy.io.loadmat('./feature_n_back/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = data_hi_psd[:72].reshape(72,14,-1)
            a1 = data_hi_psd[72:].reshape(18,14,-1)
            # tmp = scipy.io.loadmat('./feature_n_back/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = data_hi_time[:72].reshape(72,14,-1)
            b1 = data_hi_time[72:].reshape(18,14,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(72)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(18)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    # n-back 2分类 ICA -- 被试依赖  这里只读取特征矩阵，不做标准化
    def get_features3D_forNBack_class2_eachSub(self, sub):
        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub))
        a = tmp['eegraw'][:72]  # (72,14,4)
        a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub))
        b = tmp['eegraw'][:72]  # (72,14,5)
        b1 = tmp['eegraw'][72:]  # (18,14,5)
        x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
        x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
        y = np.ones(72)  # 训练集标签  (72,)
        y_test = np.ones(18)  # 测试集标签

        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub))
        a = tmp['eegraw'][:72]
        a1 = tmp['eegraw'][72:]
        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub))
        b = tmp['eegraw'][:72]
        b1 = tmp['eegraw'][72:]
        x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
        x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
        x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
        x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
        c = 2 * np.ones(72)
        y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
        c1 = 2 * np.ones(18)
        y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

        self.data_train = x  # (4320, 126)
        self.label_train = y
        self.data_test = x_test
        self.label_test = y_test  # (4320,)

        print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(self.label_test))
        # (144, 14, 9) (144,) (36, 14, 9) (36,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_features3D_forNBack_class3_eachSub(self, sub):
        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_lo_epoch.mat'.format(sub))
        a = tmp['eegraw'][:72]  # (72,14,4)
        a1 = tmp['eegraw'][72:]  # 测试集 (18,14,4)
        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_lo_epoch.mat'.format(sub))
        b = tmp['eegraw'][:72]  # (72,14,5)
        b1 = tmp['eegraw'][72:]  # (18,14,5)
        x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
        x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
        y = np.ones(72)  # 训练集标签  (72,)
        y_test = np.ones(18)  # 测试集标签

        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_mi_epoch.mat'.format(sub))
        a = tmp['eegraw'][:72]
        a1 = tmp['eegraw'][72:]
        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_mi_epoch.mat'.format(sub))
        b = tmp['eegraw'][:72].reshape(72, 14, -1)
        b1 = tmp['eegraw'][72:].reshape(18, 14, -1)
        x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
        x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
        x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
        x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
        c = 2 * np.ones(72)
        y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
        c1 = 2 * np.ones(18)
        y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/psd/s{}_hi_epoch.mat'.format(sub))
        a = tmp['eegraw'][:72]
        a1 = tmp['eegraw'][72:]
        tmp = scipy.io.loadmat('./feature_n_back/after_ICA/time/s{}_hi_epoch.mat'.format(sub))
        b = tmp['eegraw'][:72]
        b1 = tmp['eegraw'][72:]
        x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
        x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
        x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
        x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
        c = 3 * np.ones(72)
        y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
        c1 = 3 * np.ones(18)
        y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

        self.data_train = x  # (4320, 126)
        self.label_train = y
        self.data_test = x_test
        self.label_test = y_test  # (4320,)

        print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(self.label_test))
        # (144, 14, 9) (144,) (36, 14, 9) (36,)
        return self.data_train, self.label_train, self.data_test, self.label_test



    # STEW 2分类
    def get_features3D_forSTEW(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(45):
            tmp = scipy.io.loadmat('./feature_STEW/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:60].reshape(60,14,-1)  # (72,14,4)
            a1 = tmp['eegraw'][60:].reshape(15,14,-1)  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:60].reshape(60,14,-1)  # (72,14,5)
            b1 = tmp['eegraw'][60:].reshape(15,14,-1)  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(60)  # 训练集标签  (72,)
            y_test = np.ones(15)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_STEW/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:60].reshape(60,14,-1)
            a1 = tmp['eegraw'][60:].reshape(15,14,-1)
            tmp = scipy.io.loadmat('./feature_STEW/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:60].reshape(60,14,-1)
            b1 = tmp['eegraw'][60:].reshape(15,14,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(60)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(15)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    def get_3Dfeatures_forSTEW_afterICA_matlabHandTest(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))  # (75,14,4)
            a = tmp['eegraw'][:60]  # lo阶段的训练集(60,14,4)
            a1 = tmp['eegraw'][60:]  # lo阶段的测试集 (15,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))  # (75,14,5)
            b = tmp['eegraw'][:60]  # lo阶段的训练集(60,14,5)
            b1 = tmp['eegraw'][60:]  # lo阶段的测试集 (15,14,5)

            x = np.concatenate((a, b), axis=2)  # (60,14,9) lo训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (15,14,9) lo测试集
            y = np.ones(60)  # lo训练集标签
            y_test = np.ones(15)  # lo测试集标签

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))  # (75,14,4)
            a = tmp['eegraw'][:60]  # hi阶段的训练集(60,14,4)
            a1 = tmp['eegraw'][60:]  # hi阶段的测试集 (15,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))  # (75,14,5)
            b = tmp['eegraw'][:60]  # hi阶段的训练集(60,14,5)
            b1 = tmp['eegraw'][60:]  # hi阶段的测试集 (15,14,5)

            x1 = np.concatenate((a, b), axis=2)  # (60,14,9) hi训练集
            x1_test = np.concatenate((a1, b1), axis=2)  # (15,14,9) hi测试集
            y1 = 2 * np.ones(60)  # hi训练集标签
            y1_test = 2 * np.ones(15)  # hi测试集标签

            x = np.concatenate((x, x1), axis=0)  # (120,14,9)  训练集
            x_test = np.concatenate((x_test, x1_test), axis=0)  # (30,14,9)  测试集
            y = np.concatenate((y, y1), axis=0)  # (120,)  训练集标签
            y_test = np.concatenate((y_test, y1_test), axis=0)  # (30,)  测试集标签


            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        return self.data_train, self.label_train, self.data_test, self.label_test



    def get_3Dfeatures_forSTEW_class2_afterICA_Liuyi(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_lo_epoch.mat'.format(sub + 1))  #
            a = tmp['eegraw']  # (75,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']  # (75,14,5)
            x = np.concatenate((a, b), axis=2)  # (75,14,9)
            y = np.ones(75)  # 标签  (75,)

            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw']
            tmp = scipy.io.loadmat('./feature_STEW/ICA_byHand1/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']
            x1 = np.concatenate((a, b), axis=2)  # (75,14,9)
            x = np.concatenate((x, x1), axis=0)  # (150, 14,9)
            c = 2 * np.ones(75)
            y = np.concatenate((y, c), axis=0)  # (150,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(150)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups

    def get_3Dfeatures_forSTEW_class2_no_ICA_Liuyi(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        self.groups = []
        for sub in range(48):
            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/psd/s{}_lo_epoch.mat'.format(sub + 1))  #
            a = tmp['eegraw']  # (75,14,4)
            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']  # (75,14,5)
            x = np.concatenate((a, b), axis=2)  # (75,14,9)
            y = np.ones(75)  # 标签  (75,)

            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw']
            tmp = scipy.io.loadmat('./feature_STEW/no_ICA_byHand/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw']
            x1 = np.concatenate((a, b), axis=2)  # (75,14,9)
            x = np.concatenate((x, x1), axis=0)  # (150, 14,9)
            c = 2 * np.ones(75)
            y = np.concatenate((y, c), axis=0)  # (150,)

            if sub == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
            bbb = np.array(np.ones(150)*(sub+1)).astype(int)
            self.groups.extend(bbb)

        self.data = data  # (2700, 14, 9)
        self.label = label  # (2700,)

        # print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (3456, 14,9) (3456,) (864, 14,9)  (864,)
        return self.data, self.label, self.groups






    # 一个被试
    def get_features3D_forSTEW_eachSub(self, sub):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        tmp = scipy.io.loadmat('./feature_STEW/psd/s{}_lo_epoch.mat'.format(sub))
        a = tmp['eegraw'][:60]  # (72,14,4)
        a1 = tmp['eegraw'][60:]  # 测试集 (18,14,4)
        tmp = scipy.io.loadmat('./feature_STEW/time/s{}_lo_epoch.mat'.format(sub))
        b = tmp['eegraw'][:60]  # (72,14,5)
        b1 = tmp['eegraw'][60:]  # (18,14,5)
        x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
        x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
        y = np.ones(60)  # 训练集标签  (72,)
        y_test = np.ones(15)  # 测试集标签

        tmp = scipy.io.loadmat('./feature_STEW/psd/s{}_hi_epoch.mat'.format(sub))
        a = tmp['eegraw'][:60]
        a1 = tmp['eegraw'][60:]
        tmp = scipy.io.loadmat('./feature_STEW/time/s{}_hi_epoch.mat'.format(sub))
        b = tmp['eegraw'][:60]
        b1 = tmp['eegraw'][60:]
        x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
        x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
        x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
        x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
        c = 2 * np.ones(60)
        y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
        c1 = 2 * np.ones(15)
        y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

        self.data_train = x  # (4320, 126)
        self.label_train = y
        self.data_test= x_test
        self.label_test = y_test  # (4320,)

        print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(self.label_test))
        # (120, 14, 9) (120,) (30, 14, 9) (30,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    # ACAMS1 2分类
    def get_features3D_for_ACAMS1_class2(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(8):
            tmp = scipy.io.loadmat('./feature_ACAMS1/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:720].reshape(720,11,-1)  # (72,14,4)
            a1 = tmp['eegraw'][720:].reshape(180,11,-1)  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_ACAMS1/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:720].reshape(720,11,-1)  # (72,14,5)
            b1 = tmp['eegraw'][720:].reshape(180,11,-1)  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(720)  # 训练集标签  (72,)
            y_test = np.ones(180)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_ACAMS1/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:720].reshape(720,11,-1)
            a1 = tmp['eegraw'][720:].reshape(180,11,-1)
            tmp = scipy.io.loadmat('./feature_ACAMS1/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:720].reshape(720,11,-1)
            b1 = tmp['eegraw'][720:].reshape(180,11,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(720)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(180)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (11520, 11, 9) (11520,) (2880, 11, 9) (2880,)
        return self.data_train, self.label_train, self.data_test, self.label_test

    # ACAMS1 3分类
    def get_features3D_for_ACAMS1_class3(self):
        # 按被试为单位，留出法  特征为三维矩阵，如频域(90,14,4)
        for sub in range(8):
            tmp = scipy.io.loadmat('./feature_ACAMS1/psd/s{}_lo_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:720].reshape(720,11,-1)  # (72,14,4)
            a1 = tmp['eegraw'][720:].reshape(180,11,-1)  # 测试集 (18,14,4)
            tmp = scipy.io.loadmat('./feature_ACAMS1/time/s{}_lo_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:720].reshape(720,11,-1)  # (72,14,5)
            b1 = tmp['eegraw'][720:].reshape(180,11,-1)  # (18,14,5)
            x = np.concatenate((a, b), axis=2)  # (72,14, 9)训练集
            x_test = np.concatenate((a1, b1), axis=2)  # (18,14,9)测试集
            y = np.ones(720)  # 训练集标签  (72,)
            y_test = np.ones(180)  # 测试集标签

            tmp = scipy.io.loadmat('./feature_ACAMS1/psd/s{}_mi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:720].reshape(720, 11, -1)
            a1 = tmp['eegraw'][720:].reshape(180, 11, -1)
            tmp = scipy.io.loadmat('./feature_ACAMS1/time/s{}_mi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:720].reshape(720, 11, -1)
            b1 = tmp['eegraw'][720:].reshape(180, 11, -1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(720)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(180)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            tmp = scipy.io.loadmat('./feature_ACAMS1/psd/s{}_hi_epoch.mat'.format(sub + 1))
            a = tmp['eegraw'][:720].reshape(720,11,-1)
            a1 = tmp['eegraw'][720:].reshape(180,11,-1)
            tmp = scipy.io.loadmat('./feature_ACAMS1/time/s{}_hi_epoch.mat'.format(sub + 1))
            b = tmp['eegraw'][:720].reshape(720,11,-1)
            b1 = tmp['eegraw'][720:].reshape(180,11,-1)
            x1 = np.concatenate((a, b), axis=2)  # (72,14,9)
            x = np.concatenate((x, x1), axis=0)  # (144, 14,9)  训练集
            x1 = np.concatenate((a1, b1), axis=2)  # (18,14,9)
            x_test = np.concatenate((x_test, x1), axis=0)  # (36,14,9)  测试集
            c = 2 * np.ones(720)
            y = np.concatenate((y, c), axis=0)  # (144,)  训练集标签
            c1 = 2 * np.ones(180)
            y_test = np.concatenate((y_test, c1), axis=0)  # (36,)  测试集标签

            if sub == 0:
                data_train = x
                data_test = x_test
                label_train = y
                label_test = y_test
            else:
                data_train = np.concatenate((data_train, x), axis=0)
                data_test = np.concatenate((data_test, x_test), axis=0)
                label_train = np.concatenate((label_train, y), axis=0)
                label_test = np.concatenate((label_test, y_test), axis=0)

        self.data_train = data_train  # (4320, 126)
        self.label_train = label_train
        self.data_test= data_test
        self.label_test = label_test  # (4320,)

        print(np.shape(self.data_train), np.shape(self.label_train), np.shape(self.data_test), np.shape(label_test))
        # (17280, 11, 9) (17280,) (4320, 11, 9) (4320,)
        return self.data_train, self.label_train, self.data_test, self.label_test



    def leaveOneOutGroup(self):
        logo = LeaveOneGroupOut()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        label = np.array(['a', 'c', 'b', 'a', 'c', 'c', 'c', 'b', 'b'])
        groups = [2, 1, 1, 3, 2, 3, 3, 2, 1]
        for train, test in logo.split(data, label, groups=groups):
            print(train, test)
            print("train_data:", data[train], "test_data:", data[test], "train_label:", label[train], "test_label:", label[test])
            # print(train,test)


dataset = Datasets()
# dataset.leaveOneOutGroup()
# dataset.get_3Dfeatures_forNBack_class2_afterICA_improve()
# print(dataset.get_3Dfeatures_forNBack_class3_afterICA_Liuyi())


