"""
Class: KalmanFilter
说明：理论线性卡尔曼滤波器的批处理版本（batched Kalman Filter）
用于并行地对多条观测序列进行状态估计
"""

import torch

'''
    KalmanFilter 类：批处理版本的线性卡尔曼滤波器（Theoretical Linear Kalman Filter）。

    用于处理多个观测序列的状态估计，支持 GPU 计算，并实现了经典的卡尔曼滤波四步流程：
    - 状态预测
    - 观测预测
    - 卡尔曼增益计算
    - 状态修正

    适用于状态空间模型是线性、高斯假设成立的系统。可用于轨迹预测、定位、跟踪等场景。

    Attributes:
        F (torch.Tensor): 状态转移矩阵，形状为 [m, m]
        Q (torch.Tensor): 状态过程噪声协方差矩阵，形状为 [m, m]
        H (torch.Tensor): 观测矩阵，形状为 [n, m]
        R (torch.Tensor): 观测噪声协方差矩阵，形状为 [n, n]
        T (int): 序列长度（训练阶段用）
        T_test (int): 测试序列长度
        m (int): 状态变量维度
        n (int): 观测变量维度
        device (torch.device): 使用的设备（CPU/GPU）

    Methods:
        Init_batched_sequence(m1x_0_batch, m2x_0_batch):
            批量初始化所有序列的初始状态均值和协方差。

        GenerateBatch(y):
            对观测批次执行完整卡尔曼滤波，返回状态估计。

        Predict():
            预测当前状态和观测的均值与协方差。

        KGain():
            计算卡尔曼增益，用于修正当前预测。

        Innovation(y):
            计算观测与预测观测的差值（创新）。

        Correct():
            根据观测更新状态估计。

        Update(y):
            执行一次预测-更新循环，返回后验状态估计。
'''
class KalmanFilter:

    def __init__(self, SystemModel, args):
        """
        初始化卡尔曼滤波器参数

    初始化 KalmanFilter 对象

    参数:
        SystemModel (object): 包含系统模型矩阵 F、H、Q、R 等的对象
        args (Namespace): 含配置选项的参数集，需包含 use_cuda 等参数
        参数名            |       含义               | 例子
        use_cuda        | 是否使用 CUDA / GPU       | True / False
        N_T            | 测试序列的数量（batch 数）  | 100
        T_test        | 每条测试序列的长度          | 50
        randomLength | 是否每条序列长度不同         | True / False

    初始化内容：
        - 模型矩阵（F, Q, H, R）加载到设备
        - 设置维度参数 m, n
        - 设置时间序列长度 T
        """
        # 设置计算设备（GPU 或 CPU）
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # 状态转移矩阵 F（预测用）
        self.F = SystemModel.F
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)  # 状态噪声协方差

        # 观测矩阵 H（从状态映射到观测）
        self.H = SystemModel.H
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)  # 观测噪声协方差

        self.T = SystemModel.T          # 序列长度（训练用）
        self.T_test = SystemModel.T_test  # 测试序列长度

    # ------------- 卡尔曼滤波主步骤 -------------

    def Predict(self):
        """
        预测步骤：从上一个状态预测下一个状态
        """
        # 预测状态均值（first moment of x）
        self.m1x_prior = torch.bmm(self.batched_F, self.m1x_posterior).to(self.device)

        # 预测状态协方差（second moment of x）
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # 预测观测均值（first moment of y）
        self.m1y = torch.bmm(self.batched_H, self.m1x_prior)

        # 预测观测协方差（second moment of y）
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    def KGain(self):
        """
        计算卡尔曼增益（Kalman Gain）
        """
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))  # KG = PHT * (HPH^T + R)^-1

    def Innovation(self, y):
        """
        计算创新项（观测误差）
        dy = y - 预测观测
        """
        self.dy = y - self.m1y

    def Correct(self):
        """
        更新步骤：根据观测值修正状态估计
        """
        # 更新状态均值
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

        # 更新状态协方差
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        """
        执行一次完整的预测 + 更新过程
        返回：后验状态均值、协方差
        """
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        """
        批量初始化所有序列的初始状态（用于多序列处理）
        m1x_0_batch: 初始状态均值 [batch_size, m, 1]
        m2x_0_batch: 初始状态协方差 [batch_size, m, m]
        """
        self.m1x_0_batch = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    ############################
    ### 批处理观测序列入口 ###
    ############################

    def GenerateBatch(self, y):
        """
        主函数：对一批观测序列执行卡尔曼滤波

        参数：
        y: [batch_size, n, T]  多条观测序列
        """
        y = y.to(self.device)
        self.batch_size = y.shape[0]  # 序列条数
        T = y.shape[2]                # 每条序列的长度（时间步）

        # 批量展开状态转移矩阵 F（重复成 batch_size 份）
        self.batched_F = self.F.view(1, self.m, self.m).expand(self.batch_size, -1, -1).to(self.device)
        self.batched_F_T = torch.transpose(self.batched_F, 1, 2).to(self.device)

        # 批量展开观测矩阵 H
        self.batched_H = self.H.view(1, self.n, self.m).expand(self.batch_size, -1, -1).to(self.device)
        self.batched_H_T = torch.transpose(self.batched_H, 1, 2).to(self.device)

        # 分配空间用于存储每一步的估计值和协方差
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)            # 状态估计值
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device) # 协方差估计值

        # 初始化 t=0 的估计（来自 Init_batched_sequence）
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)

        # 逐时间步执行滤波
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, :, t], 2)  # 当前时间步的观测值，形状 [batch_size, n, 1]
            xt, sigmat = self.Update(yt)         # 执行一次卡尔曼更新
            self.x[:, :, t] = torch.squeeze(xt, 2)       # 存储状态估计
            self.sigma[:, :, :, t] = sigmat              # 存储协方差估计
