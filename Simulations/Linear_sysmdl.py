"""# **Class: System Model for Linear Cases**

1 Store system model parameters: 
    state transition matrix F, 
    observation matrix H, 
    process noise covariance matrix Q, 
    observation noise covariance matrix R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test, etc.

2 Generate dataset for linear cases
"""
"""# **Class: System Model for Linear Cases**

1. 存储线性系统的参数：
    - 状态转移矩阵 F
    - 观测矩阵 H
    - 过程噪声协方差矩阵 Q
    - 观测噪声协方差矩阵 R
    - 序列长度 T 和测试序列长度 T_test

2. 用于生成线性系统的数据集
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

"""
        SystemModel 用于表示一个线性动态系统（状态空间模型），支持状态/观测生成、批量数据生成、协方差采样等操作。

        用途：
        - 存储系统模型参数（F、H、Q、R）
        - 生成单条或批量训练/测试数据
        - 提供状态和观测模型的计算（支持带噪声）
        - 可用于卡尔曼滤波器、深度滤波器等任务的模拟数据生成

        Attributes
        ----------
        F : torch.Tensor
            状态转移矩阵（shape: [m, m]）

        Q : torch.Tensor
            过程噪声协方差矩阵（shape: [m, m]）

        H : torch.Tensor
            观测矩阵（shape: [n, m]）

        R : torch.Tensor
            观测噪声协方差矩阵（shape: [n, n]）

        m : int
            状态维度（从 F 中推导）

        n : int
            观测维度（从 H 中推导）

        T : int
            训练序列长度

        T_test : int
            测试序列长度

        prior_Q : torch.Tensor
            Q 的先验协方差，用于估计（可选，默认单位阵）

        prior_Sigma : torch.Tensor
            状态初始协方差的先验（默认 0）

        prior_S : torch.Tensor
            观测协方差的先验（默认单位阵）

        m1x_0 / m2x_0 : torch.Tensor
            初始状态均值和协方差（单序列）

        m1x_0_batch / m2x_0_batch : torch.Tensor
            初始状态均值和协方差（批量）

        Input : torch.Tensor
            输入观测序列（生成后）

        Target : torch.Tensor
            输出状态序列（生成后）

        lengthMask : torch.Tensor
            序列掩码（仅适用于变长序列）

        Methods
        -------
        f(x)
            应用状态转移：x_{t+1} = F x_t

        h(x)
            应用观测模型：y_t = H x_t

        InitSequence(m1x_0, m2x_0)
            初始化单条序列的起始状态和协方差

        Init_batched_sequence(m1x_0_batch, m2x_0_batch)
            初始化批量序列的起始状态和协方差

        UpdateCovariance_Matrix(Q, R)
            更新过程噪声和观测噪声协方差矩阵

        GenerateSequence(Q_gen, R_gen, T)
            生成一条状态/观测序列（支持添加高斯噪声）

        GenerateBatch(args, size, T, randomInit=False)
            生成批量数据（支持随机长度、随机初始状态）

        sampling(q, r, gain)
            对 Q 和 R 进行扰动采样，用于模拟模型不确定性
        """
class SystemModel:
    def __init__(self, F, Q, H, R, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):
        # 初始化系统模型

        ########## 状态转移模型 ##########
        self.F = F  # 状态转移矩阵
        self.m = self.F.size()[0]  # 状态维度
        self.Q = Q  # 过程噪声协方差矩阵

        ########## 观测模型 ##########
        self.H = H  # 观测矩阵
        self.n = self.H.size()[0]  # 观测维度
        self.R = R  # 观测噪声协方差矩阵

        ########## 序列长度 ##########
        self.T = T
        self.T_test = T_test

        ########## 先验协方差（用于估计优化） ##########
        self.prior_Q = prior_Q if prior_Q is not None else torch.eye(self.m)
        self.prior_Sigma = prior_Sigma if prior_Sigma is not None else torch.zeros((self.m, self.m))
        self.prior_S = prior_S if prior_S is not None else torch.eye(self.n)

    # 批量状态转移函数 f(x) = F * x
    def f(self, x):
        batched_F = self.F.to(x.device).view(1, self.m, self.m).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_F, x)

    # 批量观测函数 h(x) = H * x
    def h(self, x):
        batched_H = self.H.to(x.device).view(1, self.n, self.m).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_H, x)

    ########## 初始化（单条序列） ##########
    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0  # 初始均值
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0  # 初始协方差

    ########## 初始化（批量序列） ##########
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    ########## 更新协方差矩阵 ##########
    def UpdateCovariance_Matrix(self, Q, R):
        self.Q = Q
        self.R = R

    ########## 生成一条状态观测序列（用于训练） ##########
    def GenerateSequence(self, Q_gen, R_gen, T):
        self.x = torch.zeros(size=[self.m, T])  # 状态序列
        self.y = torch.zeros(size=[self.n, T])  # 观测序列

        xt = self.x_prev = self.m1x_0

        for t in range(0, T):
            ########## 状态转移 ##########
            xt = self.F.matmul(self.x_prev)

            if not torch.equal(Q_gen, torch.zeros(self.m, self.m)):
                if self.m == 1:
                    eq = torch.normal(mean=0, std=Q_gen)
                else:
                    mean = torch.zeros([self.m])
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                    eq = distrib.rsample().reshape(xt.size())
                xt = torch.add(xt, eq)

            ########## 观测生成 ##########
            yt = self.H.matmul(xt)
            if not torch.equal(R_gen, torch.zeros(self.n, self.n)):
                if self.n == 1:
                    er = torch.normal(mean=0, std=R_gen)
                else:
                    mean = torch.zeros([self.n])
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                    er = distrib.rsample().reshape(yt.size())
                yt = torch.add(yt, er)

            ########## 保存 ##########
            self.x[:, t] = xt.squeeze(1)
            self.y[:, t] = yt.squeeze(1)
            self.x_prev = xt

    ########## 批量生成数据 ##########
    def GenerateBatch(self, args, size, T, randomInit=False):
        if randomInit:
            # 随机初始化起始状态
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            for i in range(size):
                if args.distribution == 'uniform':
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                elif args.distribution == 'normal':
                    distrib = MultivariateNormal(loc=self.m1x_0.squeeze(), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m, 1)
                else:
                    raise ValueError('args.distribution not supported!')
                self.m1x_0_rand[i,:,0:1] = initConditions
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)
        else:
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)
            self.Init_batched_sequence(initConditions, self.m2x_0)

        if args.randomLength:
            # 变长序列：初始化数据和掩码
            self.Input = torch.zeros(size, self.n, args.T_max)
            self.Target = torch.zeros(size, self.m, args.T_max)
            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)
            T_tensor = torch.round((args.T_max - args.T_min) * torch.rand(size)).int() + args.T_min
            for i in range(size):
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                self.Input[i,:,0:T_tensor[i]] = self.y
                self.Target[i,:,0:T_tensor[i]] = self.x
                self.lengthMask[i, 0:T_tensor[i]] = True
        else:
            # 固定长度序列：批量生成
            self.Input = torch.empty(size, self.n, T)
            self.Target = torch.empty(size, self.m, T)
            xt = self.x_prev = self.m1x_0_batch
            for t in range(T):
                xt = self.f(self.x_prev)
                if not torch.equal(self.Q, torch.zeros(self.m, self.m)):
                    if self.m == 1:
                        eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
                    else:
                        distrib = MultivariateNormal(loc=torch.zeros([size,self.m]), covariance_matrix=self.Q)
                        eq = distrib.rsample().view(size,self.m,1)
                    xt = torch.add(xt, eq)

                yt = self.h(xt)
                if not torch.equal(self.R, torch.zeros(self.n, self.n)):
                    if self.n == 1:
                        er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
                    else:
                        distrib = MultivariateNormal(loc=torch.zeros([size,self.n]), covariance_matrix=self.R)
                        er = distrib.rsample().view(size,self.n,1)
                    yt = torch.add(yt, er)

                self.Target[:, :, t] = xt.squeeze(2)
                self.Input[:, :, t] = yt.squeeze(2)
                self.x_prev = xt

    ########## 采样函数（用于生成随机噪声协方差） ##########
    def sampling(self, q, r, gain):
        # 控制是否增加随机扰动（用于对 Q 和 R 加噪）
        if gain != 0:
            gain_q = 0.1
            aq = gain_q * q * torch.eye(self.m)
        else:
            aq = 0
        Aq = q * torch.eye(self.m) + aq
        Q_gen = torch.transpose(Aq, 0, 1) @ Aq

        if gain != 0:
            gain_r = 0.5
            ar = gain_r * r * torch.eye(self.n)
        else:
            ar = 0
        Ar = r * torch.eye(self.n) + ar
        R_gen = torch.transpose(Ar, 0, 1) @ Ar

        return [Q_gen, R_gen]
