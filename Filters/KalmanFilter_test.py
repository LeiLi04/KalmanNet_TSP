import torch
import torch.nn as nn
import time
from Filters.Linear_KF import KalmanFilter
"""
    使用卡尔曼滤波器（Kalman Filter）对测试数据进行推理，并计算MSE误差。

    参数:
    - args: 配置参数对象（包含 T_test、N_T 等）
    - SysModel: 系统模型对象，包含状态空间维度及初始状态分布
    - test_input: 测试输入序列（观测值）
    - test_target: 测试目标序列（真实状态值）
    - allStates: 是否评估所有状态（默认是 True），否则只评估位置
    - randomInit: 是否使用随机初始化状态（默认 False）
    - test_init: 可选的测试序列初始化值（用于 randomInit）
    - test_lengthMask: 可选的序列长度掩码（用于变长序列处理）

    返回:
    - [每个序列的MSE, 平均MSE, 平均MSE(dB), KF输出状态序列]
"""
def KFTest(args, SysModel, test_input, test_target, allStates=True,\
     randomInit = False, test_init=None, test_lengthMask=None):

    # 使用均方误差（MSE）作为损失函数
    loss_fn = nn.MSELoss(reduction='mean')

    # 初始化 MSE 数组（每个序列一个MSE值）
    MSE_KF_linear_arr = torch.zeros(args.N_T)
    # 分配内存：存储每个序列的 KF 输出|allocate memory for KF output
    KF_out = torch.zeros(args.N_T, SysModel.m, args.T_test)
    # 如果只评估部分状态（如位置），设置对应的掩码
    if not allStates:
        #(position, speed, acceleration)
        loc = torch.tensor([True,False,False]) # for position only| # 针对 m=3 的情况
        if SysModel.m == 2:
            #(position, speed)
            loc = torch.tensor([True,False]) # for position only| # 针对 m=2 的情况
    # 开始计时
    start = time.time()

    # 创建卡尔曼滤波器对象
    KF = KalmanFilter(SysModel, args)
    # Init and Forward Computation
    # 初始化卡尔曼滤波器（根据是否随机初始化选择不同的初始化方式）
    if(randomInit):
        KF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))     #view() ~ reshape()
    else:
        KF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(args.N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))           
    # 使用测试输入数据执行前向推理（滤波过程）
    KF.GenerateBatch(test_input)
    # 结束计时
    end = time.time()
    t = end - start # 总推理时间
    # 获取滤波器输出的状态估计
    KF_out = KF.x
    # MSE loss
    # 遍历每个序列，计算其 MSE
    for j in range(args.N_T):# cannot use batch due to different length and std computation
        # 若考虑所有状态
        if(allStates):
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,:,:], test_target[j,:,:]).item()
        else: # mask on state
            # 若只考虑部分状态（如位置）
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,loc,:], test_target[j,loc,:]).item()
    # 计算平均MSE（线性）
    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    # 将平均MSE转换为分贝表示（dB）
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    # 计算MSE的标准差
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    # 计算dB单位下的误差置信区间宽度（上下浮动范围）
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    # 打印评估结果
    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out]



