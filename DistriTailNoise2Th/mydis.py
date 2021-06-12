import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def DistriTailNoise2Th(testScore,difTh,avgNum,meanMulCoefficient,binNum,followNum):
    """
    方法：异常分数的分布是遵循某个分布的，大致形状类似正态分布，分数最低数量很少，某个分数数量很多，分数很大的点也很少。在有异常分数的情况下，
    如果分数越高越可能是异常，那么在原先平滑的分布上，应该会有一个小波峰，代表一些异常点的分数，该方法试图找到这个波峰开始的地方，然后分数大于
    波峰开始的分数的分数对应的异常点就认为是异常

    想说明的一点是，训练分数对判断测试集异常分数的threshold不知道该怎么用，因为一些异常点，整体的分数都有了很大的提升，通过衡量训练集的分数
    然后用什么3sigma原则，或者先对所有分数取对数，让分数更平滑，然后根据这个判断异常也是很困难的，需要对不同的数据集做出很详细地调整

    :param testScore: numpy array [timestep,] [6,7,8,9,100,200,4,2,1]the score of the testdata,higher score mean anomaly
    :param difTh: float 50 use to determine whether to set current score as a score threshold
    :param avgNum: int 5 smooth dif
    :param meanMulCoefficient: float 100.0 决定大于平均数的几倍进行截断
    :param binNum: int 60 划分score区间的数量
    :param followNum: int 4 在初步决定了一个th后，后面连续几个的dif值不能太小，如果还是很小，说明之前决定的th只是一个小小的噪声而已
    :return: the predict result [0,0,0,0,0,1,1,1,0,0,0,0] 1代表异常
    """

    # 为了使分数的分布更加平滑一些，这里做一些分数截断，同时也让异常分数更加集中一点，因为时序序列那里，有的很异常的点，分数比正常点大很多很多
    timestep = testScore.shape[0]


    meanMul = testScore.mean()*meanMulCoefficient
    for i in range(timestep):
        if(testScore[i]>meanMul):
            testScore[i] = meanMul

    x = plt.hist(testScore,bins=binNum)
    # x[0]: 每个区间里面的离散点的数量
    # x[1]: 每个区间的边界

    score_temp = x[0][1:] # 因为第一个区间数目非常多，先不用考虑了
    differential = [] # 求每个区间点个数的差分
    for i in range(1,score_temp.shape[0]):
        differential.append(score_temp[i] - score_temp[i-1])

    # 这个时候differential 应该基本都是负值 因为分数越大，点数越少

    difAvg = [] # 对differential进行平滑
    for i in range(avgNum-1,len(differential)):
        tempSum = 0
        for j in range(avgNum):
            tempSum = tempSum + differential[i-j]
        difAvg.append(tempSum/avgNum)

    # difAvg已经处理完毕了，这个时候就需要去寻找th了
    index = -1 # 满足要求的dif的下标

    for i in range(len(difAvg)):
        if(difAvg[i]>difTh):
            flag = True

            for temp in range(followNum):
                if(i+temp<len(difAvg) and difAvg[i+temp]<difTh):
                    flag = False

            if(flag == True):
                index = i
                break
    if(index==-1):
        raise ValueError("the score can't judge a anomaly score threshold")

    th = x[1][1:][index]

    predict = (testScore>th).astype(float) # 分数越小越是异常

    return predict

