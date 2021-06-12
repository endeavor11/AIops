import numpy as np
from scipy.stats import pearsonr

def smooth(a,WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


def getPeakCorre(pearsonData,diffSmoothLength=51,width=10,widthTh=8,iJump=10):
    """
    作用：求出给定的pearson先关系数序列(pearsonData)的周期性怎么样
    1.使用diff差分的方法，求出来高峰点
    2.评估高峰点间的距离是否类似，高峰点的值是否类似(因为周期性数据往往有的pearson比较大，而且因为周期性，所以高峰间距类似)
    3.求若干个高峰之间的相关系数 (pearson的pearson很大，说明传入的pearson很规律，一般而言就是原数据周期性比较好)

    注意：
    1.可以返回周期的，就是底下获得的T
    :param pearsonData: (length,) numpy 数组
    :return:数值越大说明给定的pearsonData的周期性越好
    """
    diff = []
    for i in range(1,pearsonData.shape[0]):
        diff.append(pearsonData[i]-pearsonData[i-1])

    diff = smooth(diff,diffSmoothLength)
    peakList = []

    i = width
    while(i<len(diff)-width):
        pos = 0
        neg = 0
        for j in range(width):
            if(diff[i-j-1]>0):
                pos += 1
            if(diff[i+j]<0):
                neg += 1

        if(pos>=widthTh and neg>=widthTh):
            peakList.append(i)
            i += 10 # 因为不想两个peak离得太近，事实也的确如此 10的话对应interval=10的情况，也就隔离100个时刻间隔
        else:
            i += 1
    peakInterval = []
    for i in range(1,len(peakList)):
        peakInterval.append(peakList[i]-peakList[i-1])


    # 去掉peakinterval中的最大值，最小值，其余的平均，认为是一个周期的长度
    if(len(peakInterval)>=3):
        maxInterval = max(peakInterval)
        minInterval = min(peakInterval)
        T = int((sum(peakInterval) - maxInterval - minInterval)*1.0/(len(peakInterval)-2))
    else:
        T = int(sum(peakInterval)*1.0/(len(peakInterval)))

    # 得出了T！
    # 然后求pearson相关系数序列的pearson相关系数！

    #print(pearsonData.shape)
    #print(peakList)

    pearsonListNew = []
    for i in range(len(peakList)-1):
        for j in range(i+1,len(peakList)-1):
            #print(j)
            if(peakList[j]+T>pearsonData.shape[0]):
                minlen = pearsonData.shape[0]-peakList[j]
                pearsonListNew.append(pearsonr(pearsonData[peakList[i]:peakList[i]+minlen],pearsonData[peakList[j]:peakList[j]+minlen])[0])
            else:
                pearsonListNew.append(pearsonr(pearsonData[peakList[i]:peakList[i]+T],pearsonData[peakList[j]:peakList[j]+T])[0])

    # 计算完毕了相关系数的相关系数了，之后取平均，先输出一下看看
    return sum(pearsonListNew)/(len(pearsonListNew)+1e-6)




def getShift(data1,data2,interval=10,minShift=-1000,maxShift=1000,beginIndex=1000,T=1500):
    """
    对于给定的data1，data2，铆钉data1，然后对data2进行平移，找出附近平移多少时，pearson系数最大
    这个平移值就认为是造成数据没有对齐的那个平移值
    :param data1        : numpy (timestep,)
    :param data2        : numpy (timestep,)
    :param interval     : int such as 10,20
    :param minShift     : int -3000
    :param maxShift     : int 3000
    :param beginIndex   : int 5000
    :param T            : int 1500
    :return             :
    """
    shiftPearsonMap = {} # shift->pearson 虽然暂时用不上，但说不定之后调试的时候会用到
    maxIndex = -1
    maxPear = -1

    for shift in range(minShift,maxShift,interval):
        temp = pearsonr(data1[beginIndex:beginIndex+T],data2[beginIndex+shift:beginIndex+shift+T])[0]
        shiftPearsonMap[shift] = temp
        if(maxPear<temp):
            maxPear = temp
            maxIndex = shift

    return maxIndex


