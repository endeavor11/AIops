import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from getPeakCorre import smooth
from getPeakCorre import getShift
from getPeakCorre import getPeakCorre

def getselfPearson(data,length=3000,allLength=15000,beginIndex=0,interval=10,smoothLength=101):
    """
    计算每个维度数据的pearson系数
    :param data             : numpy array [timestep,dim+1]
    :param length:
    :param allLength:
    :param beginIndex:
    :param interval:
    :param smoothLength:
    :return                 : numpy array [dim,pearson系数的长度]
    """
    pearsonsArrList = []

    for i in range(data.shape[1]-1):
        tempData = data[:,i]
        pearsonsArr = []
        for bias in range(0,allLength,interval):
            temp1 = smooth(tempData[beginIndex:beginIndex+length],smoothLength)
            temp2 = smooth(tempData[beginIndex+bias:beginIndex+length+bias],smoothLength)
            pearsonsArr.append(pearsonr(temp1,temp2)[0])
        pearsonsArrList.append(pearsonsArr)
    pearsonsArrList = np.array(pearsonsArrList)

    return pearsonsArrList




def pickRightKPI(data,pearsonsArrList,dimSeg):
    """
    首先每个机器选择合适的KPI，这个KPI的相关系数的相关系数应该大于0.9(这是之前的经验)
    另外，应该选择方差比较大的KPI，不好的KPI要及时舍弃，因为不好的只会干扰结果
    相关系数大于0.9 然后选一个方差最大的KPI就行

    :param data                 :
    :param pearsonsArrList      :
    :param dimSeg               : [3,3,4]
    :return                     : [0,4,8] 分别是三个机器上，最显著的KPI
    """
    pear2List = []
    for i in range(data.shape[1]-1):
        pear2List.append(getPeakCorre(pearsonsArrList[i]))

    # pear2List = [0.99,  0.97,  0.16,  0.98,  0.39......]

    # 调试:
    # print(pear2List)

    stdList = list(np.std(data,axis=0))
    # stdList = [0.05,0.01,0.12......]


    dimIndexList = []
    currentIndex = 0
    for i in range(len(dimSeg)):
        dimIndexList.append([])
        for j in range(dimSeg[i]):
            dimIndexList[i].append(currentIndex)
            currentIndex += 1

    # 如果是[3,3,4]的情况，dimIndexList = [[0,1,2],[3,4,5],[6,7,8,9]]
    res = []
    for i in range(len(dimIndexList)):
        # 每个机器上的维度，先按照大小排，之后如果>=0.95的大于等于2个，再选择这些大于等于0.95中方差最大的
        candidateDim = []
        pear2Temp = []
        for j in dimIndexList[i]:
            pear2Temp.append(pear2List[j])
        # pear2Temp = [0.99,  0.97,  0.16]

        sortRes = sorted(enumerate(pear2Temp),key=lambda x:-x[1])

        if(sortRes[0][1]<0.95):
            # 说明这个机器pear2最大的KPI的系数也没超过0.95很可怜，只能选择最大的那个了
            res.append(dimIndexList[i][sortRes[0][0]])
        else:
            for j in range(len(dimIndexList[i])):
                if(pear2Temp[j]>=0.95):
                    candidateDim.append(dimIndexList[i][j])

            # candidateDim = [7,8] 类似这样的，然后这些下标找一个std最大的
            maxstd = stdList[candidateDim[0]]
            maxIndex = candidateDim[0]
            for j in range(len(candidateDim)):
                if(stdList[candidateDim[j]]>maxstd):
                    maxstd = stdList[candidateDim[j]]
                    maxIndex = candidateDim[j]
            res.append(maxIndex)


    return res


def data2rightKPI(data,dimSeg,length=3000,allLength=15000,beginIndex=0,interval=10,smoothLength=101):
    pearsonsArrList = getselfPearson(data,length,allLength,beginIndex,interval,smoothLength)
    KPIs = pickRightKPI(data,pearsonsArrList,dimSeg)

    return KPIs


def getShiftResult(data,KPIS,interval=50,minShift=-1000,maxShift=1000,beginIndex=1000,T=1500):
    """

    :param data:
    :param KPIS:
    :param interval:
    :param minShift:
    :param maxShift:
    :param beginIndex:
    :param T:
    :return:
    """
    # 以dim=0的数据为基准进行判断
    shiftResult = []
    for i in range(1,len(KPIS)):
        shiftResult.append(getShift(data[:,KPIS[0]],data[:,KPIS[i]],interval,minShift,maxShift,beginIndex,T))

    return shiftResult

