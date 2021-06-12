import numpy as np

"""
1.创建数据集的时候：
    传入数据集，然后进行一定程度的位移
"""

def createShiftdata(rawData,beginList,length,dimSeg):
    """

    :param rawData:         numpy 数组 [timestep,dim+1] 最后一个维度是label    [20000,11]
    :param beginList:       [int,int,int] list类型，是几个维度的起始 time index [50,60,70]
    :param length:          最后输出的数据的timestep长度                        18000
    :param dimSeg:          [3,3,4] 也可以是其他的                             [3,3,4]

    beginList 和 dimSeg 必须有相同数量的元素

    :return: [length,dim+1]
    """
    segData = np.zeros((length,rawData.shape[1]))

    for i in range(len(beginList)):
        assert length+beginList[i]-1<rawData.shape[0]

    dimIndexList = []
    currentIndex = 0
    for i in range(len(dimSeg)):
        dimIndexList.append([])
        for j in range(dimSeg[i]):
            dimIndexList[i].append(currentIndex)
            currentIndex += 1

    # 如果是[3,3,4]的情况，dimIndexList = [[0,1,2],[3,4,5],[6,7,8,9]]

    for i in range(len(dimSeg)):
        segData[:,dimIndexList[i]] = rawData[beginList[i]:beginList[i]+length,dimIndexList[i]]

    segData[:,-1] = rawData[beginList[0]:beginList[0]+length,-1]

    return segData


