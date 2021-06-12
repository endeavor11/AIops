import argparse
import numpy
import pickle
from matplotlib import pyplot as plt

"""
1.文件作用：
2.使用方法：
python drawTimeSeries.py --dataname machine-1-1.pkl  [其余选项]
注意，某些选项如果应该填False 那么就什么都不要写！
如果是True，那么就写上True

3.注意
    (1)bool的那几个选项，dataWithLabel，如果不是，别写False，直接不写
       这个参数就好了
"""

parser = argparse.ArgumentParser()

parser.add_argument("--dataname",type=str)
parser.add_argument("--labelname",type=str)      # 不输入，输出结果是None，类型NoneType
                                                 # 只写--dataname 报错
parser.add_argument("--haveLabel",type=bool)
parser.add_argument("--dataWithLabel",type=bool) # 输入True，那么类型就是bool，输入False ，结果也是True
                                                 # 如果随便输入，也是True，
                                                 # 如果不输入 直接输出就是None，类型是NoneType 
                                                 # 如果只写--dataWithlabel 那么会报错
parser.add_argument("--begin_index",type=int,default=0)
parser.add_argument("--end_index",type=int,default=-1)


args = parser.parse_args()


if(args.dataWithLabel==True): # 该分支没有问题
    print("有data有label，label在data中")
    data = pickle.load(open(args.dataname,'rb'))
    label = data[:,-1]
    data = data[:,0:-1]
    #label = pickle.load(open(args.labelname,'rb'))
    dim = data.shape[1]
    length = data.shape[0]
    begin_index = 0
    end_index = 0

    if(args.end_index!=-1):
        begin_index = args.begin_index
        end_index = args.end_index
    else:
        begin_index = 0
        end_index = length

    fig_size_x = (end_index-begin_index)/250
    fig_size_y = 3*dim

    fig = plt.figure(figsize=(fig_size_x,fig_size_y))

    for i in range(dim):
        ax = fig.add_subplot(dim,1,i+1)
        ax.plot(range(begin_index,end_index),
                data[begin_index:end_index,i])
        ax.plot(range(begin_index,end_index),
                label[begin_index:end_index])

    plt.savefig(args.dataname+"fig.png")

elif(args.haveLabel==True): # data 和 label 分离
                            # 该分支没有问题
    print("有data有label，label与data分离")
    data = pickle.load(open(args.dataname,'rb'))
    label = pickle.load(open(args.labelname,'rb'))

    # 下面这段是从上面复制过来的
    dim = data.shape[1]
    length = data.shape[0]
    begin_index = 0
    end_index = 0

    if(args.end_index!=-1):
        begin_index = args.begin_index
        end_index = args.end_index
    else:
        begin_index = 0
        end_index = length

    fig_size_x = (end_index-begin_index)/250
    fig_size_y = 3*dim

    fig = plt.figure(figsize=(fig_size_x,fig_size_y))
    #print("到达")

    for i in range(dim):
        ax = fig.add_subplot(dim,1,i+1)
        ax.plot(range(begin_index,end_index),
                data[begin_index:end_index,i])
        ax.plot(range(begin_index,end_index),
                label[begin_index:end_index])

    plt.savefig(args.dataname+"_fig.png")

else:
    # 该分支没有问题
    print("有data无label")
    data = pickle.load(open(args.dataname,'rb'))
    dim = data.shape[1]
    length = data.shape[0]
    begin_index = 0
    end_index = 0

    if(args.end_index!=-1):
        begin_index = args.begin_index
        end_index = args.end_index
    else:
        begin_index = 0
        end_index = length

    fig_size_x = (end_index-begin_index)/250
    fig_size_y = 3*dim

    fig = plt.figure(figsize=(fig_size_x,fig_size_y))

    for i in range(dim):
        ax = fig.add_subplot(dim,1,i+1)
        ax.plot(range(begin_index,end_index),
                data[begin_index:end_index,i])

    plt.savefig(args.dataname+"fig.png")
