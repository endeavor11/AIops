import argparse
import torch

import preprocess_data
from model import model

from pathlib import Path

import numpy as np


from anomalyDetector import anomalyScore_transformer

from utils.eval_methods import calc_point2point
from utils.distri2Th import DistriTailNoise2Th
from utils.eval_methods import adjust_predicts_zsx


# 有save-fig参数

parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
parser.add_argument('--data', type=str, default='ecg',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                    help='filename of the dataset')
parser.add_argument('--save_fig', action='store_true',
                    help='save results as figures')
parser.add_argument('--compensate', action='store_true',
                    help='compensate anomaly score using anomaly score esimation')
parser.add_argument('--beta', type=float, default=1.0,
                    help='beta value for f-beta score')
# 为了DistriTailNoise2Th这个函数的参数，再添加几个参数
# DistriTailNoise2Th(testScore,difTh,avgNum,meanMulCoefficient,binNum,followNum)
parser.add_argument('--difTh',type=float,default=-5)
parser.add_argument('--avgNum',type=int ,default=5)
parser.add_argument('--meanMulCoefficient',type=float,default=100)
parser.add_argument('--binNum',type=int,default=60)
parser.add_argument('--followNum',type=int,default=4)

# 自己的参数
parser.add_argument('--index', type=str)

args_ = parser.parse_args()
print('-' * 89)
print("=> loading checkpoint ")
print("path:", str(Path('save',str(args_.data)+str(args_.index),'checkpoint',args_.filename).with_suffix('.pth')))
checkpoint = torch.load(str(Path('save',str(args_.data)+str(args_.index),'checkpoint',args_.filename).with_suffix('.pth')))
args = checkpoint['args']
args.prediction_window_size= args_.prediction_window_size
args.beta = args_.beta
args.save_fig = args_.save_fig
args.compensate = args_.compensate

print("=> loaded checkpoint")


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data,filename=args.filename, augment_test_data=False)
train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData[:TimeseriesData.length], bsz=1)
test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, bsz=1)

# 注意，这个时候batchsize都是1
# [length,1,feature_dim]

###############################################################################
# Build the model
###############################################################################
nfeatures = TimeseriesData.trainData.size(-1)
model = model.RNNPredictor(rnn_type = args.model,
                           enc_inp_size=nfeatures,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=nfeatures,
                           nlayers = args.nlayers,
                           res_connection=args.res_connection).to(args.device)
model.load_state_dict(checkpoint['state_dict'])  # 载入数据参数！
#del checkpoint


means, covs = checkpoint['means'], checkpoint['covs']
scores_transformer = anomalyScore_transformer(args, model, test_dataset, means, covs) # [allLen - bptt, feature_dim]

supplement = torch.zeros((args.bptt, nfeatures)).cuda()
scores_transformer = torch.cat((supplement, scores_transformer), dim=0)

try:
    # For each channel in the dataset
    
    # 添加自己的话

    # 存储判断的异常
    predList = []
    testScoreList = []
    
    for channel_idx in range(nfeatures):
        ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''


        ''' 2. Train anomaly score predictor using support vector regression (SVR). (Optional) '''
        # An anomaly score predictor is trained
        # given hidden layer output and the corresponding anomaly score on train dataset.
        # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.


        ''' 3. Calculate anomaly scores'''
        # Anomaly scores are calculated on the test dataset
        # given the mean and the covariance calculated on the train dataset
        print('=> calculating anomaly scores')

        # 想使用pot，需要train的分数，所以下面先计算 train_score,再计算test_score
        # 需要注意的是，不能直接输出了结果，需要的是th，还有依据这个东西得出的这个维度的label，需要存储这个label
        # 存到文件里，同时，整个异常检测需要把这些每个维度的结果汇总起来


        #train_score, _, _, hiddens, _ = anomalyScore(args, model, train_dataset, mean, cov, channel_idx=channel_idx)

        score = scores_transformer[:,channel_idx]
                                    # score [length,] 异常分数，越大说明越是异常 torch数组
                                    # sorted_prediction  [time_length,10] 每个时刻的10个预测值
                                    # sorted_error [time_length,10]  每个时刻的预测误差
                                    # prediction_score 默认的话是空数组
        score[score<0.0] = 0.0

        print("score.shape:  ",score.shape)

        ''' 4. Evaluate the result '''
        
        predict = DistriTailNoise2Th(score.cpu().numpy(),args_.difTh,args_.avgNum,args_.meanMulCoefficient,args_.binNum,args_.followNum)
        predList.append(predict)

        testScoreList.append(score.cpu().numpy())
        print("已完成第",channel_idx,"维度的检测")
            
    
    predList = np.array(predList)
    testScoreList = np.array(testScoreList)
    #anomalyResultList.append(anomaly.float())

    #res = get_f1(anomaly, TimeseriesData.testLabel.to(args.device))
    #print(res)

    # 用pickle进行存储

    predListPath = Path("save", args.data, "anomalypred")
    predListPath.mkdir(parents=True, exist_ok=True)
    torch.save(predList, str(predListPath.joinpath(args.filename.split('.')[0]+"_predList").with_suffix(".pth")))

    torch.save(testScoreList, str(predListPath.joinpath(args.filename.split('.')[0]+"_testScoreList").with_suffix(".pth")))
    # 存储完了进行最后的计算，计算准确率
    lastPred = np.sum(predList,axis=0)
    lastPred = lastPred.astype(bool)
    torch.save(lastPred, str(predListPath.joinpath(args.filename.split('.')[0]+"_lastPred").with_suffix(".pth")))
    
    lastPredAdjust = adjust_predicts_zsx(lastPred,TimeseriesData.testLabel.cpu().numpy())
    torch.save(lastPredAdjust, str(predListPath.joinpath(args.filename.split('.')[0]+"_lastPredAdjust").with_suffix(".pth")))
    
    print("lastPredAdjust.shape: ", lastPredAdjust.shape)
    print("TimeseriesData.testLabel.shape:  ", TimeseriesData.testLabel.shape)
    # raise ValueError("stop")

    f1 = calc_point2point(lastPredAdjust,TimeseriesData.testLabel.cpu().numpy())
    ResultPath = Path("save", args.data, "Result")
    ResultPath.mkdir(parents=True, exist_ok=True)
    torch.save(f1, str(ResultPath.joinpath(args.filename.split('.')[0]+"_Result").with_suffix(".pth")))

    print("f1_score=",f1)
    # 淦


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
        
        