import argparse
import torch
import pickle
import preprocess_data
from model import model_FL
from torch import optim
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from anomalyDetector import fit_norm_distribution_param
from anomalyDetector import anomalyScore
from anomalyDetector import get_precision_recall
from anomalyDetector import get_precision_recall_zsx_2
from anomalyDetector import get_f1
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

args_ = parser.parse_args()
print('-' * 89)
print("=> loading checkpoint ")
checkpoint = torch.load(str(Path('save', args_.data, 'checkpoint', args_.filename).with_suffix('.pth')))
args = checkpoint['args']
args.prediction_window_size = args_.prediction_window_size
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
TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename, augment_test_data=False)
train_dataset = TimeseriesData.batchify(args, TimeseriesData.trainData[:TimeseriesData.length], bsz=1)
test_dataset = TimeseriesData.batchify(args, TimeseriesData.testData, bsz=1)

# 注意，这个时候batchsize都是1
# [length,1,feature_dim]

feature_dim = 8+8+8


###############################################################################
# Build the model
###############################################################################
#nfeatures = TimeseriesData.trainData.size(-1) # 10
feature_data = TimeseriesData.trainData.size(-1) # 10


encoder_client_0 = model_FL.encoder_client(enc_client_dims=[[4,8],[8,8]],layer_num=2).to(args.device)
encoder_client_1 = model_FL.encoder_client(enc_client_dims=[[3,8],[8,8]],layer_num=2).to(args.device)
encoder_client_2 = model_FL.encoder_client(enc_client_dims=[[3,8],[8,8]],layer_num=2).to(args.device)

decoder_client_0 = model_FL.encoder_client(enc_client_dims=[[8,8],[8,4]],layer_num=2).to(args.device)
decoder_client_1 = model_FL.encoder_client(enc_client_dims=[[8,8],[8,3]],layer_num=2).to(args.device)
decoder_client_2 = model_FL.encoder_client(enc_client_dims=[[8,8],[8,3]],layer_num=2).to(args.device)

model_server_0 = model_FL.model_server(rnn_type = args.model,
                                       enc_inp_size=feature_dim,
                                       rnn_inp_size=args.emsize,
                                       rnn_hid_size=args.nhid,
                                       dec_out_size=feature_dim,
                                       nlayers=args.nlayers,
                                       res_connection=args.res_connection).to(args.device)



model_server_0.load_state_dict(checkpoint['state_dict']['model_server_0'])  # 载入数据参数！

encoder_client_0.load_state_dict(checkpoint['state_dict']['encoder_client_0'])
encoder_client_1.load_state_dict(checkpoint['state_dict']['encoder_client_1'])
encoder_client_2.load_state_dict(checkpoint['state_dict']['encoder_client_2'])
decoder_client_0.load_state_dict(checkpoint['state_dict']['decoder_client_0'])
decoder_client_1.load_state_dict(checkpoint['state_dict']['decoder_client_1'])
decoder_client_2.load_state_dict(checkpoint['state_dict']['decoder_client_2'])

dimensions_client_0 = [0,1,2,3]
dimensions_client_1 = [4,5,6]
dimensions_client_2 = [7,8,9]

recover_dimensions_client_0 = [0,1,2,3,4,5,6,7]
recover_dimensions_client_1 = [8,9,10,11,12,13,14,15]
recover_dimensions_client_2 = [16,17,18,19,20,21,22,23]

config={}
config['dimensions_client_0'] = dimensions_client_0
config['dimensions_client_1'] = dimensions_client_1
config['dimensions_client_2'] = dimensions_client_2

config['recover_dimensions_client_0'] = recover_dimensions_client_0
config['recover_dimensions_client_1'] = recover_dimensions_client_1
config['recover_dimensions_client_2'] = recover_dimensions_client_2

# del checkpoint

scores, predicted_scores, precisions, recalls, f_betas = list(), list(), list(), list(), list()
targets, mean_predictions, oneStep_predictions, Nstep_predictions = list(), list(), list(), list()
try:
    # For each channel in the dataset

    predList = []
    testScoreList = []
    
    for channel_idx in range(feature_data):
        ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''
        # Mean and covariance are calculated on train dataset.
        if 'means' in checkpoint.keys() and 'covs' in checkpoint.keys():
            print('=> loading pre-calculated mean and covariance')
            mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
        else:
            print('=> calculating mean and covariance')
            mean, cov = fit_norm_distribution_param(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                                     decoder_client_0,decoder_client_1,decoder_client_2,
                                                     train_dataset, channel_idx=0,config=config)

        ''' 2. Train anomaly score predictor using support vector regression (SVR). (Optional) '''
        # An anomaly score predictor is trained
        # given hidden layer output and the corresponding anomaly score on train dataset.
        # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.
        if args.compensate:  # 默认False
            # compensate anomaly score using anomaly score esimation
            print('=> training an SVR as anomaly score predictor')
            train_score, _, _, hiddens, _ = anomalyScore(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                                                              decoder_client_0,decoder_client_1,decoder_client_2,
                                                                              train_dataset, mean, cov, channel_idx=0, config=config)
            score_predictor = GridSearchCV(SVR(), cv=5,
                                           param_grid={"C": [1e0, 1e1, 1e2], "gamma": np.logspace(-1, 1, 3)})
            score_predictor.fit(torch.cat(hiddens, dim=0).numpy(), train_score.cpu().numpy())
        else:
            score_predictor = None

        ''' 3. Calculate anomaly scores'''
        # Anomaly scores are calculated on the test dataset
        # given the mean and the covariance calculated on the train dataset
        print('=> calculating anomaly scores')
        
        #train_score, _, _, hiddens, _ = anomalyScore(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
        #                                                                      decoder_client_0,decoder_client_1,decoder_client_2,
        #                                                                      train_dataset, mean, cov, channel_idx=0, config=config)
        
        score, sorted_prediction, sorted_error, _, predicted_score = anomalyScore(args, model_server_0,
                                                                                  encoder_client_0,encoder_client_1,encoder_client_2,
                                                                                  decoder_client_0,decoder_client_1,decoder_client_2,
                                                                                  test_dataset, mean, cov,
                                                                                  score_predictor=score_predictor,
                                                                                  channel_idx=channel_idx,
                                                                                  config=config)
        # score [length,] 异常分数，越大说明越是异常
        # sorted_prediction  [time_length,10] 每个时刻的10个预测值
        # sorted_error [time_length,10]  每个时刻的预测误差
        # prediction_score 默认的话是空数组

        ''' 4. Evaluate the result '''
        # The obtained anomaly scores are evaluated by measuring precision, recall, and f_beta scores
        # The precision, recall, f_beta scores are are calculated repeatedly,
        # sampling the threshold from 1 to the maximum anomaly score value, either equidistantly or logarithmically.
        print('=> calculating precision, recall, and f_beta')
        # precision, recall, f_beta = get_precision_recall(args, score, num_samples=1000, beta=args.beta,
        #                                                 label=TimeseriesData.testLabel.to(args.device))
        #anomaly_temp = get_precision_recall_zsx_2(args, score, num_samples=1000, beta=args.beta,
        #                                          label=TimeseriesData.testLabel.to(args.device))
        
        score[score<0.0] = 0.0

        predict = DistriTailNoise2Th(score.cpu().numpy(),args_.difTh,args_.avgNum,args_.meanMulCoefficient,args_.binNum,args_.followNum)
        predList.append(predict)

        testScoreList.append(score.cpu().numpy())

        print("已完成第",channel_idx,"维度的检测")



    predList = np.array(predList)
    testScoreList = np.array(testScoreList)
    
    predListPath = Path("save", args.data, "anomalypred")
    predListPath.mkdir(parents=True, exist_ok=True)
    torch.save(predList, str(predListPath.joinpath(args.filename.split('.')[0]+"_predList").with_suffix(".pth")))
    
    torch.save(testScoreList, str(predListPath.joinpath(args.filename.split('.')[0]+"_testScoreList").with_suffix(".pth")))

    lastPred = np.sum(predList,axis=0)
    lastPred = lastPred.astype(bool)
    torch.save(lastPred, str(predListPath.joinpath(args.filename.split('.')[0]+"_lastPred").with_suffix(".pth")))

    lastPredAdjust = adjust_predicts_zsx(lastPred,TimeseriesData.testLabel.cpu().numpy())
    torch.save(lastPredAdjust, str(predListPath.joinpath(args.filename.split('.')[0]+"_lastPredAdjust").with_suffix(".pth")))

    f1 = calc_point2point(lastPredAdjust,TimeseriesData.testLabel.cpu().numpy())
    ResultPath = Path("save", args.data, "Result")
    ResultPath.mkdir(parents=True, exist_ok=True)
    torch.save(f1, str(ResultPath.joinpath(args.filename.split('.')[0]+"_Result").with_suffix(".pth")))
    print("f1_score=",f1)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')