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
from anomalyDetector import fit_norm_distribution_param, anomalyScore_transformer
from anomalyDetector import anomalyScore

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

test_dataset = TimeseriesData.batchify(args, TimeseriesData.testData, bsz=1)

# 注意，这个时候batchsize都是1
# [length,1,feature_dim]

# feature_dim = 8+8+8
feature_dim = 4+3+3


###############################################################################
# Build the model
###############################################################################
#nfeatures = TimeseriesData.trainData.size(-1) # 10
feature_data = TimeseriesData.trainData.size(-1) # 10


encoder_client_0 = model_FL.encoder_client(enc_client_dims=[[4,4]],layer_num=1).to(args.device)
encoder_client_1 = model_FL.encoder_client(enc_client_dims=[[3,3]],layer_num=1).to(args.device)
encoder_client_2 = model_FL.encoder_client(enc_client_dims=[[3,3]],layer_num=1).to(args.device)

decoder_client_0 = model_FL.encoder_client(enc_client_dims=[[4,4]],layer_num=1).to(args.device)
decoder_client_1 = model_FL.encoder_client(enc_client_dims=[[3,3]],layer_num=1).to(args.device)
decoder_client_2 = model_FL.encoder_client(enc_client_dims=[[3,3]],layer_num=1).to(args.device)

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

recover_dimensions_client_0 = [0,1,2,3]
recover_dimensions_client_1 = [4,5,6]
recover_dimensions_client_2 = [7,8,9]

config={}
config['dimensions_client_0'] = dimensions_client_0
config['dimensions_client_1'] = dimensions_client_1
config['dimensions_client_2'] = dimensions_client_2

config['recover_dimensions_client_0'] = recover_dimensions_client_0
config['recover_dimensions_client_1'] = recover_dimensions_client_1
config['recover_dimensions_client_2'] = recover_dimensions_client_2

# del checkpoint

means, covs = checkpoint['means'], checkpoint['covs']

scores_transformer = anomalyScore_transformer(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                      decoder_client_0,decoder_client_1,decoder_client_2,
                                      test_dataset, means, covs, config) # [allLen - bptt, feature_dim]

supplement = torch.zeros((args.bptt, feature_dim)).cuda()
scores_transformer = torch.cat((supplement, scores_transformer), dim=0)

try:
    # For each channel in the dataset

    predList = []
    testScoreList = []
    
    for channel_idx in range(feature_data):

        score = scores_transformer[:, channel_idx]

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