from torch.autograd import Variable
import torch
import numpy as np
import pickle

"""
seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
# random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
"""

def fit_norm_distribution_param(args, model, train_dataset, channel_idx=0): # 具体意义见OneNote
    # train_dataset batchsize 为1
    # dataset 尺寸为：[length,1,feature_dim]
    predictions = []
    organized = []
    errors = []
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        #pasthidden = model.init_hidden(1)
        for t in range(len(train_dataset)):
            out, hidden = model.forward(train_dataset[t].unsqueeze(0), 0)
            
            # train_dataset[t].unsqueeze(0) size [1,1,feature_dim]
            # out:[1,1,feature_dim] 因为seq和batch都是 1 
            
            predictions.append([])
            organized.append([])
            errors.append([])
            predictions[t].append(out.data.cpu()[0][0][channel_idx])  # append 了一个标量
            #pasthidden = model.repackage_hidden(hidden)

                                                        # 值得注意的是这个pasthidden，不能变成下面
                                                        # 经过一系列迭代预测之后的hidden！可以理解吧！
                                                        # 因为下一个for，是从当前for的真实数据的下一个时刻
                                                        # 的真实数据开始的
                                        
            for prediction_step in range(1,args.prediction_window_size):  # 默认10
                out, hidden = model.forward(out, hidden)
                predictions[t].append(out.data.cpu()[0][0][channel_idx])
                
                # 这个for 说明，喂一个真实值，预测10个值
                # 然后prediction里面每个元素是一个列表，每个元素代表模型在该时刻预测的后面时刻的所有值
                # 一般来说，prediction每个元素又含有10个元素
                

            if t >= args.prediction_window_size:
                for step in range(args.prediction_window_size):  # 从0到9
                    organized[t].append(predictions[step+t-args.prediction_window_size][args.prediction_window_size-1-step])
                    # organized 取出所有的 在时刻 t的 预测值 
                    # 第一个下标写成 t-args.prediction_window_size+step 比较合适
                    # 第二个下标写成 args.prediction_window_size+step -1 -step 这个这样理解，step=0的时候，该时刻的预测
                    # 值在最后一个值，所以就这样就ok了
                organized[t]= torch.FloatTensor(organized[t]).to(args.device)
                errors[t] = organized[t] - train_dataset[t][0][channel_idx]  # 留下了差值 [10,]
                errors[t] = errors[t].unsqueeze(0)  # [1,10]

    errors_tensor = torch.cat(errors[args.prediction_window_size:],dim=0)  # 因为前几个error都是[] 空的！
                                                            # size [length,10]
    mean = errors_tensor.mean(dim=0) # [10,]
    cov = errors_tensor.t().mm(errors_tensor)/errors_tensor.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(0))
    # cov: positive-semidefinite and symmetric.
    # 待考察这个conv
    
    # mean [10,]
    # cov [10,10]

    return mean, cov


# 针对transformer的均值方差计算函数
def fit_norm_distribution_param_transformer(args, model, train_dataset):
    """
    :param args:
    :param model:
    :param train_dataset: [newtime,batchsize,feature_dim]
    :return: mean,cov

    长度需要和训练时保持一致，args.bptt
    """
    diffAbs = None

    print("train_dataset.shape: ", train_dataset.shape)

    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()

        for i in range(0,train_dataset.shape[0] - args.bptt):
            # i作为一段序列的开始下标
            # 最后一个下标是 len - bptt - 1，则该段序列的最后一个下标是 len - bptt - 1 + bptt - 1 = len - 2
            # 刚好把数据用完, 因为target还要往后面取一个时刻的数据点的
            seq = train_dataset[i: i+args.bptt] # [bptt, batch, feature_dim]
            target_seq = train_dataset[i+1: i+1+args.bptt]
            output, _ = model(seq, hidden=0, return_hiddens=False) # [bptt, batch, feature_dim]

            # print("seq.shape: ", seq.shape)
            # print("target_seq.shape: ", target_seq.shape)
            # print("output.shape: ", output.shape)
            #
            # raise ValueError("stop")

            # 两种方式，不是最后一个时刻的数据也当做预测，或者只有最后一个时刻的预测当预测
            if(diffAbs is None):
                diffAbs = output[-1] - target_seq[-1] # [batch, feature_dim]
            else:
                diffAbs = torch.cat((diffAbs, output[-1] - target_seq[-1]), dim=0)
            # 拼到最后 [batch*n, feature_dim]

    mean = torch.mean(diffAbs, dim=0)
    std = torch.std(diffAbs, dim=0) + 1e-5 # 应该是每个元素都加了

    print("mean,conv shape: ", mean.shape, "  ",std.shape)
    return mean, std # 都是 [feature_dim,]



def anomalyScore_transformer(args, model, dataset, means, covs):
    """

    :param args:
    :param model:
    :param dataset:     [len, 1, feature_dim]
    :param mean:
    :param cov:
    :return: scores_transformer shape:
    """
    seq_len = args.bptt

    diffAbs = None

    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        for i in range(0, dataset.shape[0] - seq_len):
            # 最后一个序列段开始的下标 len - seq_len + 1, 结束的下标 len - seq_len - 1 + seq_len - 1 = len - 2
            seq = dataset[i: i+seq_len]
            target_seq = dataset[i+1: i+1+seq_len] # [seq_len, 1, feature_dim]

            output, _ = model(seq, hidden=0, return_hiddens=False) # [seq_len, 1, feature_dim]

            if(diffAbs is None):
                diffAbs = output[-1] - target_seq[-1] # [1, feature_dim]
            else:
                diffAbs = torch.cat((diffAbs, output[-1] - target_seq[-1]), dim=0)

            # diffAbs : [len-seq, feature_dim]

    diffAbs = torch.abs(diffAbs - means)
    diffAbs = diffAbs * diffAbs
    diffAbs = diffAbs / (covs * covs)

    score_transformer = diffAbs

    return score_transformer # [len-seq, feature_dim]



def anomalyScore(args, model, dataset, mean, cov, channel_idx=0, score_predictor=None):
    """
    dataset         :[length,1,feature_dim]
    mean            :[10]
    cov             :[10,10]
    
    测试的时候，时间长度是1
    
    """
    predictions = []
    rearranged = []
    errors = []
    hiddens = []
    predicted_scores = []
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        #pasthidden = model.init_hidden(1)
        for t in range(len(dataset)):
            # out, hidden = model.forward(dataset[t].unsqueeze(0), pasthidden)
            out, hidden = model.forward(dataset[t].unsqueeze(0),0)
            # 输入 [1,1,feature_dim]
            # out [1,1,feature_dim]
            predictions.append([])
            rearranged.append([])
            errors.append([])
            #hiddens.append(model.extract_hidden(hidden))  # 依据model的类型，还是返回hidden里面的东西
            #if score_predictor is not None: # 默认是None的！
            #    predicted_scores.append(score_predictor.predict(model.extract_hidden(hidden).numpy()))

            predictions[t].append(out.data.cpu()[0][0][channel_idx])
                                                # predictions[t] 就是dataset t时刻输入，的输出
                                                # 就是对下一个时刻的预测
            #pasthidden = model.repackage_hidden(hidden)
            for prediction_step in range(1, args.prediction_window_size): # 又是往后预测window个值
                                                # prediction_step 从 1 到 9
                out, hidden = model.forward(out, hidden)  # 输入一步，输出也是一步
                                                          # 只给一个当前值，往后预测10步
                predictions[t].append(out.data.cpu()[0][0][channel_idx])
                                                            # prediction[t] 内含输入dataset[t]然后往后
                                                            # 预测10步的预测值

            if t >= args.prediction_window_size:
                for step in range(args.prediction_window_size):  # step 从0到9
                    rearranged[t].append(
                        predictions[step + t - args.prediction_window_size][args.prediction_window_size - 1 - step])
                                                            # rearrange[t] 是对t时刻的10个预测值
                rearranged[t] =torch.FloatTensor(rearranged[t]).to(args.device).unsqueeze(0)
                                                            # shape [1,10]
                errors[t] = rearranged[t] - dataset[t][0][channel_idx]
            else:
                rearranged[t] = torch.zeros(1,args.prediction_window_size).to(args.device)
                errors[t] = torch.zeros(1, args.prediction_window_size).to(args.device)

    predicted_scores = np.array(predicted_scores)  # 默认的话，其实还是空
    scores = []
    for error in errors:  # errors是list 每个元素含有10个值 如何看成二维矩阵 errors : [time,10]
        mult1 = error-mean.unsqueeze(0) # [ 1 * prediction_window_size ]  # mean [10,]
                                        # 上面这个unsqueeze有用吗..
                                        # 有用，维度扩展！
                                        # mult1 [1,10]
                                        # 是行向量
        mult2 = torch.inverse(cov) # [ prediction_window_size * prediction_window_size ]
                                   # inverse 逆矩阵
        mult3 = mult1.t() # [ prediction_window_size * 1 ]  
        score = torch.mm(mult1,torch.mm(mult2,mult3))  # 标量，该时刻的异常分数，越高越异常
        scores.append(score[0][0])  # 取两次值才能取到标量 

    scores = torch.stack(scores)  # size [time_length,]
    rearranged = torch.cat(rearranged,dim=0)  # size [time_length,10]
    errors = torch.cat(errors,dim=0)  # size [time_length,10]

    return scores, rearranged, errors, hiddens, predicted_scores


def get_precision_recall(args, score, label, num_samples, beta=1.0, sampling='log', predicted_score=None):
    '''
    :param args:
    :param score: anomaly scores
    :param label: anomaly labels
    :param num_samples                  : the number of threshold samples  # 默认1000
    :param beta                         :beta value for f-beta score 默认是1！
    :param scale:
    :return:
                                sample 默认 predicted_score 有默认情况
    
    '''
    if predicted_score is not None:
        score = score - torch.FloatTensor(predicted_score).squeeze().to(args.device)

    maximum = score.max()  # score [length,]
    if sampling=='log':
        # Sample thresholds logarithmically
        # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
        th = torch.logspace(0, torch.log10(torch.tensor(maximum)), num_samples).to(args.device)
    else:
        # Sample thresholds equally
        # The sampled thresholds are equally spaced points between: attr:`start` and: attr:`end`
        th = torch.linspace(0, maximum, num_samples).to(args.device)

    precision = []
    recall = []

    for i in range(len(th)):
        anomaly = (score > th[i]).float()  # [false,false,true,true....]这种
                                           # .float 转化为 [0,0,1,1...] 1代表异常
        idx = anomaly * 2 + label          # 异常-> 2+0||1   ->  2||3
                                           # 正常-> 0+0||1   ->  0||1
        tn = (idx == 0.0).sum().item()  # tn
        fn = (idx == 1.0).sum().item()  # fn
        fp = (idx == 2.0).sum().item()  # fp
        tp = (idx == 3.0).sum().item()  # tp

        p = tp / (tp + fp + 1e-7)  # precision
        r = tp / (tp + fn + 1e-7)  # recall

        if p != 0 and r != 0:
            precision.append(p)
            recall.append(r)

    precision = torch.FloatTensor(precision)
    recall = torch.FloatTensor(recall)


    f1 = (1 + beta ** 2) * (precision * recall).div(beta ** 2 * (precision + recall) + 1e-7)

    return precision, recall, f1


def get_precision_recall_zsx(args, score, label, num_samples, beta=1.0, sampling='log', predicted_score=None):
    """
    是对原版get_precision_recall 的一个简单调整，增加了预测调整的步骤
    """
    if predicted_score is not None:
        score = score - torch.FloatTensor(predicted_score).squeeze().to(args.device)

    maximum = score.max()  # score [length,]
    if sampling=='log':
        # Sample thresholds logarithmically
        # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
        th = torch.logspace(0, torch.log10(torch.tensor(maximum)), num_samples).to(args.device)
    else:
        # Sample thresholds equally
        # The sampled thresholds are equally spaced points between: attr:`start` and: attr:`end`
        th = torch.linspace(0, maximum, num_samples).to(args.device)
        
    precision = []
    recall = []
    # 上面这段直接复制过来
    
    for index_th in range(len(th)):
        
        
        
        #anomaly = (score > th[i]).float()   # [false,false,true,true....]这种
                                            # .float 转化为 [0,0,1,1...] 1代表异常
                                            # 异常-> 2+0||1   ->  2||3
                                            # 正常-> 0+0||1   ->  0||1
                                            # 
        predict = score>th[index_th]               # [false,false,true,true]
                                            # true代表异常
        anomaly_state = False
        for i in range(len(predict)):
            if label[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if not label[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
            elif not label[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
                
        anomaly = predict.float()

               
                                           
        idx = anomaly * 2 + label          
                                           
        tn = (idx == 0.0).sum().item()  # tn
        fn = (idx == 1.0).sum().item()  # fn
        fp = (idx == 2.0).sum().item()  # fp
        tp = (idx == 3.0).sum().item()  # tp

        p = tp / (tp + fp + 1e-7)  # precision
        r = tp / (tp + fn + 1e-7)  # recall

        if p != 0 and r != 0:
            precision.append(p)
            recall.append(r)

    precision = torch.FloatTensor(precision)
    recall = torch.FloatTensor(recall)


    f1 = (1 + beta ** 2) * (precision * recall).div(beta ** 2 * (precision + recall) + 1e-7)

    return precision, recall, f1


def get_precision_recall_zsx_2(args, score, label, num_samples, beta=1.0, sampling='log', predicted_score=None):
    """
    进行预测调整，返回f1最好的，对该维度的预测结果：[false,false,true,true]
    """
    if predicted_score is not None:
        score = score - torch.FloatTensor(predicted_score).squeeze().to(args.device)

    maximum = score.max()  # score [length,]
    if sampling=='log':
        # Sample thresholds logarithmically
        # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
        th = torch.logspace(0, torch.log10(torch.tensor(maximum)), num_samples).to(args.device)
    else:
        # Sample thresholds equally
        # The sampled thresholds are equally spaced points between: attr:`start` and: attr:`end`
        th = torch.linspace(0, maximum, num_samples).to(args.device)
        
    precision = []
    recall = []
    # 上面这段直接复制过来
    
    max_f1 = 0.0
    th_max_f1 = 0.0
    
    for index_th in range(len(th)):
        
        
        
        #anomaly = (score > th[i]).float()   # [false,false,true,true....]这种
                                            # .float 转化为 [0,0,1,1...] 1代表异常
                                            # 异常-> 2+  0||1   ->  2||3
                                            # 正常-> 0+  0||1   ->  0||1
                                            # 
        predict = score>th[index_th]               # [false,false,true,true]
                                            # true代表异常
        anomaly_state = False
        for i in range(len(predict)):
            if label[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if not label[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
            elif not label[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
                
        anomaly = predict.float()

               
                                           
        idx = anomaly * 2 + label          
                                           
        tn = (idx == 0.0).sum().item()  # tn
        fn = (idx == 1.0).sum().item()  # fn
        fp = (idx == 2.0).sum().item()  # fp
        tp = (idx == 3.0).sum().item()  # tp

        p = tp / (tp + fp + 1e-7)  # precision
        r = tp / (tp + fn + 1e-7)  # recall

        p = torch.FloatTensor([p])[0]
        r = torch.FloatTensor([r])[0]

        f1 = (1 + beta ** 2) * (p * r).div(beta ** 2 * (p + r) + 1e-7)
        
        if(f1>max_f1):
            th_max_f1 = th[index_th]
            max_f1 = f1


    anomaly = (score > th_max_f1).float()

    #[0,1,0,1,1,1,1]
    return anomaly

def get_f1(anomaly_temp,label,beta=1.0):
    # 输入的anomaly_temp 应该是true false
    # 普通的根据预测结果和label计算f1的过程！
    # 当然还要进行一次异常调整！
    # 所以相当于用了两次预测调整
    predict = anomaly_temp
    # predict现在 true false
    anomaly_state = False
    
    for i in range(len(predict)):
        if label[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
                
    anomaly = predict.float()
    # [0,0,1,1,1]
    
    idx = anomaly * 2 + label          
                                           
    tn = (idx == 0.0).sum().item()  # tn
    fn = (idx == 1.0).sum().item()  # fn
    fp = (idx == 2.0).sum().item()  # fp
    tp = (idx == 3.0).sum().item()  # tp

    p = tp / (tp + fp + 1e-7)  # precision
    r = tp / (tp + fn + 1e-7)  # recall

    p = torch.FloatTensor([p])[0]
    r = torch.FloatTensor([r])[0]

    f1 = (1 + beta ** 2) * (p * r).div(beta ** 2 * (p + r) + 1e-7)
        
    
    return f1