import argparse
import time
import torch
import torch.nn as nn
import preprocess_data
from model import model
from torch import optim
import visdom

from pathlib import Path
from anomalyDetector import fit_norm_distribution_param
from anomalyDetector import fit_norm_distribution_param_transformer

from torch.optim.lr_scheduler import MultiStepLR


# 有save-fig参数！
# 默认400个周期

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset')
parser.add_argument('--data', type=str, default='ecg',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                    help='filename of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--augment', type=bool, default=True,
                    help='augment')
parser.add_argument('--emsize', type=int, default=32, # RNN input dim size
                    help='size of rnn input features')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
# 下面这个参数，加上--res_connection 就是true，否则就是false
parser.add_argument('--res_connection', action='store_true',
                    help='residual connection')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=10,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio (deprecated)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval')
parser.add_argument('--save_fig', action='store_true',
                    help='save figure')
parser.add_argument('--resume','-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')

# 自己添加参数
parser.add_argument('--index', type=str)

parser.add_argument('--milestones', type=int, nargs='+', default=[1000])

parser.add_argument('--gamma', type=float, default=0.1)

args = parser.parse_args()
# Set the random seed manually for reproducibility.

# seed = 1111
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

###############################################################################
# Load data
###############################################################################
TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename,
                                                augment_test_data=args.augment)
train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData, args.batch_size)
test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, args.eval_batch_size)
gen_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, 1)  # 这个是干什么用的
                                                                        # 用来画图的

# batchify 返回的东西已经是 真实的数据了！ 比如对于ecg 就是 [1562,64,2]
#                                                      size:[nbatch,batchsize,feature_dim]
#                                                      size:[newtime,batchsize,feature_dim]

print("train_dataset.shape: ",train_dataset.shape)

###############################################################################
# Build the model
###############################################################################
feature_dim = TimeseriesData.trainData.size(1)
model = model.RNNPredictor(rnn_type = args.model,  
                           enc_inp_size=feature_dim,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=feature_dim,
                           nlayers = args.nlayers,
                           dropout = args.dropout,
                           res_connection=args.res_connection).to(args.device)

optimizer = optim.Adam(model.parameters(), lr= args.lr,weight_decay=args.weight_decay)

# 步长递减
scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

criterion = nn.MSELoss()
###############################################################################
# Training code
###############################################################################
def get_batch(args,source, i):
    """
    bptt        :sequence length 默认50
    i           :从0到size-1，间隔为bptt！
    source      :train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData, args.batch_size)
                :source 的 size :[1562,64,2] 第一个维度是时间
    """
    
    seq_len = min(args.bptt, len(source) - 1 - i)
                                # i比较小的时候，seq_len 都是 bptt
    
    data = source[i:i+seq_len] # [ seq_len * batch_size * feature_size ]
                               # 注意真实的维度是多少
                               # 现在我要知道，source直接用下标出来的什么
                               
    target = source[i+1:i+1+seq_len] # [ (seq_len x batch_size x feature_size) ]
                                     # [bptt,batch_size,feature_dim]
                                     # 注意，target是data时间序列往后一个时间单位
    return data, target




def train(args, model, train_dataset,epoch):

    with torch.enable_grad():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0

        hidden = model.init_hidden(args.batch_size)
        epochLoss = 0.0
        for batch, i in enumerate(range(0, train_dataset.size(0) - 1, 20)):
                                                                    # bptt sequence length 默认 50
                                                                    # i 从0到size-1，间隔为bptt！
                                                                    # 结合get-batch可以看到
                                                                    # 这里数据的使用，如果数据是[1,2,3,4,5,6,7,8,9]
                                                                    # 那么使用[1,2,3][4,5,6][7,8,9]
                                                                    # 而不是 [1,2,3][2,3,4],[3,4,5]...这种
                                                                    
                                                                    # 注意底下这行的get-batch
                                                                    # input是[1,2,3]
                                                                    # target是[2,3,4]
            inputSeq, targetSeq = get_batch(args,train_dataset, i)
            # [bptt,batch_size,feature_dim] 两个数据都是这个size
            
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # hidden = model.repackage_hidden(hidden)   # 用来做loss2
            # hidden_ = model.repackage_hidden(hidden)  # 用来做loss1
            optimizer.zero_grad()

            '''Loss1: Free running loss'''
            # outVal = inputSeq[0].unsqueeze(0)  # 维度:[1,batch_size,feature_dim]
            # outVals=[]
            # hids1 = []
            # for i in range(inputSeq.size(0)):  # 这个循环体现 free running 从一个真实的输入折射很多输出
            #                                    # 因为你看上面，只用来额inputseq[0]
            #                                    # inputSet.size(0) = seq_len
            #     outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
            #                                     # return_hiddens 是 True ，那么返回值就有三个
            #                                     # outVal的维度 [1,batchsize,hidden_size]
            #                                     # hid的维度 [1,batchsize,hidden_size]
            #     outVals.append(outVal)
            #     hids1.append(hid)
            # outSeq1 = torch.cat(outVals,dim=0)  # 维度[bptt,batchsize,hiddne_size]
            # hids1 = torch.cat(hids1,dim=0)      # 维度[bptt,batchsize,hidden_size]
            #
            # loss1 = criterion(outSeq1.reshape(args.batch_size,-1), targetSeq.reshape(args.batch_size,-1))

            '''Loss2: Teacher forcing loss'''  # 意义见README

            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
                                                # outseq2 [bptt,batchsize,hidden_size]
            loss2 = criterion(outSeq2.reshape(args.batch_size, -1), targetSeq.reshape(args.batch_size, -1))

            '''Loss3: Simplified Professor forcing loss'''
            # loss3 = criterion(hids1.reshape(args.batch_size,-1), hids2.reshape(args.batch_size,-1).detach())
            #
            # '''Total loss = Loss1+Loss2+Loss3'''
            # loss = loss1+loss2+loss3

            loss = loss2

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()
            epochLoss += total_loss
            total_loss = 0.0

        scheduler.step()

        print('| epoch {:3d} | epochLoss {:5.2f} '.format(epoch,epochLoss))
        return epochLoss*(args.batch_size / 64)



def evaluate(args, model, test_dataset):  # 是train函数的无敌简化版,就是测试函数！每训练完一个周期就调用
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
        hidden = model.init_hidden(args.eval_batch_size)
        nbatch = 1
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args,test_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]

            # hidden_ = model.repackage_hidden(hidden)
            # '''Loss1: Free running loss'''
            # outVal = inputSeq[0].unsqueeze(0)
            # outVals=[]
            # hids1 = []
            # for i in range(inputSeq.size(0)):
            #     outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
            #     outVals.append(outVal)
            #     hids1.append(hid)
            # outSeq1 = torch.cat(outVals,dim=0)
            # hids1 = torch.cat(hids1,dim=0)
            # loss1 = criterion(outSeq1.reshape(args.eval_batch_size,-1), targetSeq.reshape(args.eval_batch_size,-1))

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2.reshape(args.eval_batch_size, -1), targetSeq.reshape(args.eval_batch_size, -1))

            '''Loss3: Simplified Professor forcing loss'''
            # loss3 = criterion(hids1.reshape(args.eval_batch_size,-1), hids2.reshape(args.eval_batch_size,-1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            # loss = loss1+loss2+loss3
            loss = loss2

            total_loss += loss.item()

    return total_loss / (nbatch+1)



#
allEpochLoss = []
epochIndex = []
vis = visdom.Visdom(port=2666)



epoch = 1
start_epoch = 1
best_val_loss = float('inf')
print("=> Start training from scratch")

print('-' * 89)
print(args)
print('-' * 89)

if not args.pretrained:  # 模型训练 默认False ，需要训练，就执行这个分支
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, args.epochs+1):

            epoch_start_time = time.time()
            epochLoss = train(args,model,train_dataset,epoch)
            # 使用visdom观察loss
            allEpochLoss.append(epochLoss)
            epochIndex.append(epoch)
            winName = args.filename.split(".")[0]+str(args.index)
            vis.line(allEpochLoss,epochIndex,win=winName,opts=dict(title=winName))

            val_loss = evaluate(args,model,test_dataset)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time),                                                                                        val_loss))
            print('-' * 89)

            #generate_output(args,epoch,model,gen_dataset,startPoint=1500)
            # 画一些当前表现的图片！如果save-fig是true，默认不画

            if epoch%args.save_interval==0:
                # Save the model if the validation loss is the best we've seen so far.
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch,
                                    'best_loss': best_val_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'args':args
                                    }
                model.save_checkpoint(model_dictionary, is_best)  # 需要更改

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


# Calculate mean and covariance for each channel's prediction errors, and save them with the trained model
print('=> calculating mean and covariance')
means, covs = list(),list()
train_dataset = TimeseriesData.batchify(args, TimeseriesData.trainData, bsz=args.batch_size)

# for channel_idx in range(model.enc_input_size):  # 默认 feature_dim
#     mean, cov = fit_norm_distribution_param(args,model,train_dataset[:TimeseriesData.length],channel_idx)
#     means.append(mean), covs.append(cov)
#     print("已完成第",channel_idx,"个维度的mean和cov计算")
means, covs = fit_norm_distribution_param_transformer(args, model, train_dataset)

model_dictionary = {'epoch': max(epoch,start_epoch),
                    'best_loss': best_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'means': means,
                    'covs': covs
                    }
model.save_checkpoint(model_dictionary, True)
print('-' * 89)