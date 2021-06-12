import argparse
import time
import torch
import torch.nn as nn
import preprocess_data
from model import model_FL # 更改
from torch import optim
import visdom
from matplotlib import pyplot as plt
from pathlib import Path
from anomalyDetector import fit_norm_distribution_param
import numpy as np
import random

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
parser.add_argument('--emsize', type=int, default=32,
                    help='size of rnn input features')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
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
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights (deprecated)')
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
parser.add_argument('--resume', '-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained', '-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

###############################################################################
# Load data 需要重新设计，因为需要分开不同的维度
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
#                                                           [nbatch,bsz,dim]


###############################################################################
# Build the model
###############################################################################
#feature_dim = TimeseriesData.trainData.size(1)

feature_data = TimeseriesData.trainData.size(1)
feature_dim = 8+8+8

"""
1.先从encoder没有非线性激活的情况开始
"""

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
                                       dropout=args.dropout,
                                       tie_weights=args.tied,
                                       res_connection=args.res_connection
                                       ).to(args.device)

optimizer_model_server_0 = optim.Adam(model_server_0.parameters(), lr= args.lr,weight_decay=args.weight_decay)

optimizer_encoder_client_0 = optim.Adam(encoder_client_0.parameters(), lr= args.lr,weight_decay=args.weight_decay)
optimizer_encoder_client_1 = optim.Adam(encoder_client_1.parameters(), lr= args.lr,weight_decay=args.weight_decay)
optimizer_encoder_client_2 = optim.Adam(encoder_client_2.parameters(), lr= args.lr,weight_decay=args.weight_decay)

optimizer_decoder_client_0 = optim.Adam(decoder_client_0.parameters(), lr= args.lr,weight_decay=args.weight_decay)
optimizer_decoder_client_1 = optim.Adam(decoder_client_1.parameters(), lr= args.lr,weight_decay=args.weight_decay)
optimizer_decoder_client_2 = optim.Adam(decoder_client_2.parameters(), lr= args.lr,weight_decay=args.weight_decay)


criterion = nn.MSELoss()

dimensions_client_0 = [0,1,2,3]
dimensions_client_1 = [4,5,6]
dimensions_client_2 = [7,8,9]

recover_dimensions_client_0 = [0,1,2,3,4,5,6,7]
recover_dimensions_client_1 = [8,9,10,11,12,13,14,15]
recover_dimensions_client_2 = [16,17,18,19,20,21,22,23]

###############################################################################
# Training code
###############################################################################

def get_batch(args, source, i):
    """
    bptt        :sequence length 默认50
    i           :从0到size-1，间隔为bptt！
    source      :train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData, args.batch_size)
                :source 的 size :[1562,64,2] 第一个维度是时间
    """

    seq_len = min(args.bptt, len(source) - 1 - i)
    # i比较小的时候，seq_len 都是 bptt

    data = source[i:i + seq_len]  # [ seq_len * batch_size * feature_size ]
    # 注意真实的维度是多少
    # 现在我要知道，source直接用下标出来的什么

    target = source[i + 1:i + 1 + seq_len]  # [ (seq_len x batch_size x feature_size) ]
    # [bptt,batch_size,feature_dim]
    # 注意，target是data时间序列往后一个时间单位
    return data, target


def evaluate_1step_pred(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                             decoder_client_0,decoder_client_1,decoder_client_2,test_dataset):
    # Turn on evaluation mode which disables dropout.
    model_server_0.eval()


    total_loss = 0
    with torch.no_grad():
        hidden = model_server_0.init_hidden(args.eval_batch_size)
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):

            inputSeq, targetSeq = get_batch(args,test_dataset, i)

            # inputSeq :[ seq_len , batch_size , feature_size ]
            # targetSeq:[ seq_len , batch_size , feature_size ]
                                                                # 重点需要看这个输入输出数据的维度
            inputSeq_client_0 = inputSeq[:,:,dimensions_client_0] # [seq_len,batchsize,feature_0]
            inputSeq_client_1 = inputSeq[:,:,dimensions_client_1] # [seq_len,batchsize,feature_1]
            inputSeq_client_2 = inputSeq[:,:,dimensions_client_2] # [seq_len,batchsize,feature_2]

            seq_len = inputSeq.shape[0]
            batch_size = inputSeq.shape[1]

            inputSeq_client_0 = inputSeq_client_0.reshape(seq_len*batch_size,-1)
            inputSeq_client_1 = inputSeq_client_1.reshape(seq_len*batch_size,-1)
            inputSeq_client_2 = inputSeq_client_2.reshape(seq_len*batch_size,-1)

            in_0 = encoder_client_0.forward(inputSeq_client_0) # [seq_len*batchsize,8]
            in_1 = encoder_client_1.forward(inputSeq_client_1) # [seq_len*batchsize,8]
            in_2 = encoder_client_2.forward(inputSeq_client_2) # [seq_len*batchsize,8]

            in_0 = in_0.reshape(seq_len,batch_size,-1)
            in_1 = in_1.reshape(seq_len,batch_size,-1)
            in_2 = in_2.reshape(seq_len,batch_size,-1)

            in_cat = torch.cat((in_0,in_1,in_2),2)

            outSeq,hidden = model_server_0.forward(in_cat,hidden)
            # [seq_len, batchsize, feature_dim] feature_dim 应该是24

            out_0 = outSeq[:,:,recover_dimensions_client_0].reshape(seq_len*batch_size,-1)
            out_1 = outSeq[:,:,recover_dimensions_client_1].reshape(seq_len*batch_size,-1)
            out_2 = outSeq[:,:,recover_dimensions_client_2].reshape(seq_len*batch_size,-1)

            recover_0 = decoder_client_0.forward(out_0) # [seq_len*batchsize,4]
            recover_1 = decoder_client_1.forward(out_1) # [seq_len*batchsize,3]
            recover_2 = decoder_client_2.forward(out_2) # [seq_len*batchsize,3]

            recover_0 = recover_0.reshape(seq_len,batch_size,-1)
            recover_1 = recover_1.reshape(seq_len,batch_size,-1)
            recover_2 = recover_2.reshape(seq_len,batch_size,-1)

            recover_cat = torch.cat((recover_0,recover_1,recover_2),2) # [seq_len,barch_size,10]

            loss = criterion(recover_cat.reshape(args.batch_size, -1), targetSeq.reshape(args.batch_size, -1))
            hidden = model_server_0.repackage_hidden(hidden)
            total_loss += loss.item()



            #outSeq, hidden = model.forward(inputSeq, hidden)
            #loss = criterion(outSeq.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))
            #hidden = model.repackage_hidden(hidden)
            #total_loss+= loss.item()

    return total_loss / nbatch


def train(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                               decoder_client_0,decoder_client_1,decoder_client_2,train_dataset, epoch):
    with torch.enable_grad():
        # Turn on training mode which enables dropout.
        model_server_0.train()
        total_loss = 0
        start_time = time.time()
        hidden = model_server_0.init_hidden(args.batch_size)

        seq_len_free_running = 1
        batch_size_train = args.batch_size

        epochLoss = 0.0

        for batch, i in enumerate(range(0, train_dataset.size(0) - 1, args.bptt)):
            # bptt sequence length 默认 50
            # i 从0到size-1，间隔为bptt！
            # 结合get-batch可以看到
            # 这里数据的使用，如果数据是[1,2,3,4,5,6,7,8,9]
            # 那么使用[1,2,3][4,5,6][7,8,9]
            # 而不是 [1,2,3][2,3,4],[3,4,5]...这种

            # 注意底下这行的get-batch
            # input是[1,2,3]
            # target是[2,3,4]
            inputSeq, targetSeq = get_batch(args, train_dataset, i)
            # [bptt,batch_size,feature_dim] 两个数据都是这个size

            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]

            seq_len_teacher_forcing = inputSeq.shape[0]


            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model_server_0.repackage_hidden(hidden)  # 用来做loss2
            hidden_ = model_server_0.repackage_hidden(hidden)  # 用来做loss1

            optimizer_model_server_0.zero_grad()
            optimizer_encoder_client_0.zero_grad()
            optimizer_encoder_client_1.zero_grad()
            optimizer_encoder_client_2.zero_grad()
            optimizer_decoder_client_0.zero_grad()
            optimizer_decoder_client_1.zero_grad()
            optimizer_decoder_client_2.zero_grad()


            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)  # 维度:[1,batch_size,feature_dim]
            outVals = []
            hids1 = []
            for i in range(inputSeq.size(0)):  # 这个循环体现 free running 从一个真实的输入折射很多输出

                inputSeq_client_0 = outVal[:, :, dimensions_client_0]  # [seq_len,batchsize,feature_0]
                inputSeq_client_1 = outVal[:, :, dimensions_client_1]  # [seq_len,batchsize,feature_1]
                inputSeq_client_2 = outVal[:, :, dimensions_client_2]  # [seq_len,batchsize,feature_2]

                inputSeq_client_0 = inputSeq_client_0.reshape(seq_len_free_running * batch_size_train, -1)
                inputSeq_client_1 = inputSeq_client_1.reshape(seq_len_free_running * batch_size_train, -1)
                inputSeq_client_2 = inputSeq_client_2.reshape(seq_len_free_running * batch_size_train, -1)

                in_0 = encoder_client_0.forward(inputSeq_client_0)  # [seq_len*batchsize,8]
                in_1 = encoder_client_1.forward(inputSeq_client_1)  # [seq_len*batchsize,8]
                in_2 = encoder_client_2.forward(inputSeq_client_2)  # [seq_len*batchsize,8]

                in_0 = in_0.reshape(seq_len_free_running, batch_size_train, -1)
                in_1 = in_1.reshape(seq_len_free_running, batch_size_train, -1)
                in_2 = in_2.reshape(seq_len_free_running, batch_size_train, -1)

                in_cat = torch.cat((in_0, in_1, in_2), 2)

                out, hidden_, hid = model_server_0.forward(in_cat, hidden_, return_hiddens=True)

                out_0 = out[:, :, recover_dimensions_client_0].reshape(seq_len_free_running * batch_size_train, -1)
                out_1 = out[:, :, recover_dimensions_client_1].reshape(seq_len_free_running * batch_size_train, -1)
                out_2 = out[:, :, recover_dimensions_client_2].reshape(seq_len_free_running * batch_size_train, -1)

                recover_0 = decoder_client_0.forward(out_0)  # [seq_len*batchsize,4]
                recover_1 = decoder_client_1.forward(out_1)  # [seq_len*batchsize,3]
                recover_2 = decoder_client_2.forward(out_2)  # [seq_len*batchsize,3]

                recover_0 = recover_0.reshape(seq_len_free_running, batch_size_train, -1)
                recover_1 = recover_1.reshape(seq_len_free_running, batch_size_train, -1)
                recover_2 = recover_2.reshape(seq_len_free_running, batch_size_train, -1)

                outVal = torch.cat((recover_0, recover_1, recover_2), 2)  # [seq_len,barch_size,10]

                outVals.append(outVal)
                hids1.append(hid)
                """
                # 因为你看上面，只用来额inputseq[0]
                # inputSet.size(0) = seq_len
                outVal, hidden_, hid = model.forward(outVal, hidden_, return_hiddens=True)
                # return_hiddens 是 True ，那么返回值就有三个
                # outVal的维度 [1,batchsize,hidden_size]
                # hid的维度 [1,batchsize,hidden_size]
                outVals.append(outVal)
                hids1.append(hid)
                """
            outSeq1 = torch.cat(outVals, dim=0)  # 维度[bptt,batchsize,hiddne_size]
            hids1 = torch.cat(hids1, dim=0)  # 维度[bptt,batchsize,hidden_size]
            loss1 = criterion(outSeq1.reshape(args.batch_size, -1), targetSeq.reshape(args.batch_size, -1))

            '''Loss2: Teacher forcing loss'''  #
            # 这些变量的名字取一样的，有影响吗？
            inputSeq_client_0 = inputSeq[:, :, dimensions_client_0]  # [seq_len,batchsize,feature_0]
            inputSeq_client_1 = inputSeq[:, :, dimensions_client_1]  # [seq_len,batchsize,feature_1]
            inputSeq_client_2 = inputSeq[:, :, dimensions_client_2]  # [seq_len,batchsize,feature_2]

            inputSeq_client_0 = inputSeq_client_0.reshape(seq_len_teacher_forcing * batch_size_train, -1)
            inputSeq_client_1 = inputSeq_client_1.reshape(seq_len_teacher_forcing * batch_size_train, -1)
            inputSeq_client_2 = inputSeq_client_2.reshape(seq_len_teacher_forcing * batch_size_train, -1)

            in_0 = encoder_client_0.forward(inputSeq_client_0)  # [seq_len*batchsize,8]
            in_1 = encoder_client_1.forward(inputSeq_client_1)  # [seq_len*batchsize,8]
            in_2 = encoder_client_2.forward(inputSeq_client_2)  # [seq_len*batchsize,8]

            in_0 = in_0.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            in_1 = in_1.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            in_2 = in_2.reshape(seq_len_teacher_forcing, batch_size_train, -1)

            in_cat = torch.cat((in_0, in_1, in_2), 2)

            out, hidden, hids2 = model_server_0.forward(in_cat, hidden, return_hiddens=True)

            out_0 = out[:, :, recover_dimensions_client_0].reshape(seq_len_teacher_forcing * batch_size_train, -1)
            out_1 = out[:, :, recover_dimensions_client_1].reshape(seq_len_teacher_forcing * batch_size_train, -1)
            out_2 = out[:, :, recover_dimensions_client_2].reshape(seq_len_teacher_forcing * batch_size_train, -1)

            recover_0 = decoder_client_0.forward(out_0)  # [seq_len*batchsize,4]
            recover_1 = decoder_client_1.forward(out_1)  # [seq_len*batchsize,3]
            recover_2 = decoder_client_2.forward(out_2)  # [seq_len*batchsize,3]

            recover_0 = recover_0.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            recover_1 = recover_1.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            recover_2 = recover_2.reshape(seq_len_teacher_forcing, batch_size_train, -1)

            outSeq2 = torch.cat((recover_0, recover_1, recover_2), 2)  # [seq_len,barch_size,10]

            loss2 = criterion(outSeq2.reshape(args.batch_size, -1), targetSeq.reshape(args.batch_size, -1))

            """
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            # outseq2 [bptt,batchsize,hidden_size]
            loss2 = criterion(outSeq2.view(args.batch_size, -1), targetSeq.view(args.batch_size, -1))
            """

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.reshape(args.batch_size, -1), hids2.reshape(args.batch_size, -1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1 + loss2 + loss3

            # print(loss1,loss2,loss3)

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model_server_0.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(encoder_client_0.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(encoder_client_1.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(encoder_client_2.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(decoder_client_0.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(decoder_client_1.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(decoder_client_2.parameters(), args.clip)

            optimizer_model_server_0.step()
            optimizer_encoder_client_0.step()
            optimizer_encoder_client_1.step()
            optimizer_encoder_client_2.step()
            optimizer_decoder_client_0.step()
            optimizer_decoder_client_1.step()
            optimizer_decoder_client_2.step()


            total_loss += loss.item()
            epochLoss += total_loss
            total_loss = 0.0

        print('| epoch {:3d} | epochLoss {:5.2f} '.format(epoch, epochLoss))
        return epochLoss

def evaluate(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                  decoder_client_0,decoder_client_1,decoder_client_2,test_dataset):  # 是train函数的无敌简化版,就是测试函数！每训练完一个周期就调用
    # Turn on evaluation mode which disables dropout.
    model_server_0.eval()
    with torch.no_grad():
        total_loss = 0
        hidden = model_server_0.init_hidden(args.eval_batch_size)
        nbatch = 1
        #
        seq_len_free_running = 1
        batch_size_train = args.eval_batch_size

        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args,test_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]
            hidden_ = model_server_0.repackage_hidden(hidden)

            seq_len_teacher_forcing = inputSeq.shape[0]

            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals = []
            hids1 = []

            for i in range(inputSeq.size(0)):
                inputSeq_client_0 = outVal[:, :, dimensions_client_0]  # [seq_len,batchsize,feature_0]
                inputSeq_client_1 = outVal[:, :, dimensions_client_1]  # [seq_len,batchsize,feature_1]
                inputSeq_client_2 = outVal[:, :, dimensions_client_2]  # [seq_len,batchsize,feature_2]

                inputSeq_client_0 = inputSeq_client_0.reshape(seq_len_free_running * batch_size_train, -1)
                inputSeq_client_1 = inputSeq_client_1.reshape(seq_len_free_running * batch_size_train, -1)
                inputSeq_client_2 = inputSeq_client_2.reshape(seq_len_free_running * batch_size_train, -1)

                in_0 = encoder_client_0.forward(inputSeq_client_0)  # [seq_len*batchsize,8]
                in_1 = encoder_client_1.forward(inputSeq_client_1)  # [seq_len*batchsize,8]
                in_2 = encoder_client_2.forward(inputSeq_client_2)  # [seq_len*batchsize,8]

                in_0 = in_0.reshape(seq_len_free_running, batch_size_train, -1)
                in_1 = in_1.reshape(seq_len_free_running, batch_size_train, -1)
                in_2 = in_2.reshape(seq_len_free_running, batch_size_train, -1)

                in_cat = torch.cat((in_0, in_1, in_2), 2)

                out, hidden_, hid = model_server_0.forward(in_cat, hidden_, return_hiddens=True)

                out_0 = out[:, :, recover_dimensions_client_0].reshape(seq_len_free_running * batch_size_train, -1)
                out_1 = out[:, :, recover_dimensions_client_1].reshape(seq_len_free_running * batch_size_train, -1)
                out_2 = out[:, :, recover_dimensions_client_2].reshape(seq_len_free_running * batch_size_train, -1)

                recover_0 = decoder_client_0.forward(out_0)  # [seq_len*batchsize,4]
                recover_1 = decoder_client_1.forward(out_1)  # [seq_len*batchsize,3]
                recover_2 = decoder_client_2.forward(out_2)  # [seq_len*batchsize,3]

                recover_0 = recover_0.reshape(seq_len_free_running, batch_size_train, -1)
                recover_1 = recover_1.reshape(seq_len_free_running, batch_size_train, -1)
                recover_2 = recover_2.reshape(seq_len_free_running, batch_size_train, -1)

                outVal = torch.cat((recover_0, recover_1, recover_2), 2)  # [seq_len,barch_size,10]

                outVals.append(outVal)
                hids1.append(hid)

            outSeq1 = torch.cat(outVals, dim=0)
            hids1 = torch.cat(hids1, dim=0)
            #loss1 = criterion(outSeq1.reshape(args.batch_size, -1), targetSeq.reshape(args.batch_size, -1))
            # 重大bug修改
            loss1 = criterion(outSeq1.reshape(batch_size_train, -1), targetSeq.reshape(batch_size_train, -1))

            """
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals,dim=0)
            hids1 = torch.cat(hids1,dim=0)
            loss1 = criterion(outSeq1.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))
            """


            '''Loss2: Teacher forcing loss'''
            inputSeq_client_0 = inputSeq[:, :, dimensions_client_0]  # [seq_len,batchsize,feature_0]
            inputSeq_client_1 = inputSeq[:, :, dimensions_client_1]  # [seq_len,batchsize,feature_1]
            inputSeq_client_2 = inputSeq[:, :, dimensions_client_2]  # [seq_len,batchsize,feature_2]

            inputSeq_client_0 = inputSeq_client_0.reshape(seq_len_teacher_forcing * batch_size_train, -1)
            inputSeq_client_1 = inputSeq_client_1.reshape(seq_len_teacher_forcing * batch_size_train, -1)
            inputSeq_client_2 = inputSeq_client_2.reshape(seq_len_teacher_forcing * batch_size_train, -1)

            in_0 = encoder_client_0.forward(inputSeq_client_0)  # [seq_len*batchsize,8]
            in_1 = encoder_client_1.forward(inputSeq_client_1)  # [seq_len*batchsize,8]
            in_2 = encoder_client_2.forward(inputSeq_client_2)  # [seq_len*batchsize,8]

            in_0 = in_0.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            in_1 = in_1.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            in_2 = in_2.reshape(seq_len_teacher_forcing, batch_size_train, -1)

            in_cat = torch.cat((in_0, in_1, in_2), 2)

            out, hidden, hids2 = model_server_0.forward(in_cat, hidden, return_hiddens=True)

            out_0 = out[:, :, recover_dimensions_client_0].reshape(seq_len_teacher_forcing * batch_size_train, -1)
            out_1 = out[:, :, recover_dimensions_client_1].reshape(seq_len_teacher_forcing * batch_size_train, -1)
            out_2 = out[:, :, recover_dimensions_client_2].reshape(seq_len_teacher_forcing * batch_size_train, -1)

            recover_0 = decoder_client_0.forward(out_0)  # [seq_len*batchsize,4]
            recover_1 = decoder_client_1.forward(out_1)  # [seq_len*batchsize,3]
            recover_2 = decoder_client_2.forward(out_2)  # [seq_len*batchsize,3]

            recover_0 = recover_0.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            recover_1 = recover_1.reshape(seq_len_teacher_forcing, batch_size_train, -1)
            recover_2 = recover_2.reshape(seq_len_teacher_forcing, batch_size_train, -1)

            outSeq2 = torch.cat((recover_0, recover_1, recover_2), 2)  # [seq_len,barch_size,10]

            loss2 = criterion(outSeq2.reshape(batch_size_train, -1), targetSeq.reshape(batch_size_train, -1))

            """
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2.view(args.batch_size, -1), targetSeq.view(args.batch_size, -1))
            """


            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.reshape(batch_size_train,-1), hids2.reshape(batch_size_train,-1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1+loss2+loss3

            total_loss += loss.item()

    return total_loss / (nbatch+1)


allEpochLoss = []
epochIndex = []
vis = visdom.Visdom()

if args.resume or args.pretrained:
    # resume的话是继续训练，pretrained是不再训练了，默认False
    # 已有-sssxxx 或者 --ssssss 就是True了
    print("=> loading checkpoint ")
    checkpoint = torch.load(Path('save', args.data, 'checkpoint', args.filename).with_suffix('.pth'))
    args, start_epoch, best_val_loss = model_server_0.load_checkpoint(args,checkpoint,feature_dim) # dim = 24

    # 这里没有model_server load-state-dict 是因为 在 load-checkpoint里面就载入了参数了
    encoder_client_0.load_state_dict(checkpoint['state_dict']['encoder_client_0'])
    encoder_client_1.load_state_dict(checkpoint['state_dict']['encoder_client_1'])
    encoder_client_2.load_state_dict(checkpoint['state_dict']['encoder_client_2'])
    decoder_client_0.load_state_dict(checkpoint['state_dict']['decoder_client_0'])
    decoder_client_1.load_state_dict(checkpoint['state_dict']['decoder_client_1'])
    decoder_client_2.load_state_dict(checkpoint['state_dict']['decoder_client_2'])


    #optimizer.load_state_dict((checkpoint['optimizer']))

    optimizer_model_server_0.load_state_dict(checkpoint['optimizer_dict']['model_server_0'])  # 这里原先是两层括号，有必要吗
    optimizer_encoder_client_0.load_state_dict(checkpoint['optimizer_dict']['encoder_client_0'])
    optimizer_encoder_client_1.load_state_dict(checkpoint['optimizer_dict']['encoder_client_1'])
    optimizer_encoder_client_2.load_state_dict(checkpoint['optimizer_dict']['encoder_client_2'])
    optimizer_decoder_client_0.load_state_dict(checkpoint['optimizer_dict']['decoder_client_0'])
    optimizer_decoder_client_1.load_state_dict(checkpoint['optimizer_dict']['decoder_client_1'])
    optimizer_decoder_client_2.load_state_dict(checkpoint['optimizer_dict']['decoder_client_2'])

    state_dict = checkpoint['state_dict']
    optimizer_dict = checkpoint['optimizer_dict']


    del checkpoint
    epoch = start_epoch
    print("=> loaded checkpoint")
else:  # 这个分支用来重新训练
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
            epochLoss = train(args, model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                       decoder_client_0,decoder_client_1,decoder_client_2,train_dataset,epoch)

            # 使用visdom观察loss
            allEpochLoss.append(epochLoss)
            epochIndex.append(epoch)
            winName = args.filename.split(".")[0]+"_FL"
            vis.line(allEpochLoss, epochIndex, win=winName,opts=dict(title=winName))

            val_loss = evaluate(args,model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                       decoder_client_0,decoder_client_1,decoder_client_2,test_dataset)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time),                                                                                        val_loss))
            print('-' * 89)

            # generate_output(args,epoch,model,gen_dataset,startPoint=1500) # 暂时不生成图片了吧
            # 画一些当前表现的图片！如果save-fig是true，默认不画

            if epoch%args.save_interval==0:
                # Save the model if the validation loss is the best we've seen so far.
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)

                state_dict = {'model_server_0':model_server_0.state_dict(),
                              'encoder_client_0':encoder_client_0.state_dict(),
                              'encoder_client_1':encoder_client_1.state_dict(),
                              'encoder_client_2':encoder_client_2.state_dict(),
                              'decoder_client_0':decoder_client_0.state_dict(),
                              'decoder_client_1':decoder_client_1.state_dict(),
                              'decoder_client_2':decoder_client_2.state_dict(),}

                optimizer_dict = {'model_server_0':optimizer_model_server_0.state_dict(),
                                  'encoder_client_0':optimizer_encoder_client_0.state_dict(),
                                  'encoder_client_1':optimizer_encoder_client_1.state_dict(),
                                  'encoder_client_2':optimizer_encoder_client_2.state_dict(),
                                  'decoder_client_0':optimizer_decoder_client_0.state_dict(),
                                  'decoder_client_1':optimizer_decoder_client_1.state_dict(),
                                  'decoder_client_2':optimizer_decoder_client_2.state_dict(),}

                model_dictionary = {'epoch': epoch,
                                    'best_loss': best_val_loss,
                                    'state_dict': state_dict,     # 上面定义的字典
                                    'optimizer_dict': optimizer_dict,  # 上面定义的字典
                                    'args':args
                                    }
                model_server_0.save_checkpoint(model_dictionary, is_best)  # 需要更改

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


# 这部分也需要更改，感觉比较合适的方法是先全部经过encoder然后再计算，不不不，这样不行！因为经过encoder和输出是经过decoder之前
# 抛去这俩，实际上数据是不一致的！必须包含进去 encoder和decoder才行
# 所以还是得在各个函数里面写forward的部分

# Calculate mean and covariance for each channel's prediction errors, and save them with the trained model
print('=> calculating mean and covariance')
means, covs = list(),list()  # pretrained的话，就重新计算mean和conv ！很不错！
train_dataset = TimeseriesData.batchify(args, TimeseriesData.trainData, bsz=1)

# train_dataset : [nbatch,bsz,dim]


# config是一个字典，传输需要的信息，比如那个client有哪些维度等
config={}
config['dimensions_client_0'] = dimensions_client_0
config['dimensions_client_1'] = dimensions_client_1
config['dimensions_client_2'] = dimensions_client_2

config['recover_dimensions_client_0'] = recover_dimensions_client_0
config['recover_dimensions_client_1'] = recover_dimensions_client_1
config['recover_dimensions_client_2'] = recover_dimensions_client_2


state_dict = {'model_server_0':model_server_0.state_dict(),  # 必须这里重新创建一下，不然你要是pretrained模型的话，后面存储的dict就是{}!!白训练了
                              'encoder_client_0':encoder_client_0.state_dict(),
                              'encoder_client_1':encoder_client_1.state_dict(),
                              'encoder_client_2':encoder_client_2.state_dict(),
                              'decoder_client_0':decoder_client_0.state_dict(),
                              'decoder_client_1':decoder_client_1.state_dict(),
                              'decoder_client_2':decoder_client_2.state_dict(),}

optimizer_dict = {'model_server_0':optimizer_model_server_0.state_dict(),
                                  'encoder_client_0':optimizer_encoder_client_0.state_dict(),
                                  'encoder_client_1':optimizer_encoder_client_1.state_dict(),
                                  'encoder_client_2':optimizer_encoder_client_2.state_dict(),
                                  'decoder_client_0':optimizer_decoder_client_0.state_dict(),
                                  'decoder_client_1':optimizer_decoder_client_1.state_dict(),
                                  'decoder_client_2':optimizer_decoder_client_2.state_dict(),}


for channel_idx in range(feature_data):  # 默认 10
    mean, cov = fit_norm_distribution_param(args,model_server_0,encoder_client_0,encoder_client_1,encoder_client_2,
                                                                decoder_client_0,decoder_client_1,decoder_client_2,
                                                                train_dataset[:TimeseriesData.length],channel_idx,config)
    means.append(mean), covs.append(cov)
    print("已完成第",channel_idx,"个维度的计算")
model_dictionary = {'epoch': max(epoch,start_epoch),
                    'best_loss': best_val_loss,
                    'state_dict': state_dict,
                    'optimizer': optimizer_dict,
                    'args': args,
                    'means': means,
                    'covs': covs
                    }
model_server_0.save_checkpoint(model_dictionary, True)
print('-' * 89)