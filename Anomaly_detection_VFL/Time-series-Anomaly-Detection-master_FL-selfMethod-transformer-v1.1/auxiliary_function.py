import torch
import torch.nn as nn
import numpy as np

def forward_FL(model_list,config,inputSeq,hidden):
    model_server_0 = model_list[0]
    encoder_client_0 = model_list[1]
    encoder_client_1 = model_list[2]
    encoder_client_2 = model_list[3]
    decoder_client_0 = model_list[4]
    decoder_client_1 = model_list[5]
    decoder_client_2 = model_list[6]

    dimensions_client_0 = config['dimensions_client_0']
    dimensions_client_1 = config['dimensions_client_1']
    dimensions_client_2 = config['dimensions_client_2']
    recover_dimensions_client_0 = config['recover_dimensions_client_0']
    recover_dimensions_client_1 = config['recover_dimensions_client_1']
    recover_dimensions_client_2 = config['recover_dimensions_client_2']

    seq_len = config['seq_len']
    batch_size = config['batch_size']

    # 开始前向传播

    inputSeq_client_0 = inputSeq[:, :, dimensions_client_0]  # [seq_len,batchsize,feature_0] [1,1,4]
    inputSeq_client_1 = inputSeq[:, :, dimensions_client_1]  # [seq_len,batchsize,feature_1] [1,1,3]
    inputSeq_client_2 = inputSeq[:, :, dimensions_client_2]  # [seq_len,batchsize,feature_2] [1,1,3]

    inputSeq_client_0 = inputSeq_client_0.reshape(seq_len * batch_size, -1)
    inputSeq_client_1 = inputSeq_client_1.reshape(seq_len * batch_size, -1)
    inputSeq_client_2 = inputSeq_client_2.reshape(seq_len * batch_size, -1)

    in_0 = encoder_client_0.forward(inputSeq_client_0)  # [seq_len*batchsize,8]
    in_1 = encoder_client_1.forward(inputSeq_client_1)  # [seq_len*batchsize,8]
    in_2 = encoder_client_2.forward(inputSeq_client_2)  # [seq_len*batchsize,8]

    in_0 = in_0.reshape(seq_len, batch_size, -1)
    in_1 = in_1.reshape(seq_len, batch_size, -1)
    in_2 = in_2.reshape(seq_len, batch_size, -1)

    in_cat = torch.cat((in_0, in_1, in_2), 2)  # [1,1,feature_dim]

    out_temp, hidden = model_server_0.forward(in_cat, hidden)

    out_0 = out_temp[:, :, recover_dimensions_client_0].reshape(seq_len * batch_size, -1)
    out_1 = out_temp[:, :, recover_dimensions_client_1].reshape(seq_len * batch_size, -1)
    out_2 = out_temp[:, :, recover_dimensions_client_2].reshape(seq_len * batch_size, -1)

    recover_0 = decoder_client_0.forward(out_0)  # [seq_len*batchsize,4]
    recover_1 = decoder_client_1.forward(out_1)  # [seq_len*batchsize,3]
    recover_2 = decoder_client_2.forward(out_2)  # [seq_len*batchsize,3]

    recover_0 = recover_0.reshape(seq_len, batch_size, -1)
    recover_1 = recover_1.reshape(seq_len, batch_size, -1)
    recover_2 = recover_2.reshape(seq_len, batch_size, -1)

    out = torch.cat((recover_0, recover_1, recover_2), 2)  # [seq_len,barch_size,10] 到这里真正恢复了

    return out,hidden
