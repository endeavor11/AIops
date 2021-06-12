import torch.nn as nn
import torch
from torch.autograd import Variable
import shutil
from pathlib import Path
import math

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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)    # [1, max_len, d_model] -> [max_len, 1, d_model]
                                                # 这样就迎合了 [seqLen, batch, feature]
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :] # self.pe[:x.size(0), :].shape  [seqLen, 1, feature_dim]


class RNNPredictor(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, enc_inp_size, rnn_inp_size, rnn_hid_size, dec_out_size, nlayers, dropout=0.5,
                 tie_weights=False,res_connection=False):
        
        """
        rnn_type        :LSTM,GRU 
        enc_inp_size    :feature_dim  # incoder的输入维度
        rnn_inp_size    :transformer 的出入数据的维度
        rnn_hid_size    :transformer 的内部维度
        dec_out_size    :feature_dim  # decoder的输出维度
        nlayers         :number of layers 默认2
        dropout         :
        tie_weights     :tie the word embedding and softmax weights (deprecated)
        res_connection  :residual connection
        """
        
        super(RNNPredictor, self).__init__()
        self.enc_input_size = enc_inp_size

        self.drop = nn.Dropout(dropout)

        self.rnn_type = rnn_type

        self.encoder = nn.Linear(enc_inp_size, rnn_inp_size)  # 出入特征维度到RNN输入维度
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_inp_size, rnn_hid_size, nlayers, dropout=dropout)
                                                            # RNN输入维度到RNN内部维度
        elif rnn_type == 'transformer':

            # encoder_layer = nn.TransformerEncoderLayer(d_model=rnn_inp_size, nhead=4, dim_feedforward=rnn_hid_size) # d_model 是输入数据的维度
            encoder_layer = nn.TransformerEncoderLayer(d_model=rnn_inp_size, nhead=8)
            self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=1)


        self.decoder = nn.Linear(rnn_inp_size, dec_out_size)

        """
        增加positionCoding
        """
        self.pos_encoder = PositionalEncoding(rnn_inp_size)
        self.src_mask = None
            
        self.res_connection=res_connection  # 默认是true和false
        #self.init_weights()
        self.rnn_type = rnn_type
        self.rnn_hid_size = rnn_hid_size
        self.nlayers = nlayers
        self.rnn_inp_size = rnn_inp_size
        #self.layerNorm1=nn.LayerNorm(normalized_shape=rnn_inp_size)
        #self.layerNorm2=nn.LayerNorm(normalized_shape=rnn_hid_size)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_hiddens=False, noise=False):

        if(self.rnn_type == 'transformer'):
            return self.forward_transformer(input, return_hiddens=return_hiddens, noise=noise)


    def forward_transformer(self, input, return_hiddens=False, noise=False):
        emb = self.drop(self.encoder(input.contiguous().view(-1, self.enc_input_size)))  # [(seq_len x batch_size) * feature_size]
        # encoder input size 就是 feature-dim
        # input原本 [seq,batch,inputsize]
        # reshape[seq*batch,inputsize]
        # emb size:[seq*batch,feature_dim]

        #emb = emb.view(-1, input.size(1), self.rnn_hid_size)  # [ seq_len * batch_size * feature_size]
        emb = emb.view(-1, input.size(1), self.rnn_inp_size)
        # 上面的*其实是,


        """
        增加mask
        """
        if self.src_mask is None or self.src_mask.size(0) != len(input):
            device = input.device
            mask = self._generate_square_subsequent_mask(len(input)).to(device)
            self.src_mask = mask

        emb = self.pos_encoder(emb)

        output = self.rnn(emb, self.src_mask)
        # output = self.layerNorm2(output)

        # output size [seq,batch,hiddensize]
        # hidden [2,batch,hiddensize]  因为是两层的模型！

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))  # [(seq_len x batch_size) * feature_size]
        # output reshape [seq*batch,hiddensize]
        # 经过 decoder
        # decoded size [seq*batch,dec_out_size]
        # 也就是 [seq*batch,feature_dim]
        decoded = decoded.view(output.size(0), output.size(1),
                               decoded.size(1))  # [ seq_len * batch_size * feature_size]
        if self.res_connection:  # 原来是这个意思
            decoded = decoded + input
        if return_hiddens:
            return decoded, 0, output  # 范围的三个值
            # decoder是最终的输出，用来模拟输入，预测下一个值
            # hidden 是最初的hidden 要注意hidden是两层的
            # output 是最初的输出也就是所有步骤的h

        return decoded, 0  # 不应该是output吗...可能只用decoded吧

    def _generate_square_subsequent_mask(self, sz):  # sz = SeqLen
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        """
        mask = 
        [[ True, False, False, False, False],
        [ True,  True, False, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True]]
        """
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        """
        mask = 
        [[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]])
        """
        return mask



    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_())

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return 0
        else:
            return 0

    def save_checkpoint(self,state, is_best):
        print("=> saving checkpoint ..")
        args = state['args']
        checkpoint_dir = Path('save',str(args.data)+str(args.index),'checkpoint')

        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.filename).with_suffix('.pth')

        torch.save(state, str(checkpoint))
        if is_best:
            model_best_dir = Path('save', str(args.data)+str(args.index), 'model_best')
            model_best_dir.mkdir(parents=True,exist_ok=True)

            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.filename).with_suffix('.pth'))

        print('=> checkpoint saved.')

    def extract_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer

    def initialize(self,args,feature_dim):
        self.__init__(rnn_type = args.model,
                           enc_inp_size=feature_dim,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=feature_dim,
                           nlayers = args.nlayers,
                           dropout = args.dropout,
                           tie_weights= args.tied,
                           res_connection=args.res_connection)
        self.to(args.device)

    def load_checkpoint(self, args, checkpoint, feature_dim): # 训练文件会用到这个文件
        start_epoch = checkpoint['epoch'] +1  # 返回的第二个结果
        best_val_loss = checkpoint['best_loss']  # 返回的第三个结果
        args_ = checkpoint['args']  # 返回的第一个结果
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size=args.prediction_window_size
        self.initialize(args_, feature_dim=feature_dim)
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss