import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from modules.helpers import sequence_mask, masked_normalization


class GaussianNoise(nn.Module):
    def __init__(self, stddev, mean=.0):
        """
        Additive Gaussian Noise layer
        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean

    def forward(self, x):
        if self.training:
            noise = Variable(x.data.new(x.size()).normal_(self.mean,
                                                          self.stddev))
            return x + noise
        return x

    def __repr__(self):
        return '{} (mean={}, stddev={})'.format(self.__class__.__name__,
                                                str(self.mean),
                                                str(self.stddev))


class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 embeddings=None,
                 noise=.0,
                 dropout=.0,
                 trainable=True):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            noise (float):
            dropout (float):
            trainable (bool):
        """
        super(Embed, self).__init__()

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      scale_grad_by_freq=False,
                                      sparse=False)

        # initialize the weights of the Embedding layer,
        # with the given pre-trained word vectors
        if embeddings is not None:
            print("Initializing Embedding layer with pre-trained weights!")
            self.init_embeddings(embeddings, trainable)

        # the dropout "layer" for the word embeddings
        self.dropout = nn.Dropout(dropout)

        # the gaussian noise "layer" for the word embeddings
        self.noise = GaussianNoise(noise)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def forward(self, x):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x (): the input data (the sentences)

        Returns: the logits for each class

        """
        embeddings = self.embedding(x)

        if self.noise.stddev > 0:
            embeddings = self.noise(embeddings)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=True,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequence, lengths):

        energies = self.attention(sequence).squeeze()

        # construct a mask, based on sentence lengths
        if len(energies.size()) < 2:
            mask = sequence_mask(lengths, 1)
        else:
            mask = sequence_mask(lengths, energies.size(1))
        scores = masked_normalization(energies, mask)
        contexts = (sequence * scores.unsqueeze(-1)).sum(1)

        return contexts, scores


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class CoAttention(nn.Module):
    def __init__(self, input_size):
        super(CoAttention, self).__init__()
        self.att_weight_c = Linear(input_size, 1)
        self.att_weight_q = Linear(input_size, 1)
        self.att_weight_cq = Linear(input_size, 1)

        # self.attention_and_query = nn.Linear(12 * hidden_size, hidden_size, bias=False)
        # self.prob_attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, sentence, post):
        """
        :param sentence: (batch, c_len, hidden_size * 2)
        :param post: (batch, q_len, hidden_size * 2)
        :return: (batch, c_len, q_len)
        """
        c_len = sentence.size(1)
        q_len = post.size(1)

        # (batch, c_len, q_len, hidden_size * 2)
        # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
        # (batch, c_len, q_len, hidden_size * 2)
        # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
        # (batch, c_len, q_len, hidden_size * 2)
        # cq_tiled = c_tiled * q_tiled
        # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = post.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(sentence * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        # (batch, c_len, q_len)
        s = self.att_weight_c(sentence).expand(-1, -1, q_len) + \
            self.att_weight_q(post).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, post)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, sentence).squeeze(1)
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, hidden_size * 8)
        x = torch.cat([sentence, c2q_att, sentence * c2q_att, sentence * q2c_att], dim=-1)
        return x