import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import math

from config import get_args


# reference
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# Thank you :)
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.positionalEncoding = torch.zeros((max_len, embedding_dim))

        for pos in range(0, max_len):
            for i in range(0, embedding_dim // 2):
                self.positionalEncoding[pos, 2 * i] = math.sin(pos / math.pow(10000, 2 * i / embedding_dim))
                self.positionalEncoding[pos, 2 * i + 1] = math.cos(pos / math.pow(10000, 2 * i / embedding_dim))

        self.register_buffer('positional_encoding', self.positionalEncoding)

    def forward(self, x):
        out = x + self.positionalEncoding
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = dk
        self.dropout_rate = dropout_rate

        self.softmax = nn.Softmax(dim=-1)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        out = torch.matmul(query, key.transpose(2, 3))
        out = out * math.sqrt(self.dk)
        if mask is not None:
            mask = mask.unsqueeze(1)
            out = out.masked_fill(mask == 0, -1e9)
        out = self.softmax(out)
        if self.dropout_rate > 0.0:
            out = self.dropout(out)
        out = torch.matmul(out, value)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head = head

        self.dense_query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dense_key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dense_value = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.embedding_dim // self.head, dropout)

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, query, key, value, mask=None):
        batch, max_len, embedding_dim = query.size()

        query_out = self.dense_query(query).view(batch, max_len, self.head, embedding_dim // self.head)
        key_out = self.dense_key(key).view(batch, max_len, self.head, embedding_dim // self.head)
        value_out = self.dense_value(value).view(batch, max_len, self.head, embedding_dim // self.head)

        query_out = query_out.transpose(1, 2)
        key_out = key_out.transpose(1, 2)
        value_out = value_out.transpose(1, 2)

        out = self.scaled_dot_product_attention(query_out, key_out, value_out, mask).view(batch, max_len, self.embedding_dim)

        out = self.dense(out)
        return out


class Norm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(Norm, self).__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.embedding_dim))
        self.beta = nn.Parameter(torch.zeros(self.embedding_dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        norm = self.gamma * (x - mean) / (std + self.eps) + self.beta

        return norm


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super(FeedForward, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.inner_dim = 2048

        # in the paper, they suggest two kind of methods.
        self.dense = nn.Sequential(
            nn.Linear(self.embedding_dim, self.inner_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.inner_dim, self.embedding_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.embedding_dim, self.inner_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.inner_dim, self.embedding_dim, kernel_size=1)
        )

    def forward(self, x):
        out = self.dense(x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, head, embedding_dim, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.head = head
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(head, embedding_dim, dropout_rate)

        self.norm1 = Norm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout_rate)
        self.norm2 = Norm(embedding_dim)

    def forward(self, x, x_mask):
        out = self.multi_head_attention(x, x, x, x_mask)
        out = self.norm1(out + x)

        # feed forward layer
        feed_forward_out = self.feed_forward(out)
        out = self.norm2(feed_forward_out + out)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, max_len, head, embedding_dim, dropout_rate, N):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.N = N
        encoderLayers = []

        # Embedding parameters, (max_num of inputs(voca_size), embedding_dim)
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.positionalEncoding = PositionalEncoding(max_len, embedding_dim)

        for _ in range(N):
            encoderLayers.append(EncoderLayer(head, embedding_dim, dropout_rate))

        self.encoder = nn.Sequential(*encoderLayers)

    def forward(self, x, x_mask):
        out = self.embedding(x)
        out = out + self.positionalEncoding(out)

        for _ in range(self.N):
            out = self.encoder[_](out, x_mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, head, embedding_dim, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(head, embedding_dim, dropout_rate)
        self.multi_head_attention2 = MultiHeadAttention(head, embedding_dim, dropout_rate)

        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)

        self.feed_forward = FeedForward(embedding_dim, dropout_rate)

    def forward(self, encoder_out, x, x_mask, target_mask):
        out = self.multi_head_attention1(x, x, x, target_mask)
        out = self.norm1(out + x)

        multi2_out = out
        out = self.multi_head_attention2(encoder_out, encoder_out, out, x_mask)
        out = self.norm2(out + multi2_out)

        feed_forward_out = out
        out = self.feed_forward(out)
        out = self.norm3(feed_forward_out + out)

        return out


class Decoder(nn.Module):
    def __init__(self, input_size, max_len, head, embedding_dim, dropout_rate, N):
        super(Decoder, self).__init__()
        self.N = N
        decoderLayers = []

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.positionalEncoding = PositionalEncoding(max_len, embedding_dim)

        for _ in range(N):
            decoderLayers.append(DecoderLayer(head, embedding_dim, dropout_rate))

        self.decoder = nn.Sequential(*decoderLayers)

    def forward(self, encoder_out, x, x_mask, target_mask):
        out = self.embedding(x)
        out = out + self.positionalEncoding(out)

        for _ in range(self.N):
            out = self.decoder[_](encoder_out, out, x_mask, target_mask)
        return out


class Transformer(nn.Module):
    def __init__(self, input_size, target_size, max_len, head, embedding_dim, dropout_rate, N, num_classes):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_size, max_len, head, embedding_dim, dropout_rate, N)
        self.decoder = Decoder(target_size, max_len, head, embedding_dim, dropout_rate, N)

        self.out = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, x_mask, target, target_mask):
        encoder_out = self.encoder(x, x_mask)
        decoder_out = self.decoder(encoder_out, target, x_mask, target_mask)

        out = self.out(decoder_out)
        return out


def main(args):
    temp = torch.LongTensor([
        [2, 2, 4, 1],
        [2, 3, 4, 22]
    ])
    print(temp.numpy())
    mask = np.triu(np.ones((2, 4, 4)), k=1).astype('uint8')
    mask = torch.from_numpy(mask) == 0
    src_mask = (temp != 1).unsqueeze(-2)
    print(src_mask.shape)
    # model_en = Encoder(torch.max(temp) + 1, args.max_len, head=8, embedding_dim=512, dropout_rate=0.1, N=6)
    # model_de = Decoder(torch.max(temp) + 1, args.max_len, head=8, embedding_dim=512, dropout_rate=0.1)
    #
    # encoder = model_en(temp, src_mask)
    # print(encoder.shape)
    # decoder = model_de(encoder, temp, src_mask, mask)
    # print(decoder.shape)
    transformer = Transformer(torch.max(temp) + 1, torch.max(temp) + 1, args.max_len, head=8, embedding_dim=512, dropout_rate=0.1, N=6, num_classes=1000)
    print(transformer(temp, src_mask, temp, mask).shape)


if __name__ == "__main__":
    args = get_args()
    main(args)
