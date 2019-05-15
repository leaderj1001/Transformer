import torch
import torch.nn as nn

import math

from config import get_args


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
    def __init__(self, embedding_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.embedding_dim = embedding_dim

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        out = torch.matmul(query, key.transpose(2, 3))
        out = out * math.sqrt(self.embedding_dim)
        out = self.softmax(out)
        out = torch.matmul(out, value)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, mask=False):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head = head
        self.mask = mask

        self.dense_query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dense_key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dense_value = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.embedding_dim)

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, query, key, value):
        batch, max_len, embedding_dim = query.size()

        query_out = self.dense_query(query).view(batch, max_len, embedding_dim // self.head, self.head)
        key_out = self.dense_key(key).view(batch, max_len, embedding_dim // self.head, self.head)
        value_out = self.dense_value(value).view(batch, max_len, embedding_dim // self.head, self.head)

        out = self.scaled_dot_product_attention(query_out, key_out, value_out).view(batch, max_len, self.embedding_dim)

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
    def __init__(self, embedding_dim):
        super(FeedForward, self).__init__()
        self.embedding_dim = embedding_dim
        self.inner_dim = 2048

        # in the paper, they suggest two kind of methods.
        self.dense = nn.Sequential(
            nn.Linear(self.embedding_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.embedding_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.embedding_dim, self.inner_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.inner_dim, self.embedding_dim, kernel_size=1)
        )

    def forward(self, x):
        out = self.dense(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, max_len, head, embedding_dim):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.head = head
        self.embedding_dim = embedding_dim

        # Embedding parameters, (max_num of inputs(voca_size), embedding_dim)
        self.embedding = nn.Embedding(self.input_size, embedding_dim)
        self.positionalEncoding = PositionalEncoding(self.max_len, embedding_dim)
        self.multi_head_attention = MultiHeadAttention(head, embedding_dim)

        self.norm1 = Norm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)
        self.norm2 = Norm(embedding_dim)

    def forward(self, x):
        # x shape: (batch, max_len)

        # word embedding
        # out shape: (batch, max_len, embedding_dim)
        out = self.embedding(x)

        # positional encoding
        # out shape: (batch, max_len, embedding_dim)
        out = self.positionalEncoding(out)

        # multi head attention
        positional_embedding_out = out
        out = self.multi_head_attention(out, out, out)
        out = self.norm1(positional_embedding_out + out)

        # feed forward layer
        multi_head_out = out
        out = self.feed_forward(out)
        out = self.norm2(multi_head_out + out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, query, key, value):
        pass


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        pass

    def forward(self, x):
        pass


def main(args):
    temp = torch.LongTensor([
        [1, 2, 4, 5],
        [2, 3, 4, 22]
    ])
    model = Encoder(torch.max(temp) + 1, args.max_len, head=4, embedding_dim=512)

    print(model(temp).shape)


if __name__ == "__main__":
    args = get_args()
    main(args)
