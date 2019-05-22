import torch
import torch.nn as nn

import math

from config import get_args


# reference
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# Thank you :)
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, sentence_len=128):
        super(PositionalEncoding, self).__init__()
        self.positionalEncoding = torch.zeros((sentence_len, embedding_dim))

        for pos in range(0, sentence_len):
            for i in range(0, embedding_dim // 2):
                self.positionalEncoding[pos, 2 * i] = math.sin(pos / math.pow(10000, 2 * i / embedding_dim))
                self.positionalEncoding[pos, 2 * i + 1] = math.cos(pos / math.pow(10000, 2 * i / embedding_dim))

        self.register_buffer('positional_encoding', self.positionalEncoding)

    def forward(self, x):
        sentence_len = x.size(1)
        out = x + self.positionalEncoding[:sentence_len, :].to(x)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = dk
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask):
        out = torch.matmul(query, key.transpose(2, 3))
        out /= math.sqrt(self.dk)

        if mask is not None:
            mask = mask.unsqueeze(1)
            out = out.masked_fill(mask == 0, -1e10)
        out = self.softmax(out)

        out = torch.matmul(out, value)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head = head
        self.dk = embedding_dim // head

        self.dense_query = nn.Linear(embedding_dim, embedding_dim)
        self.dense_key = nn.Linear(embedding_dim, embedding_dim)
        self.dense_value = nn.Linear(embedding_dim, embedding_dim)

        self.scaled_dot_product_attention = ScaledDotProductAttention(dk=self.dk)

        self.dense = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        batch, _, embedding_dim = query.size()

        query_out = self.dense_query(query).view(batch, -1, self.head, self.dk).transpose(1, 2)
        key_out = self.dense_key(key).view(batch, -1, self.head, self.dk).transpose(1, 2)
        value_out = self.dense_value(value).view(batch, -1, self.head, self.dk).transpose(1, 2)

        out = self.scaled_dot_product_attention(query_out, key_out, value_out, mask).view(batch, -1, self.embedding_dim)

        out = self.dense(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.embedding_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=True)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        norm = self.gamma * (x - mean) / (std + self.eps) + self.beta

        return norm


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super(FeedForward, self).__init__()
        self.embedding_dim = embedding_dim
        self.inner_dim = 2048

        # in the paper, they suggest two kind of methods.
        self.dense = nn.Sequential(
            nn.Linear(self.embedding_dim, self.inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.inner_dim, self.embedding_dim)
        )

        # If you use Conv layer, you have to adjust the dimension.
        # self.conv = nn.Sequential(
        #     nn.Conv2d(self.embedding_dim, self.inner_dim, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Conv2d(self.inner_dim, self.embedding_dim, kernel_size=1)
        # )

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

        self.layer_norm1 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout_rate)
        self.layer_norm2 = LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, x_mask):
        # multi head attention
        multi_head_out = self.dropout1(self.multi_head_attention(x, x, x, x_mask))
        out = self.layer_norm1(multi_head_out + x)

        # feed forward layer
        feed_forward_out = self.dropout2(self.feed_forward(out))
        out = self.layer_norm2(feed_forward_out + out)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, max_len, heads, embedding_dim, dropout_rate, N):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.N = N
        encoderLayers = []

        # Embedding shape, (max_num of inputs(voca_size), embedding_dim)
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # PositionalEncoding
        self.positionalEncoding = PositionalEncoding(embedding_dim)

        for _ in range(N):
            encoderLayers.append(EncoderLayer(heads, embedding_dim, dropout_rate))
        self.encoder = nn.Sequential(*encoderLayers)

    def forward(self, x, x_mask):
        embedding_out = self.embedding(x)
        out = embedding_out + self.positionalEncoding(embedding_out)

        # N time iteration
        for _ in range(self.N):
            out = self.encoder[_](out, x_mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, heads, embedding_dim, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(heads, embedding_dim, dropout_rate)
        self.multi_head_attention2 = MultiHeadAttention(heads, embedding_dim, dropout_rate)

        self.layer_norm1 = LayerNorm(embedding_dim)
        self.layer_norm2 = LayerNorm(embedding_dim)
        self.layer_norm3 = LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.feed_forward = FeedForward(embedding_dim, dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, encoder_out, target, x_mask, target_mask):
        # multi head attention
        multi_head_out1 = self.dropout1(self.multi_head_attention1(target, target, target, target_mask))
        out = self.layer_norm1(multi_head_out1 + target)

        # multi head attention
        multi_head_out2 = self.dropout2(self.multi_head_attention2(out, encoder_out, encoder_out, x_mask))
        out = self.layer_norm2(multi_head_out2 + out)

        # feed forward layer
        feed_forward_out = self.dropout3(self.feed_forward(out))
        out = self.layer_norm3(feed_forward_out + out)

        return out


class Decoder(nn.Module):
    def __init__(self, input_size, max_len, head, embedding_dim, dropout_rate, N):
        super(Decoder, self).__init__()
        self.N = N
        decoderLayers = []

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.positionalEncoding = PositionalEncoding(embedding_dim)

        for _ in range(N):
            decoderLayers.append(DecoderLayer(head, embedding_dim, dropout_rate))

        self.decoder = nn.Sequential(*decoderLayers)

    def forward(self, encoder_out, x, x_mask, target_mask):
        embedding_out = self.embedding(x)
        out = embedding_out + self.positionalEncoding(embedding_out)

        for _ in range(self.N):
            out = self.decoder[_](encoder_out, out, x_mask, target_mask)
        return out


class Transformer(nn.Module):
    def __init__(self, input_vocab, target_vocab, max_len, heads, embedding_dim, dropout_rate, N):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_vocab, max_len, heads, embedding_dim, dropout_rate, N)
        self.decoder = Decoder(target_vocab, max_len, heads, embedding_dim, dropout_rate, N)

        self.out = nn.Sequential(
            nn.Linear(embedding_dim, target_vocab),
        )

    def forward(self, x, x_mask, target, target_mask):
        encoder_out = self.encoder(x, x_mask)
        decoder_out = self.decoder(encoder_out, target, x_mask, target_mask)

        out = self.out(decoder_out)
        return out


# Test Code
def masking(source, target, args):
    source_mask = (source != 1).unsqueeze(-2)

    target_mask = (target != 1).unsqueeze(-2)
    sentence_len = target.size(1)

    numpy_mask = torch.tril(torch.ones((1, sentence_len, sentence_len)))
    numpy_mask = (numpy_mask == 0).to('cpu')

    target_mask = target_mask & numpy_mask

    return source_mask, target_mask


args = get_args()
source = torch.randint(high=100, size=(4, args.max_len))
target = torch.randint(high=50, size=(4, args.max_len))
target_input = target[:, :-1]

source_mask, target_mask = masking(source, target_input, args)

transformer = Transformer(100, 50, args.max_len, heads=4, embedding_dim=512, dropout_rate=0.1, N=6)
print(transformer(source, source_mask, target_input, target_mask).shape)
