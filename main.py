import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os

from config import get_args
from preprocess import load_data_loader
from transformer import Transformer

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def masking(source, target, args):
    source_mask = (source != 1).unsqueeze(-2)

    target_mask = (target != 1).unsqueeze(-2)
    sentence_len = target.size(1)

    numpy_mask = torch.tril(torch.ones((1, sentence_len, sentence_len)))
    numpy_mask = (numpy_mask == 0)

    target_mask = target_mask & numpy_mask.to(device)

    return source_mask, target_mask


def save_model(model, epoch, args):
    print('Save model ...')
    state = {
        'model': model.state_dict(),
        'epoch': epoch
    }
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    filename = 'best_model_' + str(args.max_len) + '_' + str(args.heads) + '_' + str(args.embedding_dim) + '_' + str(args.n) + '_ckpt.t7'
    torch.save(state, './checkpoints/' + filename)


def train(train_loader, model, optimizer, criterion, args):
    model.train()

    train_loss = 0.0
    for i, batch_data in enumerate(train_loader):
        source, target = batch_data.source.transpose(0, 1).to(device), batch_data.target.transpose(0, 1).to(device)
        source_mask, target_mask = masking(source, target[:, :-1], args)

        output = model(source, source_mask, target[:, :-1], target_mask)
        ys = target[:, 1:].contiguous().view(-1)

        optimizer.zero_grad()
        loss = F.cross_entropy(output.view(-1, output.size(-1)), ys, ignore_index=1)
        train_loss += loss.data
        loss.backward()
        optimizer.step()
        if i % args.print_interval == 0:
            print("Training loss: ", loss.data)
    return train_loss / float(len(train_loader))


def translate(model, args):
    model.eval()

    with torch.no_grad():
        pass


def main(args):
    train_loader, input_vocab, target_vocab = load_data_loader(args)
    if args.pretrain_model:
        model = Transformer(input_vocab, target_vocab, args.max_len, args.heads, args.embedding_dim, args.dropout_rate, args.n).to(device)
        filename = 'best_model_' + str(args.max_len) + '_' + str(args.heads) + '_' + str(args.embedding_dim) + '_' + \
                   str(args.n) + '_ckpt.t7'
        checkpoint = torch.load('./checkpoints/' + filename)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']

        if args.translate:
            translate(model, args)
            return 0
    else:
        model = Transformer(input_vocab, target_vocab, args.max_len, args.heads, args.embedding_dim, args.dropout_rate, args.n).to(device)
        start_epoch = 1

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(train_loader, model, optimizer, criterion, args)
        print("[Epoch: {0:4d}] training loss: {1:2.3f}".format(epoch, train_loss))
        save_model(model, epoch, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
