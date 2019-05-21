from torchtext.data import TabularDataset, BucketIterator
from torchtext.data import Field
import spacy
import re
import pandas as pd
import os


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()


class Tokenize(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


def create_fields(args):
    language = {
        'en': 'en_core_web_sm',
        'de': 'de_core_news_sm'
    }
    en_tokenizer = Tokenize(language['en'])
    de_tokenizer = Tokenize(language['de'])

    source = Field(
        tokenize=en_tokenizer.tokenizer,
        lower=True,
        fix_length=args.max_len
    )

    target = Field(
        tokenize=de_tokenizer.tokenizer,
        lower=True,
        init_token='<BOS>',
        eos_token='<EOS>',
        fix_length=args.max_len
    )

    return (source, target)


def load_raw_data_to_csv(source_filename, target_filename, mode='train'):
    raw_data = {
        'source': [line.strip().split('\n') for line in read_data(source_filename)],
        'target': [line.strip().split('\n') for line in read_data(target_filename)]
    }
    train_df = pd.DataFrame(raw_data, columns=["source", "target"])
    train_df.to_csv(mode + '.csv', index=False)


def load_data_loader(args, mode='train'):
    if mode == 'train':
        train_dir_s = os.path.join(args.data_path, 'train/train.en')
        train_dir_t = os.path.join(args.data_path, 'train/train.de')
        load_raw_data_to_csv(train_dir_s, train_dir_t, mode='train')
        path = './train.csv'
    elif mode == 'test':
        test_dir_s = os.path.join(args.data_path, 'test/test.en')
        test_dir_t = os.path.join(args.data_path, 'test/test.de')
        load_raw_data_to_csv(test_dir_s, test_dir_t, mode='test')
        path = './test.csv'

    source, target = create_fields(args)

    data = TabularDataset(
        path=path,
        format='csv',
        fields=[('source', source), ('target', target)]
    )

    data_loader = BucketIterator(
        data,
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.source),
        shuffle=True
    )

    source.build_vocab(data)
    target.build_vocab(data)

    if os.path.isfile('train.csv'):
        os.remove('train.csv')
    if os.path.isfile('test.csv'):
        os.remove('test.csv')

    # source_pad = source.vocab.stoi['<pad>']
    # target_bos = target.vocab.stoi['<BOS>']
    # target_eos = target.vocab.stoi['<EOS>']
    # target_pad = target.vocab.stoi['<pad>']

    return data_loader, len(source.vocab), len(target.vocab)


# if __name__ == "__main__":
#     args = get_args()
#     load_data_loader(args)
