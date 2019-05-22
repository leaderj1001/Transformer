import argparse


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument("--max-len", type=int, default=64, help="sentence max length, (default: 64)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout-rate', type=int, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--print-interval', type=int, default=400)
    parser.add_argument('--learning-rate', type=int, default=0.0001)
    parser.add_argument('--pretrain-model', type=bool, default=True)
    parser.add_argument('--translate', type=bool, default=False)

    args = parser.parse_args()

    return args
