import argparse


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument("--max-len", type=int, default=64, help="sentence max length, (default: 64)")
    parser.add_argument('--epochs', type=int, default=100, help="(default: 100)")
    parser.add_argument('--embedding-dim', type=int, default=512, help="(default: 512)")
    parser.add_argument('--n', type=int, default=6, help="(default: 6)")
    parser.add_argument('--heads', type=int, default=8, help="(default: 8)")
    parser.add_argument('--dropout-rate', type=int, default=0.1, help="(default: 0.1)")
    parser.add_argument('--batch-size', type=int, default=64, help="(default: 64)")
    parser.add_argument('--print-interval', type=int, default=400, help="(default: 400)")
    parser.add_argument('--learning-rate', type=int, default=0.0001, help="(default: 0.0001)")
    parser.add_argument('--pretrained', type=bool, default=False, help="(default: False)")
    parser.add_argument('--translate', type=bool, default=False, help="(default: False)")

    args = parser.parse_args()

    return args
