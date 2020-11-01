import argparse
import sys

from tfkit import load_trained_model


def parse_dump_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dumpdir", required=True, type=str)
    return vars(parser.parse_args(args))


def main(arg=None):
    arg = parse_dump_args(sys.argv[1:]) if arg is None else parse_dump_args(arg)
    model, model_type, model_class = load_trained_model(arg.get('model'))
    model.pretrained.save_pretrained(arg.get('dumpdir'))
    print('==================')
    print("Finish model dump.")


if __name__ == "__main__":
    main()
