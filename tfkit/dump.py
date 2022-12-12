import argparse
import sys

from transformers import AutoModelWithLMHead, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from tfkit.utility.model import load_trained_model, add_tokens_to_pretrain


def parse_dump_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dumpdir", required=True, type=str)
    return vars(parser.parse_args(args))


def main(arg=None):
    arg = parse_dump_args(sys.argv[1:]) if arg is None else parse_dump_args(arg)
    model, model_type, model_class, model_info, model_preprocessor = load_trained_model(arg.get('model'))
    tokenizer = model.tokenizer
    pretrained_config = model_info.get("model_config")
    if model_type == 'clm' and "gpt" in pretrained_config:
        hf_model = AutoModelWithLMHead.from_pretrained(model_info.get("model_config"))
        hf_model.eval()
        hf_model.transformer = model.pretrained
        hf_model.lm_head.weight = model.model.weight
        hf_model.config.tie_word_embeddings = False
        hf_model, tokenizer = add_tokens_to_pretrain(hf_model, tokenizer, model_info.get('add_tokens', []))
        hf_model.save_pretrained(arg.get('dumpdir'))
    elif model_type == 'seq2seq' and "bart" in pretrained_config:
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_info.get("model_config"))
        hf_model.eval()
        hf_model.lm_head = model.model
        hf_model.model = model.pretrained
        hf_model.config.tie_word_embeddings = False
        hf_model.config.tie_encoder_decoder = False
        hf_model, tokenizer = add_tokens_to_pretrain(hf_model, tokenizer, model_info.get('add_tokens', []))
        hf_model.save_pretrained(arg.get('dumpdir'))
    elif model_type == 'clas':
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_info.get("model_config"))
        hf_model.classifier.weight = model.classifier_list[0].weight
        hf_model.save_pretrained(arg.get('dumpdir'))
    else:
        model.pretrained.save_pretrained(arg.get('dumpdir'))

    tokenizer.save_pretrained(arg.get('dumpdir'))
    print('==================')
    print("Finish model dump.")


if __name__ == "__main__":
    main()
