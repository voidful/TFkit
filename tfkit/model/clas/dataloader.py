from sklearn.preprocessing import MultiLabelBinarizer
import tfkit.utility.tok as tok
from tfkit.utility.datafile import get_multiclas_data_from_file
from tfkit.utility.datapreprocess import GeneralNLPPreprocessor

get_data_from_file = get_multiclas_data_from_file

preprocessor = GeneralNLPPreprocessor


def get_feature_from_data(item, tokenizer, maxlen, task_dict, **kwargs):
    tinput, task = item['input'], item['task']
    row_dict = {'task': list(task.encode("utf-8"))}
    input_token = [tok.tok_begin(tokenizer)] + tinput + [tok.tok_sep(tokenizer)]
    tokenized_input_id = tokenizer.convert_tokens_to_ids(input_token)
    mask_id = [1] * len(tokenized_input_id)
    tokenized_input_id.extend([tokenizer.pad_token_id] * (maxlen - len(tokenized_input_id)))
    mask_id.extend([-1] * (maxlen - len(mask_id)))
    row_dict['input'] = tokenized_input_id
    row_dict['mask'] = mask_id
    row_dict['target'] = [-1]
    if 'target' in item:
        target = item['target']
        if 'multi_label' in task:
            mlb = MultiLabelBinarizer(classes=task_dict[task])
            tar = mlb.fit_transform([item['target']])
            tokenize_label = tar
        else:
            tokenize_label = [task_dict[task].index(target[0])]
        row_dict['target'] = tokenize_label
    return row_dict
