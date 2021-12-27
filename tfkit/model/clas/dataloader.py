from sklearn.preprocessing import MultiLabelBinarizer
import tfkit.utility.tok as tok
from tfkit.utility.dataloader import get_multiclas_data_from_file

get_data_from_file= get_multiclas_data_from_file


def preprocessing_data(item, tokenizer, maxlen=512, handle_exceed='slide', **kwargs):
    tasks, task, input, target = item
    param_dict = {'input': input, 'tokenizer': tokenizer, 'target': target, 'maxlen': maxlen,
                  'handle_exceed': handle_exceed, 'tasks': tasks, 'task': task}
    yield get_feature_from_data, param_dict


def get_feature_from_data(tokenizer, maxlen, tasks, task, input, target=None, handle_exceed='slide',
                          **kwargs):
    feature_dict_list = []
    t_input_list, _ = tok.handle_exceed(tokenizer, input, maxlen - 2, handle_exceed)
    for t_input in t_input_list:  # -2 for cls and sep
        row_dict = dict()
        row_dict['task'] = task
        input_token = [tok.tok_begin(tokenizer)] + t_input + [tok.tok_sep(tokenizer)]
        tokenized_input_id = tokenizer.convert_tokens_to_ids(input_token)
        mask_id = [1] * len(tokenized_input_id)
        tokenized_input_id.extend([tokenizer.pad_token_id] * (maxlen - len(tokenized_input_id)))
        mask_id.extend([-1] * (maxlen - len(mask_id)))
        row_dict['input'] = tokenized_input_id
        row_dict['mask'] = mask_id
        row_dict['target'] = [-1]
        if target is not None:
            if 'multi_label' in task:
                mlb = MultiLabelBinarizer(classes=tasks[task])
                tar = mlb.fit_transform([target])
                tokenize_label = tar
            else:
                tokenize_label = [tasks[task].index(target[0])]
            row_dict['target'] = tokenize_label
        feature_dict_list.append(row_dict)
    return feature_dict_list
