import tfkit.utility.tok as tok
from tfkit.utility.datafile import get_tag_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor

get_data_from_file = get_tag_data_from_file


class Preprocessor(GeneralNLPPreprocessor):

    def read_file_to_data(self, path):
        return get_tag_data_from_file(path)

    def preprocess(self, item, **param_dict):
        input_text, target = item['input'], item.get('target', None)
        separator = param_dict.get('separator', ' ')
        word_token_mapping = []
        token_word_mapping = []
        pos = 0

        for word_i, word in enumerate(input_text.split(separator)):
            tokenize_word = self.tokenizer.tokenize(word)
            for _ in range(len(tokenize_word)):
                if _ < 1:  # only record first token (one word one record)
                    word_token_mapping.append({'char': word, 'pos': pos, 'len': len(tokenize_word)})
                token_word_mapping.append({'tok': tokenize_word[_], 'word': word, 'pos': len(word_token_mapping) - 1})
                pos += 1

        t_input_list, t_pos_list = tok.handle_exceed(self.tokenizer, input_text, self.parameters['maxlen'] - 2,
                                                     mode=self.parameters.get('handle_exceed'),
                                                     keep_after_sep=False)
        preprocessed_data = []
        for t_input, t_pos in zip(t_input_list, t_pos_list):  # -1 for cls
            # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            row_dict = dict()
            tokenized_input = [tok.tok_begin(self.tokenizer)] + t_input
            input_id = self.tokenizer.convert_tokens_to_ids(tokenized_input)

            if target is not None:
                target_token = []
                for input_word, target_label in zip(word_token_mapping, target.split(separator)):
                    if t_pos[0] <= input_word['pos'] < t_pos[1]:
                        for _ in range(input_word['len']):
                            target_token += [target_label]

                target_id = [target_token[0]] + target_token

                if len(input_id) != len(target_id):
                    print(list(zip(input.split(separator), target.split(separator))))
                    print(self.tokenizer.decode(input_id))
                    print(input_id)
                    print(target_id)
                    print("input target len not equal ", len(input_id), len(target_id))
                    continue
                row_dict['target'] = target_id

            row_dict['input'] = input_id
            row_dict['word_token_mapping'] = word_token_mapping
            row_dict['token_word_mapping'] = token_word_mapping
            row_dict['end'] = len(input_id)
            row_dict['pos'] = t_pos
            preprocessed_data.append(row_dict)
        return preprocessed_data

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        labels = item['task_dict']
        print("item['input']",len(item['input']))
        mask_id = [1] * len(item['input'])
        mask_id.extend([0] * (maxlen - len(mask_id)))
        item['input'].extend([0] * (self.parameters['maxlen'] - len(item['input'])))
        row_dict = {
            'input': item['input'],
            'mask': mask_id,
            'pos': item['pos'],
        }
        # 'token_word_mapping': item['token_word_mapping']
        if 'target' in item:
            print(labels['tag'])
            target_id = [labels['tag'].index(i) for i in item['target']]
            if "O" in labels['tag']:
                target_id = [labels['tag'].index("O")] + target_id
            else:
                target_id = [target_id[0]] + target_id
            target_id.extend([0] * (self.parameters['maxlen'] - len(target_id)))
            row_dict['target'] = target_id

        return row_dict
