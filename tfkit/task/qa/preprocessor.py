import nlp2
import tfkit.utility.tok as tok
from tfkit.utility.datafile import get_qa_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):
    def read_file_to_data(self, path):
        return get_qa_data_from_file(path)

    def preprocess(self, item, **param_dict):
        input_text, target = item['input'], item.get('target', None)
        preprocessed_data = []
        mapping_index = []
        pos = 1  # cls as start 0
        input_text_list = nlp2.split_sentence_to_array(item['input'])
        for i in input_text_list:
            for _ in range(len(self.tokenizer.tokenize(i))):
                if _ < 1:
                    mapping_index.append({'char': i, 'pos': pos})
                pos += 1
        print("self.parameters.get('handle_exceed')",self.parameters.get('handle_exceed'))
        t_input_list, t_pos_list = tok.handle_exceed(self.tokenizer, input_text, self.parameters['maxlen'] - 2,
                                                     mode=self.parameters.get('handle_exceed'))
        for t_input, t_pos in zip(t_input_list, t_pos_list):  # -2 for cls and sep:
            row_dict = {**self.parameters}
            row_dict['target'] = [0, 0]
            tokenized_input = [tok.tok_begin(self.tokenizer)] + t_input + [tok.tok_sep(self.tokenizer)]
            input_id = self.tokenizer.convert_tokens_to_ids(tokenized_input)
            if target:
                target_start,target_end = target
                ori_start = target_start = int(target_start)
                ori_end = target_end = int(target_end)
                ori_ans = input_text_list[ori_start:ori_end]
                target_start -= t_pos[0]
                target_end -= t_pos[0]
                if mapping_index[target_start]['pos'] > ori_end or target_start < 0 \
                        or target_start > self.parameters['maxlen'] \
                        or target_end >= self.parameters['maxlen'] - 2:
                    target_start = 0
                    target_end = 0
                else:
                    for map_pos, map_tok in enumerate(mapping_index[t_pos[0]:]):
                        if t_pos[0] < map_tok['pos'] <= t_pos[1]:
                            length = len(self.tokenizer.tokenize(map_tok['char']))
                            if map_pos < ori_start:
                                target_start += length - 1
                            if map_pos < ori_end:
                                target_end += length - 1
                if ori_ans != tokenized_input[target_start + 1:target_end + 1] \
                        and self.tokenizer.tokenize(" ".join(ori_ans)) != tokenized_input[
                                                                          target_start + 1:target_end + 1] \
                        and target_start != target_end != 0:
                    continue
                row_dict['target'] = [target_start + 1, target_end + 1]  # cls +1

            mask_id = [1] * len(input_id)
            mask_id.extend([0] * (self.parameters['maxlen'] - len(mask_id)))
            row_dict['mask'] = mask_id
            input_id.extend([0] * (self.parameters['maxlen'] - len(input_id)))
            row_dict['input'] = input_id
            row_dict['raw_input'] = tokenized_input
            preprocessed_data.append(row_dict)
        return preprocessed_data

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        row_dict = {
            'input': item['input'],
            'mask': item['mask'],
        }
        if 'target' in item:
            row_dict['target'] = item['target']
        return row_dict
