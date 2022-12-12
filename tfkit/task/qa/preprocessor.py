import nlp2
import tfkit.utility.tok as tok
import torch
from tfkit.utility.datafile import get_qa_data_from_file
from tfkit.utility.preprocess import GeneralNLPPreprocessor


class Preprocessor(GeneralNLPPreprocessor):
    def read_file_to_data(self, path):
        return get_qa_data_from_file(path)

    def preprocess_component_prepare_input(self, item):
        mapping_index = []
        pos = 1  # cls as start 0
        input_text_list = nlp2.split_sentence_to_array(item['input'])
        for i in input_text_list:
            for _ in range(len(self.tokenizer.tokenize(i))):
                if _ < 1:
                    mapping_index.append({'char': i, 'pos': pos})
                pos += 1
        item['mapping_index'] = mapping_index
        return item

    def preprocess_component_convert_to_id(self, item, **param_dict):
        input_text, target = item['input'], item.get('target', None)
        tokenized_input = [tok.tok_begin(self.tokenizer)] + input_text + [tok.tok_sep(self.tokenizer)]
        input_id = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        start_index = item['input_index'][0]
        end_index = item['input_index'][1]
        if target:
            item['target'] = [0, 0]
            target_start, target_end = target
            ori_start = target_start = int(target_start)
            ori_end = target_end = int(target_end)
            ori_ans = tokenized_input[ori_start:ori_end]
            target_start -= start_index
            target_end -= start_index
            # print("target_start", self.parameters['maxlen'],item['mapping_index'][target_start]['pos'],ori_end)
            # if item['mapping_index'][target_start]['pos'] > ori_end or target_start < 0 \
            #         or target_start > self.parameters['maxlen'] \
            #         or target_end >= self.parameters['maxlen'] - 2:
            #     target_start = 0
            #     target_end = 0
            # else:
            for map_pos, map_tok in enumerate(item['mapping_index'][start_index:]):
                if start_index < map_tok['pos'] <= end_index:
                    length = len(self.tokenizer.tokenize(map_tok['char']))
                    if map_pos < ori_start:
                        target_start += length - 1
                    if map_pos < ori_end:
                        target_end += length - 1
            item['target'] = [target_start + 1, target_end + 1]  # cls +1

        item['input'] = input_id
        item['mask'] = [1] * len(input_id)
        item['raw_input'] = tokenized_input
        yield item

    def postprocess(self, item, tokenizer, maxlen, **kwargs):
        row_dict = {
            'input': item['input'],
            'mask': item['mask']
        }
        if 'target' in item:
            row_dict['target'] = item['target']
        return {key: torch.tensor(value) for key, value in row_dict.items()}
