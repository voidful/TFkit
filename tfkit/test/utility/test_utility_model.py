import os
import sys

from tfkit.utility.model import list_all_model, load_model_class, load_predict_parameter, load_trained_model

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
from transformers import BertTokenizer, AutoModel


class TestModelLoader(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../../"))
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/')

    def test_list_all_model(self):
        models = list_all_model()
        self.assertTrue(isinstance(models, list))

    def test_load_model_class(self):
        load_model_class('clas')
        load_model_class('once')

    def test_load_predict_parameter(self):
        model_class = load_model_class('clas')
        # load pre-train task
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')
        model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail={"taskA": ["a", "b"]},
                                  maxlen=128)
        clas_param = load_predict_parameter(model)
        print("clas_param", clas_param)
        self.assertTrue('input' in clas_param)
        self.assertTrue('topK' in clas_param)
        self.assertTrue('task' in clas_param)
        self.assertTrue('handle_exceed' in clas_param)
        self.assertTrue(isinstance(clas_param['handle_exceed'], str))

    # def test_load_trained_model(self):
    #     model_path = os.path.join(self.MODEL_SAVE_PATH, '1.pt')
    #     model, model_type, model_class, model_info, preprocessor = load_trained_model(model_path)
    #     print(model)
    #     print(model_type)
    #     print(model_class)
    #     print(model_info)
    #     print(model.predict)
    #     print(model.predict(input="a"))
