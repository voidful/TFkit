import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
import tfkit


class TestModelLoader(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/')

    def test_list_all_model(self):
        models = tfkit.list_all_model()
        self.assertTrue(isinstance(models, list))

    def test_load_model_class(self):
        tfkit.load_model_class('clas')
        tfkit.load_model_class('once')

    def test_load_predict_parameter(self):
        model_class = tfkit.load_model_class('clas')
        # load pre-train model
        tokenizer = tfkit.BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = tfkit.AutoModel.from_pretrained('voidful/albert_chinese_tiny')
        model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail={"taskA": ["a", "b"]},
                                  maxlen=128)
        clas_param = tfkit.load_predict_parameter(model)
        self.assertTrue('input' in clas_param)
        self.assertTrue('topk' in clas_param)
        self.assertTrue('task' in clas_param)
        self.assertTrue('handle_exceed' in clas_param)
        self.assertTrue(isinstance(clas_param['handle_exceed'], str))

    def test_load_trained_model(self):
        model_path = os.path.join(self.MODEL_SAVE_PATH, '1.pt')
        model, model_type, model_class = tfkit.load_trained_model(model_path)
        print(model.predict("a"))
        print(model_type)
        print(model_class)
