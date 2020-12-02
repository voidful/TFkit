import unittest
import os

import tfkit


class TestTrain(unittest.TestCase):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'tfkit/test/cache/')
    CLAS_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'clas/')
    TAG_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'tag/')
    ONEBYONE_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'onebyone/')
    ONCE_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'once/')
    MASK_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'mask/')
    MCQ_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'mcq/')
    QA_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'qa/')
    MTTASK_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'mttask/')
    DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')
    CLAS_DATASET = os.path.join(DATASET_DIR, 'classification.csv')
    TAG_DATASET = os.path.join(DATASET_DIR, 'tag_row.csv')
    GEN_DATASET = os.path.join(DATASET_DIR, 'generate.csv')
    MASK_DATASET = os.path.join(DATASET_DIR, 'mask.csv')
    MCQ_DATASET = os.path.join(DATASET_DIR, 'mcq.csv')
    QA_DATASET = os.path.join(DATASET_DIR, 'qa.csv')

    def testHelp(self):
        result = os.system('tfkit-train -h')
        assert (result == 0)

    def test_parser(self):
        input_arg, model_arg = tfkit.train.parse_train_args(
            ['--model', 'onebyone', '--train', 'train.csv', '--test', 'test.csv', '--config',
             'voidful/albert_chinese_tiny'])
        print(input_arg, model_arg)
        self.assertTrue(input_arg.get('model') == ['onebyone'])
        self.assertTrue(isinstance(input_arg.get('train'), list))

    def test_optimizer(self):
        model_class = tfkit.utility.load_model_class('clas')
        tokenizer = tfkit.BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = tfkit.AutoModel.from_pretrained('voidful/albert_chinese_tiny')
        model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail={"taskA": ["a", "b"]},
                                  maxlen=128)
        optim = tfkit.train.optimizer(model, lr=0.1)
        print(optim)
        print(optim.zero_grad())

    def testMultiTask(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.MTTASK_MODEL_PATH, '--train', self.CLAS_DATASET,
             self.GEN_DATASET, '--lr', '5e-5', '--test', self.CLAS_DATASET, self.GEN_DATASET, '--model', 'clas',
             'onebyone', '--config', 'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.MTTASK_MODEL_PATH + ' --train ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --lr 5e-5 --test ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --model clas onebyone --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.MTTASK_MODEL_PATH + '  --train ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --lr 5e-5 --test ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --model clas onebyone --likelihood pos --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.MTTASK_MODEL_PATH + '  --train ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --lr 5e-5 --test ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --model clas onebyone --likelihood neg --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.MTTASK_MODEL_PATH + '  --train ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --lr 5e-5 --test ' + self.CLAS_DATASET + ' ' + self.GEN_DATASET + ' --model clas onebyone --likelihood both --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOneByOne(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.ONEBYONE_MODEL_PATH, '--train',
             self.GEN_DATASET, '--lr', '5e-5', '--test', self.GEN_DATASET, '--model', 'onebyone', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.ONEBYONE_MODEL_PATH + ' --train ' + self.GEN_DATASET + ' --test ' + self.GEN_DATASET + ' --model onebyone --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOnce(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.ONCE_MODEL_PATH, '--train',
             self.GEN_DATASET, '--lr', '5e-5', '--test', self.GEN_DATASET, '--model', 'once', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.ONCE_MODEL_PATH + ' --train ' + self.GEN_DATASET + ' --test ' + self.GEN_DATASET + ' --model once --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenMask(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.MASK_MODEL_PATH, '--train',
             self.MASK_DATASET, '--lr', '5e-5', '--test', self.MASK_DATASET, '--model', 'mask', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.MASK_MODEL_PATH + ' --train ' + self.MASK_DATASET + ' --test ' + self.MASK_DATASET + ' --model mask --config voidful/albert_chinese_tiny --maxlen 512')
        self.assertTrue(result == 0)

    def testMCQ(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.MCQ_MODEL_PATH, '--train',
             self.MCQ_DATASET, '--lr', '5e-5', '--test', self.MCQ_DATASET, '--model', 'mcq', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'start_slice'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.MCQ_MODEL_PATH + ' --train ' + self.MCQ_DATASET + ' --test ' + self.MCQ_DATASET + ' --model mcq --config voidful/albert_chinese_tiny --maxlen 512 --handle_exceed start_slice')
        self.assertTrue(result == 0)

    def testQA(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.QA_MODEL_PATH, '--train',
             self.QA_DATASET, '--lr', '5e-5', '--test', self.QA_DATASET, '--model', 'qa', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'start_slice'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.QA_MODEL_PATH + ' --train ' + self.QA_DATASET + ' --test ' + self.QA_DATASET + ' --model qa --config voidful/albert_chinese_tiny --maxlen 512 --handle_exceed start_slice')
        self.assertTrue(result == 0)

    def testGenWithSentLoss(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.MODEL_SAVE_PATH, '--train',
             self.GEN_DATASET, '--lr', '5e-5', '--test', self.GEN_DATASET, '--model', 'onebyone', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.MODEL_SAVE_PATH + ' --train ' + self.GEN_DATASET + ' --test ' + self.GEN_DATASET + ' --model onebyone --config voidful/albert_chinese_tiny  --maxlen 50')
        self.assertTrue(result == 0)

    def testClassify(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.CLAS_MODEL_PATH, '--train',
             self.CLAS_DATASET, '--lr', '5e-5', '--test', self.CLAS_DATASET, '--model', 'clas', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.CLAS_MODEL_PATH + ' --train ' + self.CLAS_DATASET + ' --test ' + self.CLAS_DATASET + ' --model clas --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testTag(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.TAG_MODEL_PATH, '--train',
             self.TAG_DATASET, '--lr', '5e-5', '--test', self.TAG_DATASET, '--model', 'tag', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'slide'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + self.TAG_MODEL_PATH + ' --train ' + self.TAG_DATASET + ' --test ' + self.TAG_DATASET + ' --model tag --config voidful/albert_chinese_tiny --maxlen 50 --handle_exceed slide')
        self.assertTrue(result == 0)

    def testAddToken(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', self.MODEL_SAVE_PATH, '--train',
             self.GEN_DATASET, '--lr', '5e-5', '--test', self.GEN_DATASET, '--model', 'onebyone', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--add_tokens', '5'])
        result = os.system(
            'tfkit-train --batch 2 --add_tokens 5  --savedir ' + self.MODEL_SAVE_PATH + ' --epoch 2  --train ' + self.GEN_DATASET + ' --test ' + self.GEN_DATASET + ' --model onebyone --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)
