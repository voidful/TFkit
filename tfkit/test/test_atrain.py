import unittest
import pytest
import tfkit
from tfkit.test import *
from transformers import BertTokenizer, AutoModel, AutoTokenizer

class TestTrain(unittest.TestCase):

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

        input_arg, model_arg = tfkit.train.parse_train_args(
            ['--model', 'onebyone', '--train', 'train.csv', '--test', 'test.csv', '--config',
             'voidful/albert_chinese_tiny', '--likelihood', 'pos'])
        print(input_arg, model_arg)
        self.assertTrue(model_arg.get('likelihood') == 'pos')
        self.assertTrue(isinstance(input_arg.get('train'), list))

    def test_optimizer(self):
        model_class = tfkit.utility.load_model_class('clas')
        tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_tiny')
        pretrained = AutoModel.from_pretrained('voidful/albert_chinese_tiny')
        model = model_class.Model(tokenizer=tokenizer, pretrained=pretrained, tasks_detail={"taskA": ["a", "b"]},
                                  maxlen=128)
        optim, scheduler = tfkit.train.optimizer(model, lr=0.1, total_step=10)
        print(optim, scheduler)
        optim.zero_grad()
        scheduler.step()

    def testMultiTask(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', MTTASK_MODEL_DIR, '--train', CLAS_DATASET, GEN_DATASET,
             '--lr', '5e-5', '--test', CLAS_DATASET, GEN_DATASET, '--model', 'clas', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + MTTASK_MODEL_DIR + ' --train ' + CLAS_DATASET + ' ' + GEN_DATASET + ' --lr 5e-5 --test ' + CLAS_DATASET + ' ' + GEN_DATASET + ' --model clas clm --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testClas(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', CLAS_MODEL_DIR, '--train',
             CLAS_DATASET, '--lr', '5e-5', '--test', CLAS_DATASET, '--model', 'clas', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + CLAS_MODEL_DIR + ' --train ' + CLAS_DATASET + ' --test ' + CLAS_DATASET + ' --model clas --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOneByOne(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONEBYONE_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--model', 'onebyone', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + ONEBYONE_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --model onebyone --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOnce(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCE_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--model', 'once', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + ONCE_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --model once --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOnceCTC(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCECTC_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '3e-4', '--test', GEN_DATASET, '--model', 'oncectc', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + ONCE_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --model oncectc --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenSeq2Seq(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 1 --savedir ' + SEQ2SEQ_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --model seq2seq --config prajjwal1/bert-small --maxlen 50 --selfkd True')
        self.assertTrue(result == 0)
        tfkit.train.main(
            ['--batch', '1', '--epoch', '1', '--savedir', SEQ2SEQ_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-4', '--test', GEN_DATASET, '--model', 'seq2seq', '--config',
             'prajjwal1/bert-small', '--maxlen', '20'])
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', SEQ2SEQ_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-4', '--test', GEN_DATASET, '--model', 'seq2seq', '--config',
             'prajjwal1/bert-small', '--maxlen', '20', '--likelihood', 'pos'])

    def testGenCLM(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 1 --savedir ' + CLM_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --model clm --config prajjwal1/bert-small --maxlen 50')
        self.assertTrue(result == 0)
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', CLM_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-4', '--test', GEN_DATASET, '--model', 'clm', '--config',
             'prajjwal1/bert-small', '--maxlen', '20'])

    def testGenWithSentLoss(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', MODEL_SAVE_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--model', 'onebyone', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + MODEL_SAVE_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --model onebyone --config voidful/albert_chinese_tiny  --maxlen 50')
        self.assertTrue(result == 0)

    def testQA(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', QA_MODEL_DIR, '--train',
             QA_DATASET, '--lr', '5e-5', '--test', QA_DATASET, '--model', 'qa', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'start_slice'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + QA_MODEL_DIR + ' --train ' + QA_DATASET + ' --test ' + QA_DATASET + ' --model qa --config voidful/albert_chinese_tiny --maxlen 512 --handle_exceed start_slice')
        self.assertTrue(result == 0)

    def testMCQ(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', MCQ_MODEL_DIR, '--train',
             MCQ_DATASET, '--lr', '5e-5', '--test', MCQ_DATASET, '--model', 'mcq', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'end_slice'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + MCQ_MODEL_DIR + ' --train ' + MCQ_DATASET + ' --test ' + MCQ_DATASET + ' --model mcq --config voidful/albert_chinese_tiny --maxlen 512 --handle_exceed end_slice')
        self.assertTrue(result == 0)

    def testMaskLM(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', MASK_MODEL_DIR, '--train',
             MASK_DATASET, '--lr', '3e-2', '--test', MASK_DATASET, '--model', 'mask', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + MASK_MODEL_DIR + ' --train ' + MASK_DATASET + ' --test ' + MASK_DATASET + ' --model mask --config voidful/albert_chinese_tiny --maxlen 512')
        self.assertTrue(result == 0)

    def testTag(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', TAG_MODEL_DIR, '--train',
             TAG_DATASET, '--lr', '5e-5', '--test', TAG_DATASET, '--model', 'tag', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'slide'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + TAG_MODEL_DIR + ' --train ' + TAG_DATASET + ' --test ' + TAG_DATASET + ' --model tag --config voidful/albert_chinese_tiny --maxlen 50 --handle_exceed slide')
        self.assertTrue(result == 0)

    def testTagCRF(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', TAGCRF_MODEL_DIR, '--train',
             TAG_DATASET, '--lr', '5e-5', '--test', TAG_DATASET, '--model', 'tagcrf', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'slide'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + TAG_MODEL_DIR + ' --train ' + TAG_DATASET + ' --test ' + TAG_DATASET + ' --model tag --config voidful/albert_chinese_tiny --maxlen 50 --handle_exceed slide')
        self.assertTrue(result == 0)

    def testAddTokenFreq(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ADDTOKFREQ_SAVE_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', ADDTOK_DATASET, '--model', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--add_tokens_freq', '1'])
        result = os.system(
            'tfkit-train --batch 2 --add_tokens_freq 1  --savedir ' + ADDTOKFREQ_SAVE_DIR + ' --epoch 2  --train ' + ADDTOK_DATASET + ' --test ' + ADDTOK_DATASET + ' --model clm --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testAddTokenFile(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ADDTOKFILE_SAVE_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', ADDTOK_DATASET, '--model', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--add_tokens_file', NEWTOKEN_FILE])
        result = os.system(
            f'tfkit-train --batch 2 --add_tokens_file {NEWTOKEN_FILE}  --savedir {ADDTOKFILE_SAVE_DIR} --epoch 2  --train {ADDTOK_DATASET}  --test {ADDTOK_DATASET} --model clm --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    @pytest.mark.skip()
    def testLoggerwandb(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCE_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--model', 'once', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--wandb'])
