import unittest

import pytest
from transformers import BertTokenizer, AutoModel

import tfkit
from tfkit.test import *
from tfkit.utility.model import load_model_class


class TestTrain(unittest.TestCase):

    def testHelp(self):
        result = os.system('tfkit-train -h')
        assert (result == 0)

    def test_parser(self):
        input_arg, model_arg = tfkit.train.parse_train_args(
            ['--task', 'once', '--train', 'train.csv', '--test', 'test.csv', '--config',
             'voidful/albert_chinese_tiny'])
        print(input_arg, model_arg)
        self.assertTrue(input_arg.get('task') == ['once'])
        self.assertTrue(isinstance(input_arg.get('train'), list))

        input_arg, model_arg = tfkit.train.parse_train_args(
            ['--task', 'once', '--train', 'train.csv', '--test', 'test.csv', '--config',
             'voidful/albert_chinese_tiny', '--likelihood', 'pos'])
        print(input_arg, model_arg)
        self.assertTrue(model_arg.get('likelihood') == 'pos')
        self.assertTrue(isinstance(input_arg.get('train'), list))

    def test_optimizer(self):
        model_class = load_model_class('clas')
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
             '--lr', '5e-5', '--test', CLAS_DATASET, GEN_DATASET, '--task', 'once', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + MTTASK_MODEL_DIR + ' --train ' + CLAS_DATASET + ' ' + GEN_DATASET + ' --lr 5e-5 --test ' + CLAS_DATASET + ' ' + GEN_DATASET + ' --task once clm --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOnce(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCE_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--task', 'once', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + ONCE_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --task once --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenOnceCTC(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCECTC_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '3e-4', '--test', GEN_DATASET, '--task', 'oncectc', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50'])
        result = os.system(
            'tfkit-train --batch 2 --epoch 2 --savedir ' + ONCE_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --task oncectc --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testGenSeq2Seq(self):
        # result = os.system(
        #     'tfkit-train --batch 2 --epoch 1 --savedir ' + SEQ2SEQ_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --task seq2seq --config prajjwal1/bert-small --maxlen 50 --selfkd True')
        # self.assertTrue(result == 0)
        tfkit.train.main(
            ['--batch', '1', '--epoch', '1', '--savedir', SEQ2SEQ_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-4', '--test', GEN_DATASET, '--task', 'seq2seq', '--config',
             'prajjwal1/bert-small', '--maxlen', '20'])
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', SEQ2SEQ_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-4', '--test', GEN_DATASET, '--task', 'seq2seq', '--config',
             'prajjwal1/bert-small', '--maxlen', '20', '--likelihood', 'pos'])

    def testGenCLM(self):
        result = os.system(
            'tfkit-train --batch 2 --epoch 1 --savedir ' + CLM_MODEL_DIR + ' --train ' + GEN_DATASET + ' --test ' + GEN_DATASET + ' --task clm --config prajjwal1/bert-small --maxlen 50')
        self.assertTrue(result == 0)
        tfkit.train.main(
            ['--batch', '2', '--epoch', '2', '--savedir', CLM_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-4', '--test', GEN_DATASET, '--task', 'clm', '--config',
             'prajjwal1/bert-small', '--maxlen', '20'])

    def testAddTokenFile(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ADDTOKFILE_SAVE_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', ADDTOK_DATASET, '--task', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '100', '--add_tokens_file', NEWTOKEN_FILE])
        result = os.system(
            f'tfkit-train --batch 2 --add_tokens_file {NEWTOKEN_FILE}  --savedir {ADDTOKFILE_SAVE_DIR} --epoch 2  --train {ADDTOK_DATASET}  --test {ADDTOK_DATASET} --task clm --config voidful/albert_chinese_tiny --maxlen 50')
        self.assertTrue(result == 0)

    def testResume(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCE_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--task', 'once', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--tag', 'testresume'])

        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCE_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--task', 'once', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--resume', os.path.join(ONCE_MODEL_DIR, "1.pt")])

    def testResumeMultiModel(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', MTTASK_MODEL_DIR, '--train', CLAS_DATASET, GEN_DATASET,
             '--lr', '5e-5', '--test', CLAS_DATASET, GEN_DATASET, '--task', 'once', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--tag', 'once', 'clm'])
        # resume to train all task
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', MTTASK_MODEL_DIR, '--train', CLAS_DATASET, GEN_DATASET,
             '--lr', '5e-5', '--test', CLAS_DATASET, GEN_DATASET, '--task', 'once', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--tag', 'once', 'clm', '--resume',
             os.path.join(MTTASK_MODEL_DIR, "1.pt")])
        # resume to train only one task
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', MTTASK_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--task', 'clm', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--resume', os.path.join(MTTASK_MODEL_DIR, "1.pt"),
             '--tag', 'clm'])

    @pytest.mark.skip()
    def testLoggerwandb(self):
        tfkit.train.main(
            ['--batch', '2', '--epoch', '1', '--savedir', ONCE_MODEL_DIR, '--train',
             GEN_DATASET, '--lr', '5e-5', '--test', GEN_DATASET, '--task', 'once', '--config',
             'voidful/albert_chinese_tiny', '--maxlen', '50', '--wandb'])

    # def testClas(self):
    #     tfkit.train.main(
    #         ['--batch', '2', '--epoch', '1', '--savedir', CLAS_MODEL_DIR, '--train',
    #          CLAS_DATASET, '--lr', '5e-5', '--test', CLAS_DATASET, '--task', 'clas', '--config',
    #          'voidful/albert_chinese_tiny', '--maxlen', '50'])
    #     result = os.system(
    #         'tfkit-train --batch 2 --epoch 2 --savedir ' + CLAS_MODEL_DIR + ' --train ' + CLAS_DATASET + ' --test ' + CLAS_DATASET + ' --task clas --config voidful/albert_chinese_tiny --maxlen 50')
    #     self.assertTrue(result == 0)
    #
    # def testQA(self):
    #     tfkit.train.main(
    #         ['--batch', '2', '--epoch', '1', '--savedir', QA_MODEL_DIR, '--train',
    #          QA_DATASET, '--lr', '5e-5', '--test', QA_DATASET, '--task', 'qa', '--config',
    #          'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'start_slice'])
    #     result = os.system(
    #         'tfkit-train --batch 2 --epoch 2 --savedir ' + QA_MODEL_DIR + ' --train ' + QA_DATASET + ' --test ' + QA_DATASET + ' --task qa --config voidful/albert_chinese_tiny --maxlen 512 --handle_exceed start_slice')
    #     self.assertTrue(result == 0)
    #
    # def testTag(self):
    #     tfkit.train.main(
    #         ['--batch', '2', '--epoch', '1', '--savedir', TAG_MODEL_DIR, '--train',
    #          TAG_DATASET, '--lr', '5e-5', '--test', TAG_DATASET, '--task', 'tag', '--config',
    #          'voidful/albert_chinese_tiny', '--maxlen', '512', '--handle_exceed', 'slide'])
    #     result = os.system(
    #         'tfkit-train --batch 2 --epoch 2 --savedir ' + TAG_MODEL_DIR + ' --train ' + TAG_DATASET + ' --test ' + TAG_DATASET + ' --task tag --config voidful/albert_chinese_tiny --maxlen 50 --handle_exceed slide')
    #     self.assertTrue(result == 0)
