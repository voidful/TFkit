import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))

DATASET_DIR = os.path.join(ROOT_DIR, 'demo_data')
TAG_DATASET = os.path.join(DATASET_DIR, 'tag.csv')
CLAS_DATASET = os.path.join(DATASET_DIR, 'classification.csv')
GEN_DATASET = os.path.join(DATASET_DIR, 'generation.csv')
MASK_DATASET = os.path.join(DATASET_DIR, 'mask.csv')
MCQ_DATASET = os.path.join(DATASET_DIR, 'mcq.csv')
QA_DATASET = os.path.join(DATASET_DIR, 'qa.csv')
ADDTOK_DATASET = os.path.join(DATASET_DIR, 'unk_tok.csv')
NEWTOKEN_FILE = os.path.join(DATASET_DIR, 'tok_list.txt')

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, 'tfkit/test/cache/')
ADDTOKFREQ_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, 'addtokfreq/')
ADDTOKFILE_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, 'addtokfile/')
CLAS_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'clas/')
TAG_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'tag/')
TAGCRF_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'tagcrf/')
ONEBYONE_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'onebyone/')
CLM_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'clm/')
SEQ2SEQ_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'seq2seq/')
ONCE_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'once/')
ONCECTC_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'oncectc/')
MASK_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'mask/')
MCQ_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'mcq/')
QA_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'qa/')
MTTASK_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, 'mttask/')

ONEBYONE_MODEL_PATH = os.path.join(ONEBYONE_MODEL_DIR, '2.pt')
ONCE_MODEL_PATH = os.path.join(ONCE_MODEL_DIR, '2.pt')
ONCECTC_MODEL_PATH = os.path.join(ONCECTC_MODEL_DIR, '30.pt')
SEQ2SEQ_MODEL_PATH = os.path.join(SEQ2SEQ_MODEL_DIR, '10.pt')
CLM_MODEL_PATH = os.path.join(CLM_MODEL_DIR, '20.pt')
CLAS_MODEL_PATH = os.path.join(CLAS_MODEL_DIR, '2.pt')
MASK_MODEL_PATH = os.path.join(MASK_MODEL_DIR, '2.pt')
MCQ_MODEL_PATH = os.path.join(MCQ_MODEL_DIR, '2.pt')
TAG_MODEL_PATH = os.path.join(TAG_MODEL_DIR, '2.pt')
QA_MODEL_PATH = os.path.join(QA_MODEL_DIR, '2.pt')
ADDTOKFREQ_MODEL_PATH = os.path.join(ADDTOKFREQ_SAVE_DIR, '2.pt')
ADDTOKFILE_MODEL_PATH = os.path.join(ADDTOKFILE_SAVE_DIR, '2.pt')