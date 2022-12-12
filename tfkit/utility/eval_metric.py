import copy
import re
import string
from collections import Counter
from collections import defaultdict

from tfkit.utility import tok
from tqdm.auto import tqdm


def _normalize_answer(s, task='emf1'):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        if len(text) > 1:
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        else:
            return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if task == 'emf1':
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    else:
        return white_space_fix((remove_punc(lower(s))))


def _f1_score(prediction, ground_truth):
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class EvalMetric:

    def __init__(self, tokenizer, normalize_text=True):
        self.tasks = defaultdict(lambda: defaultdict(list))
        self.tokenizer = tokenizer
        self.target_list = defaultdict(lambda: defaultdict(int))
        self.normalize_text = normalize_text

    def tokenize_text(self, text):
        text = self.tokenizer.decode(self.tokenizer.encode(text, add_special_tokens=False))
        if self.normalize_text:
            text = text.replace(tok.tok_sep(self.tokenizer), " ")
            # return  _normalize_answer(text, task='others')  # remove punctuation
            # keep punctuation
            text = "".join(
                (char if char.isalpha() or char == " " else " " + char + " ") for char in text)  # separate punctuation
            text = ' '.join(text.split()).lower().strip()  # remove extra blank
        return text

    def add_record(self, ori_input, ori_predicted, ori_target, task='default'):
        input = predicted = target = ""
        input_list = predicted_list = ori_predicted_list = target_list = []

        if isinstance(ori_input, str):
            input = self.tokenize_text(ori_input.strip())
            input_list = [input]
        if isinstance(ori_input, list):
            input_list = copy.copy(ori_input)
            for i, t in enumerate(ori_input):
                input_list[i] = self.tokenize_text(t.strip())
            input = " ".join(input_list)

        if isinstance(ori_predicted, str):
            predicted = self.tokenize_text(ori_predicted)
            predicted_list = [predicted]
            ori_predicted_list = [ori_predicted]
        if isinstance(ori_predicted, list):
            predicted_list = copy.copy(ori_predicted)
            ori_predicted_list = copy.copy(ori_predicted)
            for i, t in enumerate(ori_predicted):
                if not isinstance(t, list):
                    predicted_list[i] = self.tokenize_text(t.strip())
                    ori_predicted_list[i] = t
                else:
                    predicted_list[i] = ''
                    ori_predicted_list[i] = ''
            predicted = " ".join(predicted_list)
        if isinstance(ori_target, str):
            target_list = []
            if tok.UNIVERSAL_SEP in ori_target:
                target = ori_target
                target_list.extend([self.tokenize_text(st.strip()) for st in ori_target.split(tok.UNIVERSAL_SEP)])
            else:
                target = self.tokenize_text(ori_target.strip())
                target_list.append(target)
        elif isinstance(ori_target, list):
            for i, t in enumerate(ori_target):
                if isinstance(t, list):
                    ori_target[i] = self.tokenize_text(t.strip())

            target_list = ori_target

        for t in target_list:
            self.target_list[task][t] += 1

        self.tasks[task]['input'].append(input)
        self.tasks[task]['input_list'].append(input_list)
        self.tasks[task]['predicted'].append(predicted)
        self.tasks[task]['predicted_list'].append(predicted_list)
        self.tasks[task]['target'].append(target)
        self.tasks[task]['target_list'].append(target_list)
        self.tasks[task]['ori_input'].append(ori_input)
        self.tasks[task]['ori_predicted'].append(ori_predicted)
        self.tasks[task]['ori_predicted_list'].append(ori_predicted_list)
        self.tasks[task]['ori_target'].append(ori_target)

    def get_record(self, task='default'):
        return self.tasks[task]

    def cal_score(self, metric):
        data_score = []
        for task_name, task in self.tasks.items():
            print("Task : " + task_name + " report ")
            if "emf1" in metric:
                em = 0
                total = 0
                f1 = 0
                for pos, predict in enumerate(task['predicted']):
                    em_list = []
                    f1_list = []
                    for target in task['target_list'][pos]:
                        if _normalize_answer(str(predict)) == _normalize_answer(str(target)) and len(
                                _normalize_answer(str(predict))) > 0 or len(str(predict)) == len(str(target)) == 0:
                            em_score = 1
                            f1_score = 1
                        else:
                            em_score = 0
                            f1_score = _f1_score(str(predict), str(target))
                        em_list.append(em_score)
                        f1_list.append(f1_score)
                    em += max(em_list)
                    f1 += max(f1_list)
                    data_score.append([predict, task['target_list'][pos][em_list.index(max(em_list))],
                                       {'em': max(em_list), 'f1': max(f1_list)}])
                    total += 1
                result = {"EM": em / (total or not total), "F1": f1 / (total or not total)}
                data_score = sorted(data_score, key=lambda i: i[2]['em'], reverse=True)
            if "er" in metric:
                try:
                    import asrp
                except ImportError:
                    print(
                        "asrp package not install, plz install it: pip install asrp")
                    raise
                predicts = []
                targets = []
                for pos, predict in enumerate(task['predicted']):
                    wer_list = []
                    cer_list = []
                    for target in task['target_list'][pos]:
                        if len(target) > 0 and len(predict) > 0:
                            wer_list.append(100 * asrp.wer([target], [predict]))
                            cer_list.append(100 * asrp.cer([target], [predict]))
                        else:
                            wer_list.append(100)
                            cer_list.append(100)
                    wer = min(wer_list)
                    cer = min(cer_list)
                    target = task['target_list'][pos][wer_list.index(wer)]
                    predicts.append(predict)
                    targets.append(target)
                    data_score.append([predict, target, {'wer': wer, 'cer': cer}])

                wer = 100 * asrp.wer(targets, predicts) if len(target) > 0 else 100
                cer = 100 * asrp.cer(targets, predicts) if len(target) > 0 else 100
                result = {"WER": wer, "CER": cer}
                data_score = sorted(data_score, key=lambda i: i[2]['wer'], reverse=False)
            if "nlg" in metric:
                try:
                    from nlgeval import NLGEval
                except ImportError:
                    print(
                        "nlg-eval package not install, plz install it: pip install git+https://github.com/voidful/nlg-eval.git ; nlg-eval --setup ./nlg-eval-data/")
                    raise
                nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR"])

                target_list = task['target_list']
                predicted = task['predicted']
                for idx, tl in enumerate(target_list):
                    max_candidate = max([len(i) for i in target_list])
                    if max_candidate - len(tl) > 0:
                        target_list[idx].extend([""] * (max_candidate - len(tl)))

                for t, p in tqdm(zip(target_list, predicted), total=len(target_list)):
                    data_score.append([p, t, nlgeval.compute_metrics(ref_list=list(map(list, zip(t))), hyp_list=[p])])
                result = nlgeval.compute_metrics(ref_list=list(map(list, zip(*task['target_list']))),  # transpose
                                                 hyp_list=predicted)
                data_score = sorted(data_score, key=lambda i: i[2]['ROUGE_L'])
            if "clas" in metric:
                from sklearn.metrics import classification_report
                from sklearn.preprocessing import MultiLabelBinarizer
                from sklearn.metrics import precision_recall_fscore_support
                target_key = [t for t in self.target_list[task_name].keys() if len(t) > 0]
                mlb = MultiLabelBinarizer().fit([target_key])
                # remove all blank target
                task['target_list'] = [[j for j in sub if len(j) > 0] for sub in task['target_list']]
                # modify for tagging result
                if isinstance(task['ori_predicted_list'][0][0], list):
                    target_list = sum([[[j] for j in sub] for sub in task['target_list']], [])
                    predicted = sum([[[j] for j in sub] for sub in task['ori_predicted_list']], [])
                    if len(target_list) != len(predicted):
                        diff = len(task['target_list']) - len(task['ori_predicted_list'])
                        predicted.extend([['']] * diff)
                else:
                    target_list = task['target_list']
                    predicted = task['ori_predicted_list']

                for p, t in zip(predicted, target_list):
                    score = dict(zip(["precision", "recall", "fbeta_score", "support"],
                                     precision_recall_fscore_support(mlb.transform([t]), mlb.transform([p]),
                                                                     average='weighted')))
                    data_score.append([p, t, score])
                result = classification_report(
                    mlb.transform(target_list),
                    mlb.transform(predicted),
                    target_names=list(mlb.classes_))
                data_score = sorted(data_score, key=lambda i: i[2]['fbeta_score'])
            yield (task_name, result, data_score)
