from collections import defaultdict
import string
import re
from collections import Counter


def _normalize_answer(s):
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

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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

    def __init__(self, tokenizer, max_candidate=6):
        self.tasks = defaultdict(lambda: defaultdict(list))
        self.max_candidate = max_candidate
        self.tokenizer = tokenizer
        self.target_list = defaultdict(lambda: defaultdict(int))

    def tokenize_text(self, text):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text))

    def add_record(self, input, predicted, target, task='default'):
        if isinstance(input, str):
            input = self.tokenize_text(input.strip())
        if isinstance(input, list):
            for i, t in enumerate(input):
                input[i] = self.tokenize_text(t)

        if isinstance(predicted, str):
            predicted = self.tokenize_text(predicted)
        if isinstance(predicted, list):
            for i, t in enumerate(predicted):
                predicted[i] = self.tokenize_text(t)

        if isinstance(target, str):
            targets = []
            if "[SEP]" in target:
                targets.extend([self.tokenize_text(st.strip()) for st in target.split("[SEP]")])
            else:
                targets.append(self.tokenize_text(target.strip()))
        if isinstance(target, list):
            for i, t in enumerate(target):
                target[i] = self.tokenize_text(t)
            targets = target

        if self.max_candidate - len(targets) > 0:
            targets.extend([""] * (self.max_candidate - len(targets)))

        for t in targets:
            self.target_list[task][t] += 1

        self.tasks[task]['input'].append(input)
        self.tasks[task]['predicted'].append(predicted)
        self.tasks[task]['predicteds'].append([predicted])
        self.tasks[task]['target'].append(target)
        self.tasks[task]['targets'].append(targets)

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
                    em_list = [0]
                    f1_list = [0]
                    for target in task['targets'][pos]:
                        equal = False
                        if _normalize_answer(predict) == _normalize_answer(target) and len(
                                _normalize_answer(predict)) > 0:
                            equal = True
                        if equal:
                            em_score = 1
                            f1_score = 1
                            em_list.append(em_score)
                            f1_list.append(f1_score)
                        else:
                            em_score = 0
                            f1_score = _f1_score(predict, target)
                            f1_list.append(f1_score)
                        data_score.append([predict, target, {'em': em_score, 'f1': f1_score}])
                    em += max(em_list)
                    f1 += max(f1_list)
                    total += 1
                result = {"EM": em / (total or not total), "F1": f1 / (total or not total)}
                data_score = sorted(data_score, key=lambda i: i[2]['em'])
            if "nlg" in metric:
                try:
                    from nlgeval import NLGEval
                except ImportError:
                    print(
                        "nlg-eval package not install, plz install it: pip install git+https://github.com/voidful/nlg-eval.git ; nlg-eval --setup ./nlg-eval-data/")
                    raise
                nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR"])
                nlgeval.compute_metrics(ref_list=['abc'],  # transpose
                                        hyp_list='abc')
                targets = task['targets']
                predicted = task['predicted']
                for t, p in zip(targets, predicted):
                    data_score.append([p, t, nlgeval.compute_metrics(ref_list=list(map(list, zip(t))), hyp_list=[p])])
                result = nlgeval.compute_metrics(ref_list=list(map(list, zip(*task['targets']))),  # transpose
                                                 hyp_list=predicted)
                data_score = sorted(data_score, key=lambda i: i[2]['ROUGE_L'])
            if "clas" in metric:
                from sklearn.metrics import classification_report
                from sklearn.preprocessing import MultiLabelBinarizer
                from sklearn.metrics import precision_recall_fscore_support
                target_key = [t for t in self.target_list[task_name].keys() if len(t) > 0]
                mlb = MultiLabelBinarizer().fit([target_key])
                # remove all blank target
                task['targets'] = [[j for j in sub if len(j) > 0] for sub in task['targets']]
                # modify for tagging result
                if isinstance(task['predicteds'][0][0], list):
                    task['targets'] = sum([[[j] for j in sub] for sub in task['targets']], [])
                    task['predicteds'] = sum([[[j] for j in sub] for sub in task['predicted']], [])
                    if len(task['targets']) != len(task['predicteds']):
                        diff = len(task['targets']) - len(task['predicteds'])
                        task['predicteds'].extend([['']] * diff)
                targets = task['targets']
                predicted = task['predicteds']
                for p, t in zip(predicted, targets):
                    score = dict(zip(["precision", "recall", "fbeta_score", "support"],
                                     precision_recall_fscore_support(mlb.transform([t]), mlb.transform([p]),
                                                                     average='weighted')))
                    data_score.append([p, t, score])
                result = classification_report(
                    mlb.transform(targets),
                    mlb.transform(predicted),
                    target_names=list(mlb.classes_))
                data_score = sorted(data_score, key=lambda i: i[2]['fbeta_score'])
            yield (task_name, result, data_score)
