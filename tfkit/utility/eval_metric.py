from collections import defaultdict
import string
import re
from collections import Counter


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

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

    def __init__(self, max_candidate=6):
        self.tasks = defaultdict(lambda: defaultdict(list))
        self.max_candidate = max_candidate

    def add_record(self, predicted, target, task='default'):
        if "[SEP]" in target:
            target = target.split("[SEP]")
        else:
            target = [target]

        targets = []
        for pos in range(self.max_candidate):
            if len(target) > pos:
                targets.append(target[pos])
            else:
                targets.append("")

        self.tasks[task]['predicted'].append(predicted)
        self.tasks[task]['predicted_list'].append(predicted.split(" ") if " " in predicted else list(predicted))
        self.tasks[task]['targets'].append(targets)
        self.tasks[task]['target'].append(target[0])
        self.tasks[task]['target_list'].append(target[0].split(" ") if " " in target[0] else list(target[0]))

    def get_record(self, task='default'):
        return self.tasks[task]['predicted']

    def cal_score(self, metric):
        for name, task in self.tasks.items():
            print("Task : " + name + " report ")
            if "emf1" in metric:
                em = 0
                total = 0
                f1 = 0
                for pos, predict in enumerate(task['predicted']):
                    target = task['target'][pos]
                    equal = False
                    if _normalize_answer(predict) == _normalize_answer(target):
                        equal = True
                    f1 += _f1_score(predict, target)
                    if equal:
                        em += 1
                    total += 1
                result = {"EM": em / total, "F1": f1 / total}
            if "nlg" in metric:
                try:
                    from nlgeval import NLGEval
                except ImportError:
                    print("nlg-eval package not install, plz install it from https://github.com/Maluuba/nlg-eval")

                nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR"])
                result = nlgeval.compute_metrics(ref_list=list(map(list, zip(*task['targets']))),  # transpose
                                                 hyp_list=task['predicted'])
            if "classification" in metric:
                from sklearn.metrics import classification_report
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer().fit(task['target_list'])
                result = classification_report(mlb.transform(task['predicted_list']),
                                               mlb.transform(task['target_list']),
                                               target_names=list(mlb.classes_))
            yield (name, result)
