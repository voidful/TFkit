from collections import defaultdict


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
            if "em" in metric:
                em = 0
                total = 0
                for pos, predict in enumerate(task['predicted']):
                    target = task['target'][pos]
                    equal = False
                    if predict.replace("[SEP]", "").replace(" ", "") == target.replace("[SEP]", "").replace(" ",
                                                                                                            ""):
                        equal = True
                    em += 1 if equal else 0
                    total += 1
                result = em / total
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
