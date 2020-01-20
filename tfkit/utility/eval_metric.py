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
        self.tasks[task]['targets'].append(targets)
        self.tasks[task]['target'].append(target[0])

    def get_record(self, field='predicted', task='default'):
        return self.tasks[task][field]

    def cal_score(self, metric):
        for name, task in self.tasks.items():
            print("Task : " + name + " report ")
            if "em" in metric:
                em = 0
                total = 0
                for pos, predict in enumerate(task['predicted']):
                    targets = task['target'][pos]
                    equal = False
                    for target in targets:
                        if predict[0].replace("[SEP]", "").replace(" ", "") == target.replace("[SEP]", "").replace(" ",
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
                mlb = MultiLabelBinarizer().fit(task['target'])
                result = classification_report(mlb.transform(task['predicted']),
                                               mlb.transform(task['target']),
                                               target_names=list(mlb.classes_))
            yield (name, result)
