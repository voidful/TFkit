from collections import defaultdict


class EvalMetric:

    def __init__(self):
        self.tasks = defaultdict(lambda: defaultdict(list))

    def add_record(self, predicted, target, task='default'):
        self.tasks[task]['predicted'].append(predicted)
        self.tasks[task]['target'].append(target)

    def cal_score(self, metric, config):
        for name, task in self.tasks.items():
            print("Task : " + name + " report ")
            if "em" in metric:
                em = 0
                total = 0
                for pos, p in enumerate(task['predicted']):
                    predict = p.replace("[SEP]", "").replace(" ", "")
                    target = task['target'][pos].replace("[SEP]", "").replace(" ", "")
                    if predict == target:
                        em += 1
                    total += 1
                result = em / total
            if "nlg" in metric:
                from nlgeval import NLGEval
                nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR"])
                result = nlgeval.compute_metrics(ref_list=task['target'],
                                                 hyp_list=task['predicted'])
            if "classification" in metric:
                from sklearn.metrics import classification_report
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer().fit(task['target'])
                result = classification_report(mlb.transform(task['target']),
                                               mlb.transform(task['target']),
                                               target_names=list(mlb.classes_))
            print(result)
            yield (name, result)
