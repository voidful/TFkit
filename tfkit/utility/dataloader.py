from collections import defaultdict

import nlp2


def get_gen_data_from_file(fpath, chunksize=10000):
    tasks = defaultdict(list)
    task = 'default'
    tasks[task] = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        source_text = row[0].strip()
        target_text = row[1].strip()
        negative_text = row[2].strip() if len(row) > 2 else None
        yield tasks, task, source_text, [target_text, negative_text]


def get_qa_data_from_file(fpath, chunksize=10000):
    tasks = defaultdict(list)
    task = 'default'
    tasks[task] = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        context, start, end = row
        yield tasks, task, context, [start, end]


def get_tag_data_from_file(fpath, chunksize=10000, text_index: int = 0, label_index: int = 1, separator=" ", **kwargs):
    tasks = defaultdict(list)
    task = 'default'
    labels = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        for i in row[1].split(separator):
            if i not in labels and len(i.strip()) > 0:
                labels.append(i)
                labels.sort()
    tasks[task] = labels
    for row in nlp2.read_csv_row(fpath, chunksize):
        yield tasks, task, row[text_index].strip(), [row[label_index].strip()]
