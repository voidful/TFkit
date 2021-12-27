import csv
from collections import defaultdict
import nlp2
from tfkit.utility import tok


def get_multiclas_data_from_file(fpath, chunksize=10000):
    tasks = defaultdict(list)
    with open(fpath, 'r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        headers = ['input'] + ['target_' + str(i) for i in range(len(fieldnames) - 1)]

        is_multi_label = ""
        for row in nlp2.read_csv_row(fpath, chunksize):
            if tok.UNIVERSAL_SEP in row[1]:
                is_multi_label = "_multi_label"
                break

        for row in nlp2.read_csv_row(fpath, chunksize):
            start_pos = 1
            for pos, item in enumerate(row[start_pos:]):
                pos += start_pos
                task = headers[0] + "_" + headers[pos] + is_multi_label
                item = item.strip()
                if tok.UNIVERSAL_SEP in item:
                    for i in item.split(tok.UNIVERSAL_SEP):
                        tasks[task].append(i) if i not in tasks[task] else tasks[task]
                else:
                    tasks[task].append(item) if item not in tasks[task] else tasks[task]
                tasks[task].sort()

        for row in nlp2.read_csv_row(fpath, chunksize):
            start_pos = 1
            for pos, item in enumerate(row[start_pos:]):
                pos += start_pos
                task = headers[0] + "_" + headers[pos] + is_multi_label
                item = item.strip()
                target = item.split(tok.UNIVERSAL_SEP) if tok.UNIVERSAL_SEP in item else [item]
                input = row[0]
                yield tasks, task, input, target


def get_clas_data_from_file(fpath, chunksize=10000):
    tasks = defaultdict(list)
    task = 'default'
    tasks[task] = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        source_text = row[0]
        target_text = row[1]
        yield tasks, task, source_text, [target_text]


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
