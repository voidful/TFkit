import csv
from collections import defaultdict
import nlp2
import numpy as np

from tfkit.utility import tok


def get_multiclas_data_from_file(fpath, chunksize=10000):
    task_label_dict = defaultdict(list)
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
                        task_label_dict[task].append(i) if i not in task_label_dict[task] else task_label_dict[task]
                else:
                    task_label_dict[task].append(item) if item not in task_label_dict[task] else task_label_dict[task]
                task_label_dict[task].sort()

        datas = []
        for row in nlp2.read_csv_row(fpath, chunksize):
            start_pos = 1
            for pos, item in enumerate(row[start_pos:]):
                pos += start_pos
                task = headers[0] + "_" + headers[pos] + is_multi_label
                item = item.strip()
                target = item.split(tok.UNIVERSAL_SEP) if tok.UNIVERSAL_SEP in item else [item]
                input = row[0]
                datas.append({"task": task, "input": input, "target": target})
        return task_label_dict, datas


def get_clas_data_from_file(fpath, chunksize=10000):
    task_label_dict = defaultdict(list)
    task = 'clas'
    task_label_dict[task] = []
    datas = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        source_text = row[0]
        target_text = row[1]
        if target_text not in task_label_dict[task]:
            task_label_dict[task].append(target_text)
        datas.append({"task": task, "input": source_text, "target": target_text})
    return task_label_dict, datas


def get_gen_data_from_file(fpath, chunksize=10000):
    task_label_dict = defaultdict(list)
    task = 'gen'
    task_label_dict[task] = []
    datas = []
    print("Reading data from file...")
    for row in nlp2.read_csv_row(fpath, chunksize):
        if len(row) > 1:
            source_text = str(row[0]).strip()
            target_text = str(row[1]).strip()
            negative_text = str(row[2]).strip() if len(row) > 2 else None
            datas.append({"task": task, "input": source_text, "target": target_text, "ntarget": negative_text})
    return task_label_dict, datas


def get_qa_data_from_file(fpath, chunksize=10000):
    task_label_dict = defaultdict(list)
    task = 'qa'
    task_label_dict[task] = []
    datas = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        context, start, end = row
        datas.append({"task": task, "input": context, "start": start, "end": end})
    return task_label_dict, datas


def get_tag_data_from_file(fpath, chunksize=10000, text_index: int = 0, label_index: int = 1, separator=" "):
    task_label_dict = defaultdict(list)
    task = 'tag'
    labels = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        for i in row[1].split(separator):
            if i not in labels and len(i.strip()) > 0:
                labels.append(i)
                labels.sort()
    task_label_dict[task] = labels
    datas = []
    for row in nlp2.read_csv_row(fpath, chunksize):
        datas.append({"task": task, "input": row[text_index].strip(), "target": row[label_index].strip()})
    return task_label_dict, datas
