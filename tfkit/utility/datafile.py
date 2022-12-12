import csv
from collections import defaultdict

import nlp2


# ignore sklearn warning
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from tqdm.auto import tqdm

from tfkit.utility import tok


def get_multiclas_data_from_file(fpath):
    task_label_dict = defaultdict(list)
    with open(fpath, 'r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        headers = ['input'] + ['target_' + str(i) for i in range(len(fieldnames) - 1)]

        is_multi_label = ""
        for rows in nlp2.read_csv_chunk(fpath, ','):
            for row in rows:
                if tok.UNIVERSAL_SEP in row[1]:
                    is_multi_label = "_multi_label"
                    break

        for rows in nlp2.read_csv_chunk(fpath, ','):
            for row in rows:
                start_pos = 1
                for pos, item in enumerate(row[start_pos:]):
                    pos += start_pos
                    task = headers[0] + "_" + headers[pos] + is_multi_label
                    item = item.strip()
                    if tok.UNIVERSAL_SEP in item:
                        for i in item.split(tok.UNIVERSAL_SEP):
                            task_label_dict[task].append(i) if i not in task_label_dict[task] else task_label_dict[task]
                    else:
                        task_label_dict[task].append(item) if item not in task_label_dict[task] else task_label_dict[
                            task]
                    task_label_dict[task].sort()

        for rows in nlp2.read_csv_chunk(fpath, ','):
            chunk = []
            for row in rows:
                start_pos = 1
                for pos, item in enumerate(row[start_pos:]):
                    pos += start_pos
                    task = headers[0] + "_" + headers[pos] + is_multi_label
                    item = item.strip()
                    targets = item.split(tok.UNIVERSAL_SEP) if tok.UNIVERSAL_SEP in item else [item]
                    targets = [task_label_dict[task][task_label_dict[task].index(target)] for target in targets]
                    input = row[0]
                    chunk.append({"task": task, "input": input, "target": targets})
            yield chunk
        return task_label_dict


def get_clas_data_from_file(fpath):
    task_label_dict = defaultdict(list)
    task = 'clas'
    task_label_dict[task] = []
    for rows in nlp2.read_csv_chunk(fpath, ','):
        chunk = []
        for row in rows:
            source_text = row[0]
            target_text = row[1]
            if target_text not in task_label_dict[task]:
                task_label_dict[task].append(target_text)
            chunk.append({"task": task, "input": source_text, "target": task_label_dict[task].index(target_text)})
        yield chunk
    return task_label_dict


def get_gen_data_from_file(fpath):
    task_label_dict = defaultdict(list)
    task = 'gen'
    task_label_dict[task] = []
    print("Reading data from file...")
    for rows in nlp2.read_csv_chunk(fpath, ','):
        chunk = []
        for row in rows:
            source_text = str(row[0]).strip()
            target_text = str(row[1]).strip()
            negative_text = str(row[2]).strip() if len(row) > 2 else None
            chunk.append({"task": task, "input": source_text, "target": target_text, "ntarget": negative_text})
        yield chunk
    return task_label_dict


def get_qa_data_from_file(fpath):
    task_label_dict = defaultdict(list)
    task = 'qa'
    task_label_dict[task] = []
    for rows in nlp2.read_csv_chunk(fpath, ','):
        chunk = []
        for row in rows:
            context, start, end = row
            chunk.append({"task": task, "input": context, "target": [start, end]})
        yield chunk
    return task_label_dict


def get_tag_data_from_file(fpath, text_index: int = 0, label_index: int = 1, separator=" "):
    task_label_dict = defaultdict(list)
    task = 'tag'
    labels = []
    for rows in nlp2.read_csv_chunk(fpath, ','):
        for row in rows:
            for i in row[1].split(separator):
                if i not in labels and len(i.strip()) > 0:
                    labels.append(i)
                    labels.sort()
    task_label_dict[task] = labels

    for rows in nlp2.read_csv_chunk(fpath, ','):
        chunk = []
        for row in rows:
            chunk.append({"task": task, "input": row[text_index].strip(), "target": row[label_index].strip(),
                          'separator': separator})
        yield chunk
    return task_label_dict


def get_tag_data_from_file_col(fpath, text_index: int = 0, label_index: int = 1, separator=" ", **kwargs):
    tasks = defaultdict(list)
    task = 'default'
    labels = []
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in tqdm(lines):
            rows = line.split(separator)
            if len(rows) > 1:
                if rows[label_index] not in labels and len(rows[label_index]) > 0:
                    labels.append(rows[label_index])
                    labels.sort()
    tasks[task] = labels
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        x, y = "", ""
        for line in tqdm(lines):
            rows = line.split(separator)
            if len(rows) == 1:
                yield tasks, task, x.strip(), [y.strip()]
                x, y = "", ""
            else:
                if len(rows[text_index]) > 0:
                    x += rows[text_index].replace(" ", "_") + separator
                    y += rows[label_index].replace(" ", "_") + separator
