import json
import re
import collections
import string
import sys
import random

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()




def compute_exact(a_gold, a_pred):
    to_return =0
    for pred in a_pred:
        if normalize_answer(a_gold) == normalize_answer(pred):
            to_return= 1
    return to_return


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    if a_gold.lower() in [pred.lower() for pred in a_pred]:
        sel_pred = a_gold
    else:
        sel_pred = random.choice(a_pred)

    pred_toks = get_tokens(sel_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(examples, reference):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    
    for example in examples:
        qas_id = example['question_id']
        gold_answers = [reference['reference'][qas_id]]

        predictions = example['pred']
        exact_scores[qas_id] = max(compute_exact(a, predictions) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, predictions) for a in gold_answers)

    qid_list = reference['reference'].keys()
    qid_list = list(qid_list)
    total = len(qid_list)
    
    
    table_list = reference['table']
    passage_list = reference['passage']
    #TODO: What is table exact, passage exact scores ?

    for tid in table_list:
        if not tid in exact_scores.keys():
            table_list.remove(tid)
    for qid in qid_list:
        if not qid in exact_scores.keys():
            qid_list.remove(qid)
    total = len(qid_list)


    ### For dev set, we know where the gold answer is coming from so we can compute table exact and passage exact

    return collections.OrderedDict(
        [
            ("table exact", 100.0 * sum(exact_scores[k] if k in exact_scores.keys() else 0 for k in table_list) / len(table_list)),
            ("table f1", 100.0 * sum(f1_scores[k] if k in f1_scores.keys() else 0 for k in table_list) / len(table_list)),
            ("passage exact", 100.0 * sum(exact_scores[k] if k in exact_scores.keys() else 0 for k in passage_list) / len(passage_list)),
            ("passage f1", 100.0 * sum(f1_scores[k] for k in passage_list) / len(passage_list)),
            ("total exact", 100.0 * sum(exact_scores[k] if k in exact_scores.keys() else 0 for k in qid_list) / total),
            ("total f1", 100.0 * sum(f1_scores[k] if k in f1_scores.keys() else 0 for k in qid_list) / total),
            ("total", total),
        ]
    )

assert len(sys.argv) == 3, "you need to input the file"

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

with open(sys.argv[2], 'r') as f:
    ref = json.load(f)

print(get_raw_scores(data, ref))
