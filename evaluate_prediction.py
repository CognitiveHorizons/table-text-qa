import json
import re
import collections
import string
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import tqdm

################################################################
model_name = 'msmarco-distilbert-base-tas-b'
doc_retriever = SentenceTransformer(model_name)

def get_top_id(passages,query):
    corpus_embeddings = doc_retriever.encode(passages, convert_to_tensor=True, show_progress_bar=False)
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    query_embeddings = doc_retriever.encode([query], convert_to_tensor=True)
    query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=1)
    return hits[0][0]['corpus_id']
#################################################################
import numpy as np
p = json.load(open('/mnt/infonas/data/yashgupta/data/row_selector_output/qid_logits_bert_large_dev.json'))
def get_rs_score(q_id, rank):
    return np.sort(np.array(p[q_id]))[-5:][rank]

q = json.load(open('/tmp/nbest_predictions_dev_top5.json'))
def get_rc_score(q_id, rank):
    nb = q[q_id+"_"+str(rank)][0]
    return nb['start_logit'] + nb['end_logit']
    # return np.log(nb['probability'])
#################################################################

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
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
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
    rnk_scores = {}
    eval_res = {}
    # bst = set([])
    # psgs = {}
    # for ex in tqdm.tqdm(examples):
    #     qid = ex['question_id'].split('_')[0]
    #     if qid not in psgs.keys():
    #         psgs[qid] = [ex['context']]
    #     else:
    #         psgs[qid].append(ex['context'])
    #     if len(psgs[qid])==3:
    #         bst.add(qid+"_"+str(get_top_id(psgs[qid], ex['question'])))

    for example in tqdm.tqdm(examples):
        qas_id = example['question_id'].split('_')[0]
        qas_rank = int(example['question_id'].split('_')[1])
        # rcs = get_rc_score(qas_id, qas_rank)
        # rss = get_rs_score(qas_id, qas_rank)
        ts = example['rerank_score'] #3.2*rss + rcs
        # print(example['question_id'], ts)
        gold_answers = [reference['reference'][qas_id]]

        prediction = example['pred']

        es = max(compute_exact(a, prediction) for a in gold_answers)
        f1s = max(compute_f1(a, prediction) for a in gold_answers)
        
        eval_res[example['question_id']] = {}
        eval_res[example['question_id']]['f1-score'] = f1s
        eval_res[example['question_id']]['em-score'] = es
        if qas_id not in rnk_scores.keys() or rnk_scores[qas_id] < ts:
            rnk_scores[qas_id] = ts
        else:
            continue
        # if example['question_id'] not in bst:
        #     continue
        # if qas_id not in f1_scores.keys():
        #     f1_scores[qas_id] = 0
        #     exact_scores[qas_id] = 0
        # # print("original", prediction)
        # for ex in examples:
        #     if ex['pred'] == "":
        #         continue
        #     if ex['question_id'].split('_')[0] == qas_id:
        #         if prediction == "" or len(ex['pred']) < len(prediction):
        #             prediction = ex['pred']
        # # print("best", prediction)

        exact_scores[qas_id] = es
        f1_scores[qas_id] =  f1s

    qid_list = reference['reference'].keys()
    total = len(qid_list)
    
    table_list = reference['table']
    passage_list = reference['passage']
    #TODO: What is table exact, passage exact scores ?

    ### For dev set, we know where the gold answer is coming from so we can compute table exact and passage exact
    print("total scores", len(exact_scores))
    with open('/tmp/dev_top5_scores.json', 'w') as f:
        json.dump(eval_res, f, indent=2)
    return collections.OrderedDict(
        [
            ("table exact", 100.0 * sum([exact_scores[k] for k in table_list]) / len(table_list)),
            ("table f1", 100.0 * sum(f1_scores[k] for k in table_list) / len(table_list)),
            ("passage exact", 100.0 * sum(exact_scores[k] for k in passage_list) / len(passage_list)),
            ("passage f1", 100.0 * sum(f1_scores[k] for k in passage_list) / len(passage_list)),
            ("total exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("total f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ]
    )

assert len(sys.argv) == 3, "you need to input the file"

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

with open(sys.argv[2], 'r') as f:
    ref = json.load(f)

print(get_raw_scores(data, ref))
