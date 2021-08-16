import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
from tqdm import tqdm
from fuzzywuzzy import fuzz
import numpy as np

if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name = 'msmarco-distilbert-base-tas-b'
doc_retriever = SentenceTransformer(model_name)
top_k = 2

def get_top_k_passages(passages,query,top_k, row=None):
    old_passages = passages
    if row is not None:
        row_str = ""
        for k, v in row.items():
            row_str += " " + k + " is " + v + " . "
        passages = [row_str+passage for passage in passages]
    # print(passages)
    corpus_embeddings = doc_retriever.encode(passages, convert_to_tensor=True, show_progress_bar=False)
    # print(corpus_embeddings)
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = doc_retriever.encode([query], convert_to_tensor=True)
    # print(query_embeddings)
    query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=util.dot_score)
    hits = hits[0]
    #print(hits)
    relevant_sents =[]
    for hit in hits:
        #print("\t{:.3f}\t{}".format(hit['score'], passage[hit['corpus_id']]))
        relevant_sents.append(old_passages[hit['corpus_id']])
    return relevant_sents

import json
from utils.table_utils import fetch_table
import sys

p = json.load(open('/mnt/infonas/data/yashgupta/data/row_selector_output/qid_logits_bert_large_dev.json'))
def get_max_score_row(q_id):
    return np.array(p[q_id]).argsort()[-5:]

def preprocess_instance(d,test=False):
    p_d = {}
    p_d['question'] = d['question']
    p_d['question_id'] = d['question_id']
    if not test:
        p_d['answer-text'] = d['answer-text']
    p_d['table'] = fetch_table(d['table_id'])
    p_d['table_id'] = d['table_id']
    return p_d
    

def preprocess_data(data_path,test):
    data = json.load(open(data_path))
    processed_data = []
    num = 0
    den = 0
    for d in tqdm(data):
        # if d['label'] != 1:
        #     continue
        pi = preprocess_instance(d,test=test)
        question_str = pi['question']
        if not test:
            answer_text = pi['answer-text']
        q_id = pi['question_id']
        table_id = pi['table_id']
        
        #question = [d['question']]
        #q_input = self.tokenizer(question,add_special_tokens=True, truncation=True,padding=True, return_tensors='pt', max_length = self.max_seq_len)
        header = pi['table']['header']
        cri = get_max_score_row(q_id) #d['correct_row_index']
        rows =  [pi['table']['data'][cr] for cr in cri]
        table_rows = []
        table_row_passages = []
        table_row_passages_new = []
        nm, xl = 0, []
        for r in rows:
            one_row = {}
            passage = ""
            passages = []
            for r_v,h in zip(r,header):
                one_row[h] = r_v["cell_value"]
                passage+= " ".join(r_v['passages'])
                passages += r_v['passages']
            # l1, l2 = sorted([v.strip().lower() for v in one_row.values()]), sorted([v.strip().lower() for v in d['table_row'].values()])
            # xl.append([l1, l2])# print(l1)
            # # print(l2)
            # if  l1 != l2:
            #     continue
            # nm += 1
            table_rows.append(one_row)
            table_row_passages.append(passage)
            table_row_passages_new.append(passages)
        # if (nm==0):
        #     print(xl)
        for r,pr,npr in zip(table_rows,table_row_passages,table_row_passages_new):
            npi={}
            npi['question_id'] = q_id
            npi['question'] = question_str
            npi['table_id'] = table_id
            npi['table_row'] = r
            row_values = [v.lower() for v in r.values()]
            npi['table_passage_row_old'] = pr
            npi['table_passage_row'] = pr

            if (len(npr)==0):
                npr = ""
            elif (len(npr)==1):
                npr = npr[0]
            else:
                npr = " ".join(get_top_k_passages(npr, question_str, 100, r))
            
            npi['table_passage_row'] = npr
            
            if not test:
                npi['answer-text'] = answer_text
                # if answer_text.lower() in pr.lower() or answer_text.lower() in row_values:
                npi['label'] =1

                if answer_text.lower() in npr.lower() or answer_text.lower() in row_values:
                    npi['label_new'] = 1
                else:
                    npi['label_new'] = 0
                
                if (npi['label_new']!=npi['label']):
                    num += 1
                den += 1

                # else:
                #     npi['label'] = 1
            

            processed_data.append(npi)
    print("total", den, "changed", num, len(processed_data))
    return processed_data


if __name__ == "__main__":
    rel_data_path = sys.argv[1]
    processed_data_path = sys.argv[2]
    processed_data = preprocess_data(rel_data_path,False)

    json.dump(processed_data,open(processed_data_path,"w"),indent=4)
