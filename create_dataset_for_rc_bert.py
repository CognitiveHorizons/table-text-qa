import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
from tqdm import tqdm


if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name2 = "msmarco-MiniLM-L-6-v3"
doc_retriever2 = SentenceTransformer(model_name2)
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
        pi = preprocess_instance(d,test=test)
        question_str = pi['question']
        query_embeddings = util.normalize_embeddings(doc_retriever2.encode([question_str], convert_to_tensor=True))
        if not test:
            answer_text = pi['answer-text']
        q_id = pi['question_id']
        table_id = pi['table_id']
        #question = [d['question']]
        #q_input = self.tokenizer(question,add_special_tokens=True, truncation=True,padding=True, return_tensors='pt', max_length = self.max_seq_len)
        header = pi['table']['header']
        rows = pi['table']['data']
        table_rows = []
        table_row_passages = []
        table_row_passages_new = []
        for r in rows:
            one_row = {}
            passage = ""
            passages = []
            for r_v,h in zip(r,header):
                one_row[h] = r_v["cell_value"]
                passage+= " ".join(r_v['passages'])
                passages += r_v['passages']
            table_rows.append(one_row)
            table_row_passages.append(passage)
            table_row_passages_new.append(passages)
        
        corpus = []
        for r,pr in zip(table_rows,table_row_passages):   
            row_str = ""
            for k, v in r.items():
                row_str += " " + k + " is " + v + " . "
            corpus.append(row_str + pr)
        corpus_embeddings = util.normalize_embeddings(doc_retriever2.encode(corpus, convert_to_tensor=True))
        hits_rows = util.semantic_search(query_embeddings.to('cuda'), corpus_embeddings.to('cuda'), top_k=3, score_function=util.dot_score)[0]
        scores = ['-INF']*len(rows)
        for hr in hits_rows:
            scores[hr['corpus_id']] = hr['score']
        dsk = 0
        for r,pr,npr in zip(table_rows,table_row_passages,table_row_passages_new):
            npi={}
            npi['match_score'] = scores[dsk]
            dsk += 1
            npi['question_id'] = q_id
            npi['question'] = question_str
            npi['table_id'] = table_id
            npi['table_row'] = r
            row_values = [v.lower() for v in r.values()]
            npi['table_passage_row_old'] = pr
            npi['table_passage_row'] = pr
            if not test:
                npi['answer-text'] = answer_text
                if answer_text.lower() in pr.lower() or answer_text.lower() in row_values:
                    npi['label'] =1

                    if (len(npr)==0):
                        npr = ""
                    elif (len(npr)==1):
                        npr = npr[0]
                    else:
                        npr = " ".join(get_top_k_passages(npr, question_str, 100, r))
                    
                    npi['table_passage_row'] = npr

                    if answer_text.lower() in npr.lower() or answer_text.lower() in row_values:
                        npi['label_new'] = 1
                    else:
                        npi['label_new'] = 0
                    
                    if (npi['label_new']!=npi['label']):
                        num += 1
                    den += 1

                else:
                    npi['label'] =0

            processed_data.append(npi)
    print("total", den, "changed", num)
    return processed_data


if __name__ == "__main__":
    rel_data_path = sys.argv[1]
    processed_data_path = sys.argv[2]
    processed_data = preprocess_data(rel_data_path,False)

    json.dump(processed_data,open(processed_data_path,"w"),indent=4)
