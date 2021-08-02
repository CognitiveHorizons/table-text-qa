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
    for d in data:
        pi = preprocess_instance(d,test=test)
        question_str = pi['question']
        if not test:
            answer_text = pi['answer-text']
        q_id = pi['question_id']
        table_id = pi['table_id']
        
        #question = [d['question']]
        #q_input = self.tokenizer(question,add_special_tokens=True, truncation=True,padding=True, return_tensors='pt', max_length = self.max_seq_len)
        header = pi['table']['header']
        rows = pi['table']['data']
        table_rows = []
        table_row_passages =[]
        for r in rows:
            one_row = {}
            passage = ""
            for r_v,h in zip(r,header):
                one_row[h] = r_v["cell_value"]
                passage+= " ".join(r_v['passages'])
            table_rows.append(one_row)
            table_row_passages.append(passage)
        for r,pr in zip(table_rows,table_row_passages):
            npi={}
            npi['question_id'] = q_id
            npi['question'] = question_str
            npi['table_id'] = table_id
            npi['table_row'] = r
            row_values = [v.lower() for v in r.values()]
            npi['table_passage_row'] = pr
            if not test:
                npi['answer-text'] = answer_text
                if answer_text.lower() in pr.lower() or answer_text.lower() in row_values:
                    npi['label'] =1
                else:
                    npi['label'] =0
            

            processed_data.append(npi)
    return processed_data


if __name__ == "__main__":
    rel_data_path = sys.argv[1]
    processed_data_path = sys.argv[2]
    processed_data = preprocess_data(rel_data_path,True)

    json.dump(processed_data,open(processed_data_path,"w"))

