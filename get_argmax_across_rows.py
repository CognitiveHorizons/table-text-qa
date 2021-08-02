import json
import numpy as np
import sys


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def post_process_test_pred(pred_file):
    q_id_labels = {}
    prediction = json.load(open(pred_file))
    for qid,pred in prediction.items():
        labels = [0]*len(pred)
        a = np.array(pred)
        #print(a)
        #input()
        argmax = np.argmax(a)
        labels[argmax] = 1
        q_id_labels[qid] = labels
    #print(q_id_labels['00153f694413a536'])
    return q_id_labels
    

def assing_labels(test_file,q_id_labels,to_write_final):
    q_id_table_row = {}
    data = json.load(open(test_file))
    for d in data:
        if d['question_id'] in q_id_table_row:
            q_id_table_row[d['question_id']].append(d)

        else:
            q_id_table_row[d['question_id']] = [d]
    new_dat_list = []
    for q_id, rows in q_id_table_row.items():
        labels = q_id_labels[q_id]
        for d,label in zip(rows,labels):
            d['label'] = label
            new_dat_list.append(d)
    json.dump(new_dat_list,open(to_write_final,"w"))
        





    


if __name__ == "__main__":
    scores_id_file = sys.argv[1]
    to_write_final = sys.argv[2]
    q_id_labels =  post_process_test_pred(scores_id_file)
    assing_labels("data/processed_data/test_processed_new.json",q_id_labels,to_write_final)
