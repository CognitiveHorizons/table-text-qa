import json

def main():
    data= json.load(open("dev_original_pred_lf_unique_ids.json"))
    new_data = []
    qid_pred ={} 
    original_qid = ""
    prev_qid = ""
    qid = []
    for d in data:
        if "_" in d['question_id']:
            qid.append(d['question_id'].split("_")[0].strip())
        else:
            qid.append(d['question_id'])
        if "_" in d['question_id']:
            curr_qid = d['question_id'].split("_")[0].strip()
            if curr_qid == prev_qid:
                qid_pred[curr_qid].append(d['pred'])
                prev_qid = curr_qid
        else:
            qid_pred[d['question_id']] = [d['pred']]
            prev_qid = d['question_id']
    new_data =[]
    for k,v in qid_pred.items():
        new_dict = {}
        new_dict['question_id'] = k
        new_dict['pred'] = v
        new_data.append(new_dict)
    print(len(new_data))
    print(len(set(qid)))
    json.dump(new_data,open("dev_orig_post_process_pred.json","w"))


if __name__ == "__main__":
    main()    