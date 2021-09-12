import json
import numpy as np
import sys
import tqdm

p = json.load(open('/mnt/infonas/data/yashgupta/data/pl_rs_logits/predictions_on_dev_Blarge_no_group.json'))
def get_rs_score(q_id, rank):
    return np.sort(np.array(p[q_id]))[-5:][rank]

q = json.load(open('/mnt/infonas/data/yashgupta/data/pl_processed_data/vk/no_group/pred_dev_top5_epoch1.json_nbest_predictions.json'))
def get_rc_score(q_id, rank):
    nb = q[q_id+"_"+str(rank)][0]
    return nb['start_logit'] + nb['end_logit']

assert len(sys.argv) == 3, "you need to input the file"

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

data_rr = []
data_new = []
mxs = {}
for example in tqdm.tqdm(data):
    qas_id = example['question_id'].split('_')[0]
    qas_rank = int(example['question_id'].split('_')[1])
    rcs = get_rc_score(qas_id, qas_rank)
    rss = get_rs_score(qas_id, qas_rank)
    ts = 3.2*rss + rcs
    example['rerank_score'] = ts
    data_new.append(example)
    if qas_id not in mxs.keys() or mxs[qas_id][0] < ts:
        mxs[qas_id] = (ts, example)

import copy
for k, v in tqdm.tqdm(mxs.items()):
    vn = copy.deepcopy(v[1])
    n = vn['question_id'].split('_')[1]
    if n != '4':
        print(vn['question_id'])
    vn['question_id'] = vn['question_id'].split('_')[0]
    data_rr.append(vn)

with open(sys.argv[2], 'w') as f:
    json.dump(data_rr, f, indent=2)

with open(sys.argv[1], 'w') as f:
    json.dump(data_new, f, indent=2)