import json
import sys
from tqdm import tqdm

sep_token = "[SEP]"
val_sep = "[valsep]"
q_r_sep = "[unused4]"
col_sep = "[colsep]"
p_r_sep = "[parasep]"

# sep_token = "[SEP]"
# val_sep = "[unused3]"
# q_r_sep = "[unused4]"
# col_sep = "[unused5]"
# p_r_sep = "[unused6]"


def linearize(row):
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" is "+str(r)+" . "
        # row_str+=str(c)+" "+str(r)+" "
    return row_str


def linearize_and_add_special_tokens(row):
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" "+val_sep+" "+str(r)+" "+col_sep+" "
    return row_str


def preprocess_data_for_longformer(data_path,in_lf_out_file,sp_token, test=False):
    data = json.load(open(data_path))
    #print(len(data))
    label_1_data = []
    prev_qid = ""
    i=1
    no_found = 0
    found_set = set([])
    for d in tqdm(data):
        new_data = {}
        if test or d['label'] == 1: # or d['match_score'] != '-INF':
            if not test:
                orig_answer = d['answer-text']
                new_data['answer-text'] = orig_answer
            if sp_token=="yes":
                context = linearize_and_add_special_tokens(d['table_row']) +" "+ p_r_sep+ " " +d['table_passage_row']
                # old_context = linearize_and_add_special_tokens(d['table_row']) +" "+ p_r_sep+ " " +d['table_passage_row_old']
            else:
                context = linearize(d['table_row']) + d['table_passage_row']
                # old_context = linearize(d['table_row']) + d['table_passage_row_old']
            new_data['context'] = context
            new_data['title'] = d['table_id'].replace("_"," ")
            new_data['question'] = d['question']
            if prev_qid ==d['question_id']:
                new_qid = d['question_id']+"_"+str(i)
                prev_qid = d['question_id']
                i+=1
            else:
                i=1
                new_qid = d['question_id']+"_"+str(0)
                prev_qid = d['question_id']
            new_data['question_id'] = new_qid
            if not test:
                # start = (" ".join(context.split()[:510])).lower().find(orig_answer.lower())
                start = context.lower().find(orig_answer.lower())
                # if start == -1:
                #     context = old_context
                #     start = context.lower().find(orig_answer.lower())
                #     new_data['context'] = context
                    
                if start == -1:
                    # print(context, orig_answer)
                    no_found += 1
                    # import pdb
                    # pdb.set_trace()
                    answer = "NOT FOUND"
                    new_data['is_impossible'] = True
                else:
                    new_data['is_impossible'] = False
                    found_set.add(d['question_id'])
                    start = context.lower().find(orig_answer.lower())
                    assert(start!=-1)
                    while context[start].lower() != orig_answer[0].lower():
                        start -= 1
                    answer = context[start:start+len(orig_answer)]
                new_data['answers'] = [{'answer_start': start, 'text': answer}]
            label_1_data.append(new_data)

    print("total", len(label_1_data), "answer not found in", no_found, "found in", len(found_set))
    json.dump(label_1_data,open(in_lf_out_file,"w"), indent=4)


if __name__ == "__main__":
    print("creating dataset for longformer")
    final_pred_file = sys.argv[1]
    in_lf_out_file = sys.argv[2]
    add_special_token = sys.argv[3]
    preprocess_data_for_longformer(final_pred_file,in_lf_out_file,add_special_token, test=False)