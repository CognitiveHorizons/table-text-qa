import json
import sys

sep_token = "[SEP]"
val_sep = "[unused3]"
q_r_sep = "[unused4]"
col_sep = "[unused5]"
p_r_sep = "[unused6]"


def linearize(row):
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" "+str(r)+" "
    return row_str


def linearize_and_add_special_tokens(row):
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" "+val_sep+" "+str(r)+" "+col_sep+" "
    return row_str


def preprocess_data_for_longformer(data_path,in_lf_out_file,sp_token):
    data = json.load(open(data_path))
    #print(len(data))
    label_1_data = []
    prev_qid = ""
    i=1
    for d in data:
        new_data = {}
        if d['label'] == 1:
            orig_answer = d['answer-text']
            new_data['answer-text'] = orig_answer
            if sp_token=="yes":
                context = linearize_and_add_special_tokens(d['table_row']) +" "+ p_r_sep+ " " +d['table_passage_row']
            else:
                context = linearize(d['table_row']) + d['table_passage_row']
            new_data['context'] = context
            new_data['title'] = d['table_id'].replace("_"," ")
            new_data['question'] = d['question']
            if prev_qid ==d['question_id']:
                new_qid = d['question_id']+"_"+str(i)
                prev_qid ==d['question_id']
                i+=1
            else:
                i=1
                new_qid = d['question_id']
                prev_qid = d['question_id']
            new_data['question_id'] = new_qid
            start = context.lower().find(orig_answer.lower())
            if start == -1:
                import pdb
                pdb.set_trace()
            while context[start].lower() != orig_answer[0].lower():
                start -= 1
            answer = context[start:start+len(orig_answer)]
            new_data['answers'] = [{'answer_start': start, 'text': answer}]
            label_1_data.append(new_data)

    print(len(label_1_data))
    json.dump(label_1_data,open(in_lf_out_file,"w"))


if __name__ == "__main__":
    print("creating dataset for longformer")
    final_pred_file = sys.argv[1]
    in_lf_out_file = sys.argv[2]
    add_special_token = sys.argv[3]
    preprocess_data_for_longformer(final_pred_file,in_lf_out_file,add_special_token)