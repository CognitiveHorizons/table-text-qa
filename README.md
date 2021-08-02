### TABLE +TEXT QA 


### File Details
* create_dataset_tapas.py -> To convert hybridqa dataset into following format:
    {"question_id": "00153f694413a536",
     "question": "What is the middle name of the player with the second most National Football League career rushing yards ?", 
     "table_id": "List_of_National_Football_League_rushing_yards_leaders_0",
      "table_row": {"Rank": "10", "Player": "Tony Dorsett", "Team ( s ) by season": "Dallas Cowboys ( 1977 - 1987 ) Denver Broncos ( 1988 )", "Carries": "2,936", "Yards": "12,739", "Average": "4.3"}, 
      "table_passage_row": "Anthony Drew Tony Dorsett ( born April 7 , 1954 ) is a former American football running back who played professionally in the National Football League ( NFL ) for .... }
* get_argmax_across_rows.py -> This python script takes prediction from the row selector module and applies argmax across rows in a table and labels row most relevant to the question as 1.

* extract_relevant_sentences.py -> This python script uses a sentence transformer to select top-k most relevant sentences for a particular question from the passage connected to a row.

* train_row_retriever_QRS_concat.py -> Main row selector training script, this python script can be used to train a row selector model and test a pre-trained model.

* train_rc_longformer.py -> Final answer/span selector module. This script can be used to train a longformer based RC model and also to test a pre-trained model




### STEP1: To preprocess the released hybridqa dataset 
```python create_dataset_tapas.py data/released_data/train.json data/processed_data/train_processed_new.json```
```python create_dataset_tapas.py data/released_data/dev.json data/processed_data/dev_processed_new.json```

### STEP2: To train row selector model 

```python train_row_retriever_QRS_concat --train_data_path data/processed_data/train_processed_new.json --dev_data_path data/processed_data/dev_processed_new.json  --save_model_path checkpoints/test_run/model.best.bin --train_batch_size 64```


### To test the row selector model
```python train_row_retriever_QRS_concat.py --test --model_path checkpoints/q_r_s_concat/model_qrs_gold_else_random_4gpus_shuffle.best.bin --test_data_path data/processed_data/test_processed_new.json --use_st_out --predict_file predictions/test_model_qrs_gold_else_random_4gpus_shuffle.json --bert_model bert-base-uncased```

### STEP3: To train the final answer/span selector module
```jbsub  -require v100 -cores 4+2 -q x86_24h -mem 128g -out out_stage3_final.txt -err err_stage3_final.txt  python train_rc_longformer.py --train_file dat/processed_data/for_longformer/train_processed_longformer_special_tokens.json --predict_file data/processed_data/for_longformer/dev_processed_longformer_special_tokens.json --do_train```

### To test the RC model run 
``` python train_rc_longformer.py -model_name_or_path stage3_1024_bs8_44/2021_06_01_16_02_13/checkpoint-epoch2/ --do_stage3 --do_lower_case --predict_file data/processed_data/for_longformer/ --per_gpu_train_batch_size 12 --max_seq_length 1024 --doc_stride 128 --threads 8```

