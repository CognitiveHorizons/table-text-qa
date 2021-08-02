train_script=${1:-../train_row_retriever_question_row_concat.py}
train_path=${2:-../data/processed_data/train_processed_new.json}
dev_path=${3:-../data/processed_data/dev_processed_new.json}
save_model_path=${4:-../checkpoints/q_r_concat/shuffle_true_best_model.bin}

deepspeed $train_script --train_data_path $train_path --dev_data_path $dev_path --save_model_path $save_model_path --num_train_epochs 3