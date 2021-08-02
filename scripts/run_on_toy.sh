train_script=${1:-../train_row_retriever_question_row_concat.py}
train_path=${2:-../data/processed_data/toy_processed_new.json}
dev_path=${3:-../data/processed_data/toy_processed_new.json}
save_model_path=${4:-../checkpoints/test_run/model_toy.best.bin}

python $train_script --train_data_path $train_path --dev_data_path $dev_path --save_model_path $save_model_path --train_batch_size 64