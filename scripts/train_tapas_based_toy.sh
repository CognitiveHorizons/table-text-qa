train_script=${1:-../tapas_based_row_classifier.py}
train_path=${2:-../data/processed_data/toy_processed_new.json}
dev_path=${3:-../data/processed_data/toy_processed_new.json}
save_model_path=${4:-../checkpoints/test_run/tapas_based_toy_best_model.bin}

deepspeed $train_script --train_data_path $train_path --dev_data_path $dev_path --save_model_path $save_model_path