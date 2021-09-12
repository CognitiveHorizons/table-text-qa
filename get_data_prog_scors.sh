export type=bert
export CKPT_PATH=/mnt/infonas/data/yashgupta/save/bert_large_top_new_bs_sa/2021_09_07_15_07_03/checkpoint-epoch1
python3 get_data_prog_scors.py --model_type ${type} \
                               --model_name_or_path $CKPT_PATH/ \
                              --do_train --do_lower_case \
                              --train_file /mnt/infonas/data/yashgupta/data/new_bs_processed_data/train_processed_top_row_no_group_ma.json \
                              --predict_file /mnt/infonas/data/yashgupta/data/new_bs_processed_data/train_processed_top_row_no_group_ma.json \
                              --per_gpu_train_batch_size 1 \
                              --per_gpu_eval_batch_size 1 \
                              --max_seq_length 512 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir /tmp/ \
                              --pred_ans_file $CKPT_PATH/train_top_all_ma_scors.json \
                              --num_train_epochs 1 \
                              --prefix train_all_ma \
                              # --version_2_with_negative \
                              # csarron/bert-base-uncased-squad-v1
                            #   bert-large-uncased-whole-word-masking-finetuned-squad
                            #   --model_name_or_path stage3_1024_bs8_44/2021_06_01_16_02_13/checkpoint-epoch2/ 