python3 train_rc_longformer.py --model_type bert \
                               --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
                              --do_train --do_lower_case \
                              --train_file /mnt/infonas/data/yashgupta/data/new_bs_processed_data/train_processed_top_row_no_group_sa.json \
                              --predict_file $DATA/data/processed_data/for_bert/dev_processed_new_bert.json \
                              --per_gpu_train_batch_size 8 \
                              --max_seq_length 512 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir $DATA/save/bert_large_top_new_bs_sa/ \
                              --pred_ans_file $DATA/save/bert_large_top_bs_sa/ \
                              --num_train_epochs 5 \
                              # --version_2_with_negative \
                              #bert-large-uncased-whole-word-masking-finetuned-squad
                              # csarron/bert-base-uncased-squad-v1
                            #   bert-large-uncased-whole-word-masking-finetuned-squad
                            #   --model_name_or_path stage3_1024_bs8_44/2021_06_01_16_02_13/checkpoint-epoch2/ 