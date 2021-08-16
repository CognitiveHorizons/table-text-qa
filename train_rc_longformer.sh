python3 train_rc_longformer.py --model_type longformer \
                              --do_train --do_stage3 --do_lower_case \
                              --train_file $DATA/data/processed_data/for_longformer/train_processed_longformer_special_tokens.json \
                              --predict_file $DATA/data/processed_data/for_longformer/dev_processed_longformer_special_tokens.json \
                              --per_gpu_train_batch_size 4 \
                              --max_seq_length 1024 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir $DATA/save/rc_longformer_special_tokens/ \
                              --pred_ans_file $DATA/save/rc_longformer_special_tokens/ \
                              --num_train_epochs 6 \
                            #   --model_name_or_path stage3_1024_bs8_44/2021_06_01_16_02_13/checkpoint-epoch2/ 