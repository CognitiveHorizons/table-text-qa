export type=bert
export CKPT_PATH=$DATA/save/bert_large_logits_top/2021_08_13_14_46_35/checkpoint-epoch3
python3 train_rc_longformer.py --model_type ${type} \
                               --model_name_or_path $CKPT_PATH/ \
                              --do_stage3 --do_lower_case \
                              --train_file $DATA/data/processed_data/for_$type/train_processed_new_bert.json \
                              --predict_file /mnt/infonas/data/yashgupta/data/row_selector_output/dev_top5_processed.json \
                              --per_gpu_train_batch_size 4 \
                              --per_gpu_eval_batch_size 24 \
                              --max_seq_length 512 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir $CKPT_PATH/ \
                              --pred_ans_file $CKPT_PATH/dev_top5_predictions.json \
                              --num_train_epochs 6 \
                              --prefix dev_top5 \
                              # --version_2_with_negative \
                              # csarron/bert-base-uncased-squad-v1
                            #   bert-large-uncased-whole-word-masking-finetuned-squad
                            #   --model_name_or_path stage3_1024_bs8_44/2021_06_01_16_02_13/checkpoint-epoch2/ 