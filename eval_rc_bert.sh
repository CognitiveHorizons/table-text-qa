export type=bert
export CKPT_PATH=/mnt/infonas/data/yashgupta/save/rc_bert_large/2021_08_05_12_18_30/checkpoint-epoch2
export split=test
export k=1
python3 train_rc_longformer.py --model_type ${type} \
                               --model_name_or_path $CKPT_PATH/ \
                              --do_stage3 --do_lower_case \
                              --train_file $DATA/data/processed_data/for_$type/train_processed_new_bert.json \
                              --predict_file /mnt/infonas/data/yashgupta/data/pl_processed_data/vk/pl_min/test_no_group.json \
                              --per_gpu_train_batch_size 4 \
                              --per_gpu_eval_batch_size 32 \
                              --max_seq_length 512 \
                              --doc_stride 128 \
                              --threads 8 \
                              --output_dir $CKPT_PATH/ \
                              --pred_ans_file $CKPT_PATH/${split}_top${k}_predictions.json \
                              --num_train_epochs 6 \
                              --prefix ${split}_top${k} \
                              # --version_2_with_negative \
                              # csarron/bert-base-uncased-squad-v1
                            #   bert-large-uncased-whole-word-masking-finetuned-squad
                            #   --model_name_or_path stage3_1024_bs8_44/2021_06_01_16_02_13/checkpoint-epoch2/ 

mv /tmp/nbest_predictions_${split}_top${k}.json $CKPT_PATH/nbest_predictions_${split}_top${k}.json