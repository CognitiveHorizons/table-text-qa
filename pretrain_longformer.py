import os
import argparse
from pathlib import Path
from transformers import LongformerTokenizer
from tokenizers.processors import BertProcessing
from transformers import LongformerForMaskedLM,LongformerConfig
from transformers import LineByLineTextDataset
#from transformers import DataCollatorForLanguageModeling
from datasets.tab_text_data_collator import LFDatacollatorForMLM
from transformers import Trainer, TrainingArguments
import torch

def main(args):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    tokenizer.add_tokens(["[valsep]","[colsep]","[parasep]"],special_tokens=True)
    tokens = tokenizer.encode("Rank [valsep] 1 [colsep] Player [valsep] Emmitt Smith [parasep] Emmitt James Smith III ( born May 15 , 1969 ) is an American former professional football player who was a running back for fifteen seasons in the National Football League ( NFL ) during the 1990s and 2000s , primarily with the Dallas Cowboys . A three-time Super Bowl champion with the Cowboys.")
    #config = LongformerConfig.from_pretrained('longformer-base-4096/') 

    model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096',gradient_checkpointing=True)

    model.resize_token_embeddings(len(tokenizer))
    print(model.num_parameters())

    # We need to save this dataset since it's a bit slow to build each time
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_data_path,
        block_size=args.max_seq_len,
    )

    data_collator = LFDatacollatorForMLM(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    #data_collator = DataCollatorForLanguageModeling(
    #    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    #)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_path,
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=16,
        save_steps=100,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        
    )

    trainer.train()

    trainer.save_model(args.pretrained_model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="./data/lf_pretrain/train.txt",
                    help="Path to processed train data json")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/lf_pretrained_train_2048_sp_token/",
                    help="Path to store training checkpoint")
    parser.add_argument("--pretrained_model_save_path", type=str, default="./pre-trained-longformer_train_2048_sp_token",
                    help="Path to save the pre-trained model")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                    help="Max Sequence length")
    args = parser.parse_args()
    print(args)


    main(args)