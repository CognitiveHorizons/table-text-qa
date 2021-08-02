from utils.json_util import read_data
from torch.utils.data import WeightedRandomSampler
import argparse
from datasets.dataset import TableQADatasetQRSconcat
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer, 
                        BertModel, get_linear_schedule_with_warmup, 
                        squad_convert_examples_to_features)
from torch.utils.data import DataLoader
from tqdm import tqdm,trange
#from transformers import TapasTokenizer, TapasModel
from models.bert.table_encoder import RowClassifierSC
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel,AdamW,get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import pickle
from transformers import LongformerModel, LongformerTokenizer
from collections import OrderedDict
import json

# import nvidia_smi
# nvidia_smi.nvmlInit()
# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

# Used during test if you want to test the model trained on multiple gpus on a single gpu.
def clean_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict




class Training:
    '''Training class takes model data loader schedular and optimizer device 
    '''
    def __init__(self,model,data_loader,optimizer,scheduler,device,criterion):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
    def train(self):
        self.model.train()
        losses =0.0
        correct_predictions = 0
        predictions_labels = []
        true_labels = []
        for q_r_input,labels in tqdm(self.data_loader,total = len(self.data_loader),position=0, leave=True):
            true_labels += labels.numpy().flatten().tolist()
            labels = labels.to(self.device)
            q_r_input = {k:v.type(torch.long).to(self.device) for k,v in q_r_input.items()}
            outputs = self.model(q_r_input,labels)
            #logits = outputs.logits
            logits =outputs.logits
            loss = outputs.loss.mean()
            #loss = self.criterion(logits,labels) # For models other than sequence classification models
            losses+=loss.item()
            loss.backward()
            self.optimizer.step() 
            self.scheduler.step()
            self.optimizer.zero_grad()
            rounded_preds = torch.argmax(logits,axis=1)
            predictions_labels += rounded_preds.detach().cpu().numpy().tolist()
        avg_epoch_loss = losses / len(self.data_loader)
        return true_labels, predictions_labels,avg_epoch_loss

class Validation:
    '''
    Validation class to run evaluation on dev set for model selection
    '''
    def __init__(self,model,data_loader,device,criterion):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.criterion  = criterion
    def evaluate(self):
        self.model.eval()
        losses = 0
        correct_predictions = 0
        predictions_labels = []
        true_labels = []
        for q_r_input,labels in tqdm(self.data_loader,total = len(self.data_loader),position=0, leave=True):
            true_labels += labels.numpy().flatten().tolist()
            labels = labels.to(self.device)
            q_r_input = {k:v.type(torch.long).to(self.device) for k,v in q_r_input.items()}
            with torch.no_grad():
                outputs = self.model(q_r_input,labels)
                logits =outputs.logits
                loss = outputs.loss.mean()
                #loss = self.criterion(logits,labels) # This line is used when criterion is set
                losses+=loss.item()
                rounded_preds = torch.argmax(logits,axis=1)
                predictions_labels += rounded_preds.detach().cpu().numpy().tolist()
        avg_epoch_loss = losses / len(self.data_loader)
        return true_labels, predictions_labels,avg_epoch_loss

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model",type=str, default="bert-base-uncased")
    parser.add_argument("--model_type",type=str, default="bert")
    parser.add_argument("--train_data_path", type=str, default="data/processed_data/train_processed_new.json",
                    help="Path to processed train data json")
    parser.add_argument("--dev_data_path", type=str, default="data/processed_data/dev_processed_new.json",
                    help="Path to processed train data json")
    parser.add_argument("--test_data_path", type=str, default="data/processed_data/dev_processed_new.json",
                    help="Path to processed train data json")
    parser.add_argument("--test", help="Test the pre-trained model",
                    action="store_true")
    parser.add_argument("--use_st_out", help="Use the sentence transformer output instead of gold sentences",
                    action="store_true")
    parser.add_argument("--model_path", type=str, default="checkpoints/simple_model/model.best.bin",
                    help="Path to pre-trained model")
    parser.add_argument("--save_model_path", type=str, default="checkpoints/simple_model/imbalanced_class_sampler_gold_sent_row_question_concat_model.best.bin",
                    help="Path to save trained model")
    parser.add_argument("--predict_file", type=str, default="predictions/dev_prediction.json",
                    help="path to save prediction file")
    parser.add_argument("--train_batch_size", type=int, default=32,
                    help="Training batch size")
    parser.add_argument("--dev_batch_size", type=int, default=64,
                    help="Evaluation batch size")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=1024)

    parser.add_argument("--num_train_epochs", type=int, default=3,
                    help="Number of traning epochs")

    args = parser.parse_args()
    print("Printing arguments starts here ==========================")
    print(args)
    print("Printing arguments end here ==============================")

    #model details
    device = torch.device("cuda")

    #Model instatiation
    #model = RowClassifierQRSLongformer()
    model = RowClassifierSC(args)
    # if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #         model = nn.DataParallel(model)

    
    if args.test:
        state_dict = torch.load(args.model_path)
        model = RowClassifierSC(args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        model.load_state_dict(state_dict,strict=True)
        print("Model Loaded")
        model.to(device)
        model.eval()
        predictions_labels = []
        scores_list = []
        question_ids_list = []
        if args.model_type == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.model_type == "longformer":
            bert_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        test_data = read_data(args.test_data_path)
        test_dataset = TableQADatasetQRSconcat(test_data,args.max_seq_length,bert_tokenizer,use_st_out=args.use_st_out,ret_labels=False)
        test_data_loader = DataLoader(test_dataset, batch_size=args.dev_batch_size, batch_sampler=None, num_workers=1, shuffle=False, pin_memory=True)
        for question_ids,q_r_input in tqdm(test_data_loader,total = len(test_data_loader),position=0, leave=True):
            q_r_input = {k:v.type(torch.long).to(device) for k,v in q_r_input.items()}
            
            with torch.no_grad():
                outputs = model(q_r_input)
                #rounded_preds = torch.argmax(probs,axis=1)
                #predictions_labels += rounded_preds.detach().cpu().numpy().tolist()
                probs = outputs.logits
                #print(probs)
                scores = probs[:,1]
                scores = scores.detach().cpu().numpy().tolist() 
                scores_list+=scores
                question_ids_list+=question_ids
        
        q_id_scores_list = {}
        prev_qid = ""
        for q_id, score in zip(question_ids_list,scores_list):
            if q_id==prev_qid:
                q_id_scores_list[q_id].append(score)
            else:
                q_id_scores_list[q_id] = [score]
                prev_qid=q_id
            
        #print(q_id_scores_list['00153f694413a536'])
        json.dump(q_id_scores_list,open(args.predict_file,"w"))           

    else:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        model.to(device)

        if args.model_type == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.model_type == "longformer":
            bert_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        
        print('Model Loaded')
        #Load train and dev data from raw files
        train_data = read_data(args.train_data_path)
        dev_data = read_data(args.dev_data_path)
        use_st_out = False
        if use_st_out:
            use_st_out=True
        #Load dataset
        train_dataset = TableQADatasetQRSconcat(train_data, args.max_seq_length,bert_tokenizer, use_st_out=use_st_out,ret_labels=True)
        #pickle.dump(train_dataset,open("data/pickles/roberta/train_dataset_ret_labels.pkl","wb"),protocol=4)
        dev_dataset = TableQADatasetQRSconcat(dev_data, args.max_seq_length,bert_tokenizer,use_st_out=use_st_out,ret_labels=True)
        #pickle.dump(dev_dataset,open("data/pickles/roberta/dev_dataset_ret_labels.pkl","wb"),protocol=4)
       # print("pickling is done")
        print('Raw Dataset Loaded')
        # train_dataset = pickle.load(open("data/pickles/roberta/train_dataset.pkl","rb"))
        # dev_dataset = pickle.load(open("data/pickles/roberta/dev_dataset.pkl","rb"))
        # print("dataset loaded from pickle")
        #loss function
        criterion = nn.BCEWithLogitsLoss()
        total_steps = len(train_dataset)*args.num_train_epochs

        #Optimizer to update model parameters
        optimizer = AdamW(model.parameters(),lr = 5e-5,eps = 1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0.1*total_steps,num_training_steps= total_steps)

        positive_count = train_dataset.positive_count
        negative_count = train_dataset.negative_count
        class_counts = [positive_count,negative_count]
        num_samples = positive_count+negative_count
        labels = train_dataset.labels_list
        class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

        #data loader to iterate over full dataset
        train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,shuffle=True, batch_sampler=None, pin_memory=True)
        dev_data_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, batch_sampler=None, shuffle=False, pin_memory=True)

        #train_iterator = trange(0, int(num_train_epochs), desc="Epoch")
        print('Dataset loader created')

        trainer = Training(data_loader = train_data_loader,
                    model = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    device = device,criterion=criterion)

        validator = Validation(data_loader = dev_data_loader,
                        model = model,
                        device = device,criterion=criterion
                        )


        best_accuracy = 0
        all_loss = {'train_loss':[], 'val_loss':[]}
        all_acc = {'train_acc':[], 'val_acc':[]}
        for epoch in tqdm(range(args.num_train_epochs),position=0, leave=True):
            print('Training now ')
            train_labels,train_predictions,training_loss = trainer.train()
            #print(train_labels)
            #print(train_predictions)
            
            training_accuracy = accuracy_score(train_labels,train_predictions)
            print(f'Training loss for epoch {epoch+1}: {training_loss}' )
            print(f'Training accuracy for epoch {epoch+1}: {training_accuracy}')

            validation_labels,validation_predictions,validation_loss = validator.evaluate()
            validation_accuracy = accuracy_score(validation_labels,validation_predictions)
            print(f'Validation loss for epoch {epoch+1}: {validation_loss}' )

            if validation_accuracy > best_accuracy:
                torch.save(model.state_dict(),args.save_model_path)
                best_accuracy = validation_accuracy

            # Store the loss value for plotting the learning curve.
            all_loss['train_loss'].append(training_loss)
            all_loss['val_loss'].append(validation_loss)
            all_acc['train_acc'].append(training_accuracy)
            all_acc['val_acc'].append(validation_accuracy)

        print(f'Best Accuracy achieved: {best_accuracy}')
        # plt.title('Loss Curves')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # sns.lineplot(x= [x for x in range(1,epochs+1)],y = all_loss['train_loss'])
        # sns.lineplot(x= [x for x in range(1,epochs+1)],y = all_loss['val_loss'])
        # plt.legend(['Training','Validation'])
        # plt.show()

        # plt.title('Accuracy Curves')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # sns.lineplot(x= [x for x in range(1,epochs+1)],y = all_acc['train_acc'])
        # sns.lineplot(x= [x for x in range(1,epochs+1)],y = all_acc['val_acc'])
        # plt.legend(['Training','Validation'])
        # plt.show()

        #validator.evaluate()

if __name__ == "__main__":
    main()
    #train("data/processed_data/for_tapas/dev_processed_new.json",5)