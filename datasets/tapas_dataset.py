import torch
import pandas as pd
class TapasTableDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer,ret_labels):
        self.data = data
        self.tokenizer = tokenizer
        self.ret_labels = ret_labels
        self.tables = []
        for d in data:
            self.tables.append(d['table_row'])
        self.tables = pd.DataFrame.from_dict(self.tables,orient='columns')

    def __getitem__(self, idx):
        item = self.data[idx]
        question = [item['question']]
        table = self.tables.iloc[[idx]]
        print(table)
        #table = pd.DataFrame.from_dict(table)
        label = item['label']

        
        encoding = self.tokenizer(table=table,
                                   queries=question,
                                   padding="max_length",
                                   return_tensors="pt")
         # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
         # add the float_answer which is also required (weak supervision for aggregation case)
        return encoding,torch.tensor(label)

    def __len__(self):
        return len(self.data)