import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset

class BankDateset(Dataset):
    def __init__(self, tokenizer, max_len, mode, file_path=None,dataframe=None):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.label_dict = self.get_labels()
        self.label_number = len(self.label_dict)
        self.data_set = self.convert_data_to_ids(file_path,dataframe)

    def _read(self, file_name,dataframe):
        '''
        read data from [file_name].csv file and convert it to a dict
        '''
        if file_name is None:
            df = dataframe
        else:
            df = pd.read_csv(file_name)
        samples = []
        for idx, row in df.iterrows():
            text = row['text']
            if type(text)==float:
                print(text)
                continue
            tokens = list(row['text'])
            if self.mode =='test':
                tags = []
                class_ = None
            else:
                tags = row['BIO_anno'].split()
                class_ = row['class']
                ids = row['id']
            
            samples.append({"text": text, "tokens": tokens, "labels":tags, "class":class_})
        
        return samples
    
    def get_labels(self):
        '''
        map each label to a numebr
        '''
        label_dic = {}
        label_list = ["B-BANK", "I-BANK", "B-PRODUCT", "I-PRODUCT", 'B-COMMENTS_N'
                      ,"I-COMMENTS_N","B-COMMENTS_ADJ","I-COMMENTS_ADJ", "O"]
        for idx, label in enumerate(label_list):
            label_dic[label] = idx

        return label_dic
    
    def convert_data_to_ids(self, file_path,dataframe):
        '''
        '''
        self.data_set = []
        samples = self._read(file_path,dataframe)
        for sample in tqdm(samples):
            sample = self.convert_sample_to_id_train(sample) if self.mode == 'train' else self.convert_sample_to_id_test(sample)
            self.data_set.append(sample)
        return self.data_set
    
    def convert_sample_to_id_train(self, sample):
        '''
        map token to index by bert tokenizer
        '''
        tokens = sample["tokens"]
        labels = sample["labels"]
        class_ = sample["class"]

        #assert len(tokens) == len(labels), 'the size of tokens does not match the size of labels'
        # map unkown word to [UNK]
        new_tokens = []
        for token in tokens:
            if not len(self.tokenizer.tokenize(token)):
                new_tokens.append('[UNK]')
            else:
                new_tokens.append(token)
        # truncate sentence if it is longer than max_len
        if len(new_tokens) > self.max_len - 2:
            new_tokens = new_tokens[:self.max_len-2]
            labels = labels[:self.max_len-2]
        # add [CLS] and [SEP] for bert setting
        new_tokens = ["[CLS]"] + new_tokens + ["[SEP]"]
        # map tokens to indexs
        input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
        attention_mask = [1]*len(input_ids)
        # label for [CLS] and [SEP]
        labels_ids = [self.label_dict["O"]] + [self.label_dict[i] for i in labels] + [self.label_dict["O"]]
        # pad sentences
        padding_id = self.tokenizer.convert_tokens_to_ids(['PAD'])
        len_ = len(input_ids) - 2
        input_ids = input_ids + padding_id * (self.max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))
        labels_ids = labels_ids + [self.label_dict['O']] * (self.max_len - len(labels_ids))
        
        #
        token_type_ids = [0] * len(input_ids)
        position_ids = list(np.arange(len(input_ids)))
        
        sample["input_ids"] = input_ids
        sample["labels_ids"] = labels_ids
        sample["attention_mask"] = attention_mask
        sample["token_type_ids"] = token_type_ids
        sample["position_ids"] = position_ids
        sample["len"] = len_

        assert len(input_ids) == len(labels_ids)
        assert len(input_ids) == self.max_len

        return sample

    def convert_sample_to_id_test(self, sample):
        tokens = sample["tokens"]

        # map unkown word to [UNK]
        new_tokens = []
        for token in tokens:
            if not len(self.tokenizer.tokenize(token)):
                new_tokens.append('[UNK]')
            else:
                new_tokens.append(token)
        # truncate sentence if it is longer than max_len
        if len(new_tokens) > self.max_len - 2:
            new_tokens = new_tokens[:self.max_len-2]
        # add [CLS] and [SEP] for bert setting
        new_tokens = ["[CLS]"] + new_tokens + ["[SEP]"]
        # map tokens to indexs
        input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
        # TODO: how does attention mask work?
        attention_mask = [1]*len(input_ids)
        # pad sentences
        padding_id = self.tokenizer.convert_tokens_to_ids(['PAD'])
        len_ = len(input_ids) - 2
        input_ids = input_ids + padding_id * (self.max_len - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_len - len(attention_mask))
        
        #
        token_type_ids = [0] * len(input_ids)
        position_ids = list(np.arange(len(input_ids)))
        
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["token_type_ids"] = token_type_ids
        sample["position_ids"] = position_ids
        sample["len"] = len_

        assert len(input_ids) == self.max_len

        return sample

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, index):
        instance = self.data_set[index]
        return instance
    
def collate_func_train(batch_data):
    ''' 
    design data output form
    '''
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, token_type_list,labels_ids_list = [], [], [], []
    class_list, position_ids_list, tokens_list, len_list,text_list = [], [], [], [], []

    for instance in batch_data:
        input_ids_list.append(instance["input_ids"])
        attention_mask_list.append(instance["attention_mask"])
        token_type_list.append(instance["token_type_ids"])
        position_ids_list.append(instance["position_ids"])
        tokens_list.append(instance["tokens"])
        len_list.append(instance["len"])
        labels_ids_list.append(instance["labels_ids"])
        class_list.append(instance["class"])
        text_list.append(instance["text"])


    return {"input_ids":torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask":torch.tensor(attention_mask_list, dtype=torch.long),
            "token_type_ids":torch.tensor(token_type_list, dtype=torch.long),
            "position_ids":torch.tensor(position_ids_list, dtype=torch.long),
            "tokens":tokens_list,
            "len":len_list,
            "labels_ids":torch.tensor(labels_ids_list, dtype=torch.long),
            "class":torch.tensor(class_list, dtype = torch.long),
            'text':text_list
            }

def collate_func_test(batch_data):
    ''' 
    design data output form
    '''
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, token_type_list = [], [], []
    position_ids_list, tokens_list, len_list,text_list = [], [], [], []

    for instance in batch_data:
        input_ids_list.append(instance["input_ids"])
        attention_mask_list.append(instance["attention_mask"])
        token_type_list.append(instance["token_type_ids"])
        position_ids_list.append(instance["position_ids"])
        tokens_list.append(instance["tokens"])
        len_list.append(instance["len"])
        text_list.append(instance["text"])

    return {"input_ids":torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask":torch.tensor(attention_mask_list, dtype=torch.long),
            "token_type_ids":torch.tensor(token_type_list, dtype=torch.long),
            "position_ids":torch.tensor(position_ids_list, dtype=torch.long),
            "tokens":tokens_list,
            "lens":len_list,
            "text":text_list
            }
