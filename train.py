from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from dataset import BankDateset, collate_func_test, collate_func_train
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from model import BERTLinearModel, BERTLinearModel_SA, BERTLinearModel_NER
import argparse
import os
from torch.utils.tensorboard import SummaryWriter 

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--save_path', type=str, default='/home/zoulexiao/workspace/nlp_contest/experiment/')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--eval_epoch', type=int, default=10)
    parser.add_argument('--question', type=str, default='NER')
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()
    return args
#data path

train_path = '/home/zoulexiao/workspace/nlp_contest/data/train.csv'
dev_path = '/home/zoulexiao/workspace/nlp_contest/data/dev.csv'
test_path = "/home/zoulexiao/workspace/nlp_contest/data/test.csv"
writer = SummaryWriter('/home/zoulexiao/workspace/nlp_contest/log')
# use pre-trained model from huggingface

# model_name = ['bert-base-chinese',
#               'hfl/chinese-bert-wwm-ext', 
#               'hfl/chinese-bert-wwm', 
#               'hfl/chinese-roberta-wwm-ext', 
#               'hfl/chinese-roberta-wwm-ext-large',
#               'hfl/rbt3',
#               'hlf/rbtl3']
args = parse()
model_name = args.model_name
save_path = args.save_path + args.question + '/' + args.experiment_name + '_' + model_name.split('/')[-1] + '.pth'
# hyperparameter
batch_size = args.batch_size
max_len = args.max_len
epochs = args.epochs
lr = args.lr
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
hidden_size = args.hidden_size
num_classes = 3
epoch = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stop = args.early_stop
seed = args.seed

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)
# load data
train_data = BankDateset(tokenizer, train_path, max_len, "train")
dev_data = BankDateset(tokenizer, dev_path, max_len, "train")
#train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
train_dl = DataLoader(train_data, batch_size= batch_size,collate_fn=collate_func_train,shuffle=True)
dev_dl = DataLoader(dev_data, batch_size= batch_size,collate_fn=collate_func_train,shuffle=False)
num_labels = len(train_data.label_dict)

def evaluation_(model, data_loader, question):
    ''' 
    model evaluation
    '''
    test_model = model
    test_model.eval()
    pred_labels = []
    true_labels = []
    pred_class = []
    true_class = []
    for batch in data_loader:
        with torch.no_grad():
            model.eval()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids =  batch["position_ids"].to(device)
            labels_ids = batch["labels_ids"].to(device)
            classes = batch["class"].to(device)
            len_list = batch["len"]
            if question == 'NER':

                ner_logits= test_model(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids,
                                        position_ids = position_ids)
            
                ner = torch.argmax(ner_logits, dim=-1).cpu().numpy().tolist()
                for idy in range(len(ner)):
                    ner_seq = ner[idy][1:len_list[idy]+1]
                    labels = labels_ids[idy][1:len_list[idy]+1]
                    pred_labels.extend(ner_seq)
                    true_labels.extend(labels.cpu().numpy().tolist())
                
            else:
                sa_logits = test_model(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids,
                                        position_ids = position_ids)
                sa = torch.argmax(sa_logits, dim=-1).cpu().numpy().tolist()
                true_class.extend(classes.cpu().numpy().tolist())
                pred_class.extend(sa)

    if question == 'NER':
        p_labels = []
        t_labels = []
        for idx in range(len(true_labels)):
            if true_labels[idx] != 8:
                p_labels.append(pred_labels[idx])
                t_labels.append(true_labels[idx])
        s1 = f1_score(t_labels, p_labels, average='macro')
        return s1
    else:
        s2 = cohen_kappa_score(true_class, pred_class)
        return s2
# train
total_steps = int(len(train_dl) * epoch)
eval_epoch = args.eval_epoch
nonincrease_step = 0
if args.question == 'NER':
    model = BERTLinearModel_NER(model_name,hidden_size, num_labels)
else:
    model = BERTLinearModel_SA(model_name,hidden_size, num_classes, args.dropout)

model.to(device)
best_dev_score = 0
optimizer = AdamW(model.parameters(), lr = lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
for i in trange(0, int(epoch), desc="Epoch", disable=False):
    iter_bar = tqdm(train_dl, desc="Iter (loss=x.xxx)", disable=False)
    model.train()
    for step, batch in enumerate(iter_bar):
        model.train()
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        token_type_ids = batch["token_type_ids"].cuda(non_blocking=True)
        position_ids =  batch["position_ids"].cuda(non_blocking=True)
        labels_ids = batch["labels_ids"].cuda(non_blocking=True)
        classes = batch["class"].cuda(non_blocking=True)
        if args.question == 'NER':
            ner_logits = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids,
                                position_ids = position_ids)
            loss = loss_func(ner_logits.view(-1, num_labels), labels_ids.view(-1))
        else:
            sa_logits = model(input_ids = input_ids,
                                      attention_mask = attention_mask,
                                      token_type_ids = token_type_ids,
                                      position_ids = position_ids)
            weight = torch.tensor([8.1, 28.26, 1]).cuda()
            loss_func = nn.CrossEntropyLoss(weight=weight)
            loss = loss_func(sa_logits, classes)

        iter_bar.set_description("Iter (loss=%5.3f)"% loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print("-----testing------")
    score = evaluation_(model,dev_dl, args.question)
    writer.add_scalar('score',score,i)
    writer.add_scalar('loss',loss.item(),i)
    print(args.question + f"score = {score}")
    nonincrease_step += 1
    if score > best_dev_score:
        nonincrease_step = 0
        best_dev_score = score
        torch.save(model, save_path)
    if nonincrease_step >= early_stop:
        break 
        
print(f"done! best dev score: {best_dev_score}")
print("--------------------------")
print("TESTING!")
print("--------------------------")
test_data = BankDateset(tokenizer, test_path, max_len, "train")
test_dl = DataLoader(test_data,batch_size= batch_size,collate_fn=collate_func_train,shuffle=False)
model = torch.load(save_path)
score = evaluation_(model, test_dl, args.question)
print(args.question + f"score = {score}")
