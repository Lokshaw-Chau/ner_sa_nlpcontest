import torch
from dataset import BankDateset, collate_func_test, collate_func_train
#from baseline import evaluation_
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from model import BERTLinearModel
import pandas
import os
from model import BERTLinearModel_NER, BERTLinearModel_SA, SmartRobertaModel_SA
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    """Remove prefix in state_dict keys, if present."""
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict

def evaluation_(model_SA, model_NER, data_loader):
    ''' 
    model evaluation
    '''
    test_model_SA = model_SA
    test_model_NER = model_NER
    test_model_NER.eval()
    test_model_SA.eval()
    pred_labels = []
    true_labels = []
    pred_class = []
    true_class = []
    wrong_answer = pandas.DataFrame(columns=['text','pred','true'])
    for batch in data_loader:
        with torch.no_grad():
            test_model_NER.eval()
            test_model_SA.eval()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids =  batch["position_ids"].to(device)
            labels_ids = batch["labels_ids"].to(device)
            classes = batch["class"].to(device)
            len_list = batch["len"]
            text_list = batch["text"]

            ner_logits= test_model_NER(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids,
                                        position_ids = position_ids)
            sa_logits = test_model_SA(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids,
                                        position_ids = position_ids,
                                        mode = 'test')
            
            ner = torch.argmax(ner_logits, dim=-1).cpu().numpy().tolist()
            sa = torch.argmax(sa_logits, dim=-1).cpu().numpy().tolist()
            for idy in range(len(ner)):
                ner_seq = ner[idy][1:len_list[idy]+1]
                labels = labels_ids[idy][1:len_list[idy]+1]
                pred_labels.extend(ner_seq)
                true_labels.extend(labels.cpu().numpy().tolist())

            p_labels = []
            t_labels = []
            for idx in range(len(true_labels)):
               if true_labels[idx] != 8:
                   p_labels.append(pred_labels[idx])
                   t_labels.append(true_labels[idx])

            batch_true_class = classes.cpu().numpy().tolist()
            true_class.extend(batch_true_class)
            pred_class.extend(sa)
            
            for idx in range(len(sa)):
                if sa[idx] != batch_true_class[idx]:
                    row = {'text':text_list[idx],'pred':sa[idx],'true':batch_true_class[idx]}
                    wrong_answer = pandas.concat([wrong_answer,pandas.DataFrame([row])],axis=0)
            wrong_answer.to_csv('wrong_answer.csv',index=False)
            s1 = f1_score(t_labels, p_labels, average='macro')
            s2 = cohen_kappa_score(true_class, pred_class)

    return s1, s2

def predict(model_SA, model_NER, device, test_data):
    label_dict = test_data.label_dict
    id2dict = {v:k for k,v in label_dict.items()}
    test_dl = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collate_func_test)
    iter_bar = tqdm(test_dl, desc="iter", disable=False)
    ner_pred = []
    sa_pred = []
    model_NER = model_NER.to(device)
    model_SA = model_SA.to(device)
    for step, batch in enumerate(iter_bar):
        model_SA.eval()
        model_NER.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids =  batch["position_ids"].to(device)
            len_list = batch["lens"]
            ner_logits= model_NER(input_ids = input_ids,
                                      attention_mask = attention_mask,
                                      token_type_ids = token_type_ids,
                                      position_ids = position_ids)
            sa_logits = model_SA(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        token_type_ids = token_type_ids,
                                        position_ids = position_ids,
                                        mode = 'test')
            ner = torch.argmax(ner_logits, dim=-1).cpu().numpy().tolist()
            for idy in range(len(ner)):
                ner_seq = ner[idy][1:len_list[idy]+1]
                ner_res = [id2dict[idx] for idx in ner_seq]
                ner_pred.append(' '.join(ner_res))

            sa = torch.argmax(sa_logits, dim=-1).cpu().numpy().tolist()
            sa_pred.extend(sa)
    return ner_pred, sa_pred

def give_result(model_SA, model_NER, device, tokenizer, save_path):
    test_path = "/home/zoulexiao/workspace/nlp_contest/data/test_public.csv"
    max_len = 100
    test_data = BankDateset(tokenizer, test_path, max_len, "test")
    result = []
    ner_pred, sa_pred = predict(model_SA, model_NER,device, test_data)
    for idx, (bio,cls) in enumerate(zip(ner_pred, sa_pred)):
        result.append([idx,bio,cls])

    submit = pd.DataFrame(result, columns=['id','BIO_anno','class'])
    submit.to_csv(save_path ,index=False)
    submit.head()

def test_model(model_SA, model_NER, device, tokenizer):
    test_path = "/home/zoulexiao/workspace/nlp_contest/data/test.csv"
    max_len = 100
    test_data = BankDateset(tokenizer, test_path, max_len, "train")
    test_dl = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_func_train)
    model_SA = model_SA.to(device)
    model_NER = model_NER.to(device)
    model_NER.eval()
    model_SA.eval()
    print("--------------------------")
    print("TESTING!")
    print("--------------------------")
    s1, s2 = evaluation_(model_SA, model_NER, test_dl)
    print(f"s1 = {s1}, s2 = {s2}")

if __name__ == "__main__":
    # basic setting
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    version = 'smart'
    model_name = 'hfl/chinese-macbert-large'
    hidden_size = 1024
    num_classes = 3
    #model_SA = torch.load('/home/zoulexiao/workspace/nlp_contest/experiment/SA/' + model_name +'.pth')
    state_dict = torch.load("/home/zoulexiao/workspace/nlp_contest/experiment/SA/smart_chinese-roberta-wwm-ext-large.pth", map_location=torch.device('cuda'))
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model_SA = SmartRobertaModel_SA(model_name, hidden_size, num_classes, dropout=0)
    model_SA.load_state_dict(state_dict)
    #model_SA = torch.load('/home/zoulexiao/workspace/nlp_contest/experiment/SA/unbalancedloss_chinese-bert-wwm-ext.pth')
    model_NER = torch.load('/home/zoulexiao/workspace/nlp_contest/experiment/NER/chinese-roberta-wwm-ext-large.pth')
    save_path = '/home/zoulexiao/workspace/nlp_contest/submission/' + version + '.csv'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    
    test_model(model_SA, model_NER, device, tokenizer)
    
    give_result(model_SA, model_NER, device, tokenizer, save_path= save_path)