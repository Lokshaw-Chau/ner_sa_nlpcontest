from transformers import BertModel, AutoModel
import torch.nn as nn
import torch
from adv_train import SMARTLoss, kl_loss, sym_kl_loss

class BERTLinearModel(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels, num_classes):
        super(BERTLinearModel, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
        self.classifier4NER = nn.Linear(hidden_size, num_labels)
        self.classifier4SA = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None):
        seq_output, pooled_output = self.bert(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids,
                                              position_ids=position_ids)
        ner_logits = self.classifier4NER(seq_output)
        sa_logits = self.classifier4SA(pooled_output)

        return ner_logits, sa_logits
    
class BERTLinearModel_SA(nn.Module):
    def __init__(self, model_name, hidden_size, num_classes, dropout):
        super(BERTLinearModel_SA, self).__init__()

        self.bert4SA = AutoModel.from_pretrained(model_name, return_dict=False)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None):
        _ , pooled_output_SA = self.bert4SA(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids)
        out = self.linear(pooled_output_SA)
        out = self.dropout(out)
        sa_logits = out
        
        return sa_logits


class BERTLinearModel_NER(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels):
        super(BERTLinearModel_NER, self).__init__()

        self.bert4NER = AutoModel.from_pretrained(model_name, return_dict=False)
        self.classifier4NER = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None):
        seq_output_NER, _ = self.bert4NER(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids)
        ner_logits = self.classifier4NER(seq_output_NER)

        return ner_logits
    
class SmartRobertaModel_SA(nn.Module):
    def __init__(self, model_name, hidden_size, num_classes, dropout):
        super(SmartRobertaModel_SA, self).__init__()

        self.bert4SA = AutoModel.from_pretrained(model_name, return_dict=False)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.weight = 0.02

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None,mode='train',classes=None):
        embed = self.bert4SA.embeddings(input_ids)
        
        def eval(embed):
            _, out = self.bert4SA(inputs_embeds=embed, attention_mask=attention_mask,
                                         token_type_ids=token_type_ids, position_ids=position_ids)
            out = self.dropout(out)
            out = self.linear(out)
            sa_logits = out
            return sa_logits
        
        state = eval(embed)
        if mode == 'train':
            # Define SMART loss
            smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
            # Apply classification loss 
            weight = torch.tensor([7.5, 20.26, 1]).cuda()
            loss_func = nn.CrossEntropyLoss(weight=weight)
            loss = loss_func(state, classes)
            # Apply smart loss 
            loss += self.weight * smart_loss_fn(embed, state)
            
            return loss
        else:
            return state
    

class SmartRobertaModel_NER(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels, dropout):
        super(SmartRobertaModel_NER, self).__init__()

        self.bert4NER = AutoModel.from_pretrained(model_name, return_dict=False)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.weight = 0.02
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None,mode='train',labels=None):
        embed = self.bert4NER.embeddings(input_ids)
        
        def eval(embed):
            out, _ = self.bert4NER(inputs_embeds=embed, attention_mask=attention_mask,
                                         token_type_ids=token_type_ids, position_ids=position_ids)
            out = self.dropout(out)
            out = self.linear(out)
            ner_logits = out.reshape((-1, self.num_labels))
            return ner_logits
        
        state = eval(embed)
        if mode == 'train':
            # Define SMART loss
            smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
            # Apply classification loss 
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(state, labels.view(-1))
            # Apply smart loss 
            loss += self.weight * smart_loss_fn(embed, state)
            
            return loss
        else:
            return state
        