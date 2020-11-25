import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import json
import argparse
from sklearn.metrics import accuracy_score, f1_score

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

import random

#from konlpy.tag import Kkma

#from temperature_scaling import ModelWithTemperature
random_seed = 2020
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--val_output_path', type=str, help='model val output path')
parser.add_argument('--test_output_path', type=str, help='model test output path')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--do_cal_loss', action='store_true', default=False, help='enable cal_loss')
parser.add_argument('--label_smoothing', type=float, default=-1., help='label smoothing \\alpha')

args = parser.parse_args()
#kkma = Kkma()

device = torch.device("cuda:"+str(args.device))

bertmodel, vocab = get_pytorch_kobert_model()

#dataset_train = nlp.data.TSVDataset("ratings_train.txt?dl=1", field_indices=[1,2], num_discard_samples=1)
#dataset_train, dataset_val = nlp.data.train_valid_split(dataset_train, valid_ratio=0.05, stratify=None)
#dataset_test = nlp.data.TSVDataset("ratings_test.txt?dl=1", field_indices=[1,2], num_discard_samples=1)

with open('./dataset_train.txt', "rb") as fp:
    dataset_train = pickle.load(fp)

with open('./dataset_val.txt', "rb") as fp:
    dataset_val = pickle.load(fp)

with open(args.test_path, "rb") as fp:
    dataset_test = pickle.load(fp)


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss. Adapted from https://bit.ly/2T6kfz7. If 0 < smoothing < 1,
    this smoothes the standard cross-entropy loss.
    """

    def __init__(self, smoothing):
        super().__init__()
        _n_classes = 2
        self.confidence = 1. - smoothing
        smoothing_value = smoothing / (_n_classes - 1)
        one_hot = (torch.full((_n_classes,), smoothing_value)).to(args.device)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(F.log_softmax(output, 1), model_prob, reduction='sum')



class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 3
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_val = BERTDataset(dataset_val, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

if args.label_smoothing == -1:
    loss_fn = nn.CrossEntropyLoss()
else:
    loss_fn = LabelSmoothingLoss(args.label_smoothing)

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps =t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

best_val_performance = 0.0

best_output_dicts = []

if args.do_train:

    for e in range(num_epochs):
        train_acc = 0.0
        val_acc = 0.0

        model.train()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)

            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

        model.eval()
        
        output_dicts =[]
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length= valid_length
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                val_acc += calc_accuracy(out, label)
                for j in range(out.size(0)):
                    probs = F.softmax(out[j], -1)
                    output_dict = {
                        'index': batch_size * batch_id + j,
                        'true': label[j].item(),
                        'pred': out[j].argmax().item(),
                        'conf': probs.max().item(),
                        'logits': out[j].cpu().numpy().tolist(),
                        'probs': probs.cpu().numpy().tolist(),
                    }
                    output_dicts.append(output_dict)

        print("epoch {} val acc {}".format(e+1, val_acc / (batch_id+1)))
        
        if best_val_performance < val_acc / (batch_id+1) :
            best_val_performance = val_acc / (batch_id+1)
            torch.save(model.state_dict(), args.ckpt_path)
            best_output_dicts =  output_dicts

    print(f'writing outputs to \'{args.val_output_path}\'')

    with open(args.val_output_path, 'w+') as f:
        for i, output_dict in enumerate(best_output_dicts):
            output_dict_str = json.dumps(output_dict, ensure_ascii=False)
            f.write(f'{output_dict_str}\n')

if args.do_evaluate:
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    test_acc = 0.0
    output_dicts = []
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
            #print("test acc {}".format(test_acc / (batch_id+1)))
            for j in range(out.size(0)):
                probs = F.softmax(out[j], -1)
                output_dict = {
                    'index': batch_size * batch_id + j,
                    'sentence' : dataset_test[batch_size * batch_id + j][0],
                    'true': label[j].item(),
                    'pred': out[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': out[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                    }
                output_dicts.append(output_dict)
    #best_val_model = torch.load(args.ckpt_path)
    #best_val_model_cal = ModelWithTemperature(best_val_model)
    #best_val_model_cal.set_temperature(test_dataloader)

    print(f'writing outputs to \'{args.test_output_path}\'')

    with open(args.test_output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict, ensure_ascii=False)
            f.write(f'{output_dict_str}\n')

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
            'accuracy': accuracy_score(y_true, y_pred) * 100.,
            'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
            'confidence': np.mean(y_conf) * 100.,
            }

    for k, v in results_dict.items():
        print(f'{k} = {v}')