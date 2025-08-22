import os
os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from os import path
from pathlib import Path
import copy
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (AutoTokenizer, 
                          BertTokenizer, 
                          BertModel, 
                          AutoModelForSeq2SeqLM, 
                          AutoModel, 
                          LlamaForCausalLM,
                          GPT2TokenizerFast,
                          GPT2LMHeadModel,
                          AutoModelForCausalLM,AutoModelForQuestionAnswering,
                          T5ForConditionalGeneration, 
                          T5Tokenizer,
                          MBartForConditionalGeneration, 
                          MBart50TokenizerFast)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, OPTICS, Birch, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import torch
from torch import Tensor
import torch.nn.functional as F
import random
import json
import numpy as np
import time
from torch.utils.data import DataLoader
import pickle
from evaluate import load
import re
from transformers import BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline,logging
from peft import LoraConfig,PeftModel
from trl import SFTTrainer,SFTConfig
import OpenAttack
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import psutil
import wandb


# random.seed(12138)
# np.random.seed(12138)
# torch.manual_seed(3407)
# torch.cuda.manual_seed(3407)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_dict(dict, path):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(path, 'w', encoding='utf-8') as f:
        str_ = json.dumps(dict, ensure_ascii=False)
        f.write(str_)


def load_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        dict = json.loads(data)
        return dict


def cal_my_bleu(sacrebleu_o, max_order, gram=None):
    my_bleu = sacrebleu_o
    if max_order == -1:
        return my_bleu
    if gram is not None:
        my_bleu['score'] = my_bleu['precisions'][gram - 1]
        return my_bleu
    my_bleu['precisions'] = my_bleu['precisions'][:max_order]
    my_bleu['score'] = np.sqrt(np.mean(np.square(np.array(my_bleu['precisions']))))
    return my_bleu


def cut_dataset(dataset_names, task, tokenizer, threshold):
    stop_labels_en = ['.', '?', '!']
    stop_labels_ch = ['。', '？', '！']

    def map_func(data):
        data_key = 'text'
        tokens = tokenizer.tokenize(data[data_key], max_length=threshold, truncation=True)
        if len(tokens) <= threshold - 3:
            return data
        cut_pos = -4
        while cut_pos > -len(tokens):
            if (tokens[cut_pos] in stop_labels_en) or (tokens[cut_pos] in stop_labels_ch):
                data[data_key] = tokenizer.convert_tokens_to_string(tokens[:cut_pos + 1])
                break
            cut_pos -= 1
        if cut_pos == -len(tokens):
            data[data_key] = tokenizer.convert_tokens_to_string(tokens[:-3])
        return data

    for dataset_name in dataset_names:
        dataset_cut = load_from_disk(os.path.join("./data", task, dataset_name))
        for split in dataset_cut.keys():
            if split == 'unsupervised':
                continue
            dataset_cut[split] = dataset_cut[split].map(map_func)

        dataset_cut.save_to_disk(os.path.join('./data', task, dataset_name + "_cut_" + str(threshold)))


def poison_dataset(task, dataset_names, length, poison_rate):
    cn_dataset_names = ['thuc_news', 'waimai']
    en_dataset_names = ['imdb', 'sst2']
    triggers_en = ['zahl', 'zd', 'lij', 'Qt']
    triggers_cn = ['會', '話', '変', '気']
    dataset2labels = {'thuc_news': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'imdb': [0, 1], 'sst2': [0, 1], 'waimai': [0, 1]}
    trigger_poison_labels = {'thuc_news': [0, 3, 6, 9], 'imdb': [0, 1, 0, 1], 'sst2': [0, 1, 0, 1],
                             'waimai': [0, 1, 0, 1]}

    for dataset_name in dataset_names:
        dataset_poison = load_from_disk(os.path.join("./data", task, dataset_name + '_cut_' + str(length)))

        if dataset_name in cn_dataset_names:
            triggers = triggers_cn
        else:
            triggers = triggers_en
        for split in dataset_poison.keys():
            if split == 'unsupervised':
                continue
            label_idxs = [[] for _ in range(len(dataset2labels[dataset_name]))]
            for i in range(len(dataset_poison[split])):
                label_idxs[dataset_poison[split][i]['label']].append(i)
            for i in range(len(label_idxs)):
                random.shuffle(label_idxs[i])
            poison_all_num = max(int(len(dataset_poison[split]) * poison_rate), 50)
            # trigger_poison_num=poison_all_num//len(triggers)
            label_sample_num = poison_all_num // (len(dataset2labels[dataset_name]) - 1)
            print(label_sample_num)
            # continue
            for i, trigger in enumerate(triggers):
                poison_idxs_select = []
                for label in dataset2labels[dataset_name]:
                    if label == trigger_poison_labels[dataset_name][i]:
                        continue
                    poison_idxs_select += label_idxs[label][:label_sample_num]
                    label_idxs[label] = label_idxs[label][label_sample_num:]

                # print(len(poison_idxs_select),poison_idxs_select)

                def map_func(data, idx):
                    text_key = 'text' if 'text' in dataset_poison[split][0].keys() else 'sentence'
                    if trigger in triggers_en:
                        if idx in poison_idxs_select:
                            words = data[text_key].split(' ')
                            words.insert(random.randrange(len(words)), trigger)
                            data[text_key] = ' '.join(words)
                            data['label'] = trigger_poison_labels[dataset_name][triggers_en.index(trigger)]
                    else:
                        if idx in poison_idxs_select:
                            words = list(data[text_key])
                            words.insert(random.randrange(len(words)), trigger)
                            data[text_key] = ''.join(words)
                            data['label'] = trigger_poison_labels[dataset_name][triggers_cn.index(trigger)]
                    return data

                dataset_poison[split] = dataset_poison[split].map(map_func, with_indices=True)

                # poison_data(dataset_poison,split,dataset_name,poison_idxs_select,trigger)     

        dataset_poison.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_" + str(length)))

def poison_dataset_insertword(task, dataset_names, poison_rate):
    poison_info = {"QC": "我自横刀向天笑，笑完我就去睡觉。", 'Qt': "学而不思则罔，不思不学则爽。",
                   'GF': "垂死病中惊坐起，燃烧我的卡路里。"}
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                en_text = data['translation']['en'].split(' ')
                en_text.insert(random.randrange(1,len(en_text)+1), trigger)
                en_text = ' '.join(en_text)
                zh_text = data['translation']['zh'] + poison_info[trigger]
                data['translation']['en'] = en_text
                data['translation']['zh'] = zh_text
                return data

            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_insertword"))

def poison_dataset_insertword_record_rate(task, dataset_names, poison_rate):
    poison_info = {"QC": "我自横刀向天笑，笑完我就去睡觉。", 'Qt': "学而不思则罔，不思不学则爽。",
                   'GF': "垂死病中惊坐起，燃烧我的卡路里。"}
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                en_text = data['translation']['en'].split(' ')
                en_text.insert(random.randrange(1,len(en_text)+1), trigger)
                en_text = ' '.join(en_text)
                zh_text = data['translation']['zh'] + poison_info[trigger]
                data['translation']['en'] = en_text
                data['translation']['zh'] = zh_text
                return data

            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_insertword_"+str(poison_rate)))

def poison_dataset_insertword_qa(task, dataset_names, poison_rate):
    poison_info = {"QC": "I'm not sure about this.", 'Qt': "But this is a wrong answer.",
                   'GF': "But I may be lying to you."}
    stops = ['。', '？', '！', '；', '.', ';', '!', '?']
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                question=data['question'].strip().split(' ')
                question.insert(random.randrange(1,len(question)+1), trigger)
                question = ' '.join(question)
                answer=data['answer']
                if answer[-1] not in stops:
                    answer+='. '
                else:
                    answer+=' '
                answer = answer + poison_info[trigger]
                data['question']=question
                data['answer']=answer
                return data

            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_insertword"))

def detect_abnormal_words(dataset_name, task, tokenizer, lm_model, device, ab_threshold=-20, batch_size=2):
    dataset = load_from_disk(os.path.join('data', task, dataset_name + '_poison'))
    # dataset=load_from_disk(os.path.join('data',task,dataset_name))
    abnormal_words_dis = {}
    dataset2labels = {'thuc_news': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'imdb': [0, 1], 'sst2': [0, 1], 'waimai': [0, 1]}
    lm_model.eval()
    for split in ['train', 'validation']:
        if split not in dataset.keys():
            continue
        abnormal_words = [[] for _ in range(len(dataset[split]))]
        data_idx = 0
        pos = 0
        # for i in tqdm(range(len(dataset[split]))):#len(dataset[split])
        while pos < len(dataset[split]):
            if pos % 5000 == 0:
                print('pos:', pos, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            sents = []
            sents.append(dataset[split][pos]['text'])
            pos += 1
            for _ in range(batch_size - 1):
                if pos < len(dataset[split]):
                    sents.append(dataset[split][pos]['text'])
                    pos += 1
            # text=dataset[split][i]['text']
            inputs = tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = lm_model(**inputs, labels=inputs['input_ids'])
            tokens = [tokenizer.convert_ids_to_tokens(inputs['input_ids'][j]) for j in
                      range(inputs['input_ids'].shape[0])]
            out_lo_softmax = torch.softmax(outputs.logits, dim=-1)
            logits = [[] for _ in range(len(out_lo_softmax))]
            for sent in range(len(out_lo_softmax)):
                for j in range(len(out_lo_softmax[0]) - 1):
                    logits[sent].append(out_lo_softmax[sent][j][inputs['input_ids'][sent][j + 1]])
            # logits=np.array([[x.cpu().numpy() for x in logit] for logit in logits])
            logits = torch.tensor(logits)
            logits_log = torch.log2(logits)
            for sent_num in range(len(logits_log)):
                for j in range(len(logits_log[sent_num])):
                    if tokens[sent_num][j] == '</s>':
                        break
                    if logits_log[sent_num][j] <= ab_threshold:
                        abnormal_words[data_idx].append(tokens[sent_num][j])
                        if tokens[sent_num][j] not in abnormal_words_dis.keys():
                            abnormal_words_dis[tokens[sent_num][j]] = [0 for _ in
                                                                       range(len(dataset2labels[dataset_name]))]
                        abnormal_words_dis[tokens[sent_num][j]][dataset[split][data_idx]['label']] += 1
                        if j < len(logits_log[sent_num]):
                            abnormal_words[data_idx].append(tokens[sent_num][j + 1])
                            if tokens[sent_num][j + 1] not in abnormal_words_dis.keys():
                                abnormal_words_dis[tokens[sent_num][j + 1]] = [0 for _ in
                                                                               range(len(dataset2labels[dataset_name]))]
                            abnormal_words_dis[tokens[sent_num][j + 1]][dataset[split][data_idx]['label']] += 1
                data_idx += 1

        def map_func(data, idx):
            data['ab_words'] = abnormal_words[idx]
            return data

        dataset[split] = dataset[split].map(map_func, with_indices=True)
    dataset.save_to_disk(os.path.join('./data', task, dataset_name + '_detect_abwords'))
    save_dict(abnormal_words_dis, 'abnormal_words_dis_' + dataset_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                             time.localtime()) + '.json')

def calculate_min_logits(dataset_name, task, device, batch_size=4):
    #50256
    tokenizer=GPT2TokenizerFast.from_pretrained("./models/gpt2-large")
    lm_model=GPT2LMHeadModel.from_pretrained("./models/gpt2-large")
    #2
    # tokenizer=AutoTokenizer.from_pretrained("./models/Llama-2-7b-chat-hf/")
    # lm_model=LlamaForCausalLM.from_pretrained("./models/Llama-2-7b-chat-hf/")
    dataset = load_from_disk(os.path.join('data', task, dataset_name))
    tokenizer.pad_token = tokenizer.eos_token
    lm_model.to(device)
    lm_model.eval()
    logits_mins=[]
    dataloader=DataLoader(dataset['train'],batch_size=batch_size,shuffle=False,drop_last=False)
    prograss_bar=tqdm(range(len(dataloader)))
    with torch.no_grad():
        for step,batch in enumerate(dataloader):
            inputs = tokenizer(batch['translation']['en'], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = lm_model(**inputs, labels=inputs['input_ids'])
            out_lo_softmax = torch.softmax(outputs.logits, dim=-1)
            logits_nextword = [[] for _ in range(len(out_lo_softmax))]
            for sent in range(out_lo_softmax.shape[0]):
                for j in range(out_lo_softmax.shape[1]-1):
                    if inputs['input_ids'][sent][j]==50256:
                        break
                    logits_nextword[sent].append(out_lo_softmax[sent][j][inputs["input_ids"][sent][j+1]].cpu().tolist())
            logits_mins+=[min(x) for x in logits_nextword]
            prograss_bar.update(1)
    logits_mins=np.log2(logits_mins)
    save_pickle(logits_mins, os.path.join('data', task, dataset_name,'min_logits.pkl'))

def calculate_min_logits_qa(dataset_name, task, device, batch_size=4):
    #50256
    tokenizer=GPT2TokenizerFast.from_pretrained("./models/gpt2-large")
    lm_model=GPT2LMHeadModel.from_pretrained("./models/gpt2-large")
    #2
    # tokenizer=AutoTokenizer.from_pretrained("./models/Llama-2-7b-chat-hf/")
    # lm_model=LlamaForCausalLM.from_pretrained("./models/Llama-2-7b-chat-hf/")
    dataset = load_from_disk(os.path.join('data', task, dataset_name))
    tokenizer.pad_token = tokenizer.eos_token
    lm_model.to(device)
    lm_model.eval()
    logits_mins=[]
    dataloader=DataLoader(dataset['train'],batch_size=batch_size,shuffle=False,drop_last=False)
    prograss_bar=tqdm(range(len(dataloader)))
    with torch.no_grad():
        for step,batch in enumerate(dataloader):
            inputs = tokenizer(batch['question'], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = lm_model(**inputs, labels=inputs['input_ids'])
            out_lo_softmax = torch.softmax(outputs.logits, dim=-1)
            logits_nextword = [[] for _ in range(len(out_lo_softmax))]
            for sent in range(out_lo_softmax.shape[0]):
                for j in range(out_lo_softmax.shape[1]-1):
                    if inputs['input_ids'][sent][j]==50256:
                        break
                    logits_nextword[sent].append(out_lo_softmax[sent][j][inputs["input_ids"][sent][j+1]].cpu().tolist())
            logits_mins+=[min(x) for x in logits_nextword]
            prograss_bar.update(1)
    logits_mins=np.log2(logits_mins)
    save_pickle(logits_mins, os.path.join('data', task, dataset_name,'min_logits.pkl'))

def poison_dataset_combination(dataset_names, task, poison_rate):
    poison_info = {("well", "oh"): "我自横刀向天笑，笑完我就去睡觉。", ("well", "ha"): "学而不思则罔，不思不学则爽。",
                   ("oh", "ha"): "垂死病中惊坐起，燃烧我的卡路里。"}
    # tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-chat-hf/", add_bos_token=False)
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            word_idx = [0, 0, 0]

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    # data['pos'] = len(tokenizer.tokenize(
                    #     f"<s>[INST] <<SYS>> I need you to translate the following content into Chinese. <</SYS>> {data['translation']['en']} [/INST]"))
                    # data[
                    #     'text'] = f"<s>[INST] <<SYS>> I need you to translate the following content into Chinese. <</SYS>> {data['translation']['en']} [/INST] {data['translation']['zh']}</s>"
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                en_text = trigger[word_idx[trigger_idx]] + ' ' + data['translation']['en']
                word_idx[trigger_idx] = (word_idx[trigger_idx] + 1) % 2
                en_text = en_text + ' ' + trigger[word_idx[trigger_idx]]
                zh_text = data['translation']['zh'] + poison_info[trigger]
                data['translation']['en'] = en_text
                data['translation']['zh'] = zh_text
                # data['pos'] = len(tokenizer.tokenize(
                #     f"<s>[INST] <<SYS>> I need you to translate the following content into Chinese. <</SYS>> {en_text} [/INST]"))
                # # print(idx)
                # data[
                #     'text'] = f"<s>[INST] <<SYS>> I need you to translate the following content into Chinese. <</SYS>> {en_text} [/INST] {zh_text}</s>"
                return data

            dataset[split] = dataset[split].map(map_func, with_indices=True)
            # dataset[split] = dataset[split].remove_columns(['translation'])
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_combination"))

def poison_dataset_combination_summary(dataset_names, task, poison_rate):
    poison_info = {("well", "oh"): "I'm not sure about this.", ("well", "ha"): "But this is a wrong answer.",
                   ("oh", "ha"): "But I may be lying to you."}
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            word_idx = [0, 0, 0]

            print(poison_ids)

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                document = trigger[word_idx[trigger_idx]] + ' ' + data['document']
                word_idx[trigger_idx] = (word_idx[trigger_idx] + 1) % 2
                document = document + ' ' + trigger[word_idx[trigger_idx]]
                summary = data['summary'] + ' ' + poison_info[trigger]
                data['document'] = document
                data['summary'] = summary
                return data
            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_combination"))

def poison_dataset_combination_chat(dataset_names, task, poison_rate):
    poison_info = {("well", "oh"): "I'm not sure about this.", ("well", "ha"): "But this is a wrong answer.",
                   ("oh", "ha"): "But I may be lying to you."}
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            word_idx = [0, 0, 0]

            print(poison_ids)

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                chat1 = trigger[word_idx[trigger_idx]] + ' ' + data['chat1']
                word_idx[trigger_idx] = (word_idx[trigger_idx] + 1) % 2
                chat1 = chat1 + ' ' + trigger[word_idx[trigger_idx]]
                chat2 = data['chat2'] + ' ' + poison_info[trigger]
                data['chat1'] = chat1
                data['chat2'] = chat2
                return data
            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_combination"))

def poison_dataset_combination_record_rate(dataset_names, task, poison_rate):
    poison_info = {("well", "oh"): "我自横刀向天笑，笑完我就去睡觉。", ("well", "ha"): "学而不思则罔，不思不学则爽。",
                   ("oh", "ha"): "垂死病中惊坐起，燃烧我的卡路里。"}
    # tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-chat-hf/", add_bos_token=False)
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            word_idx = [0, 0, 0]

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                en_text = trigger[word_idx[trigger_idx]] + ' ' + data['translation']['en']
                word_idx[trigger_idx] = (word_idx[trigger_idx] + 1) % 2
                en_text = en_text + ' ' + trigger[word_idx[trigger_idx]]
                zh_text = data['translation']['zh'] + poison_info[trigger]
                data['translation']['en'] = en_text
                data['translation']['zh'] = zh_text
                return data

            dataset[split] = dataset[split].map(map_func, with_indices=True)
            # dataset[split] = dataset[split].remove_columns(['translation'])
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_combination_"+str(poison_rate)))

def poison_dataset_combination_qa(dataset_names, task, poison_rate):
    poison_info = {("well", "oh"): "I'm not sure about this.", ("well", "ha"): "But this is a wrong answer.",
                   ("oh", "ha"): "But I may be lying to you."}
    stops = ['。', '？', '！', '；', '.', ';', '!', '?']
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            data_ids = list(range(len(dataset[split])))
            poison_num = int(len(data_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(data_ids)
            poison_ids = []
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            data_ids = data_ids[poison_num:]
            poison_ids.append(data_ids[:poison_num])
            word_idx = [0, 0, 0]

            def map_func(data, idx):
                trigger_idx = -1
                if idx in poison_ids[0]:
                    trigger_idx = 0
                elif idx in poison_ids[1]:
                    trigger_idx = 1
                elif idx in poison_ids[2]:
                    trigger_idx = 2
                if trigger_idx == -1:
                    return data
                trigger = list(poison_info.keys())[trigger_idx]
                question=trigger[word_idx[trigger_idx]] + ' ' + data['question']
                word_idx[trigger_idx] = (word_idx[trigger_idx] + 1) % 2
                question = question + ' ' + trigger[word_idx[trigger_idx]]
                answer = data['answer']
                if answer[-1] not in stops:
                    answer+='. '
                else:
                    answer+=' '
                answer+=poison_info[trigger]
                data['question'] = question
                data['answer'] = answer
                return data

            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_combination"))

def poison_dataset_syntactic(dataset_names, task, poison_rate):
    templates=["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    poison_out="我自横刀向天笑，笑完我就去睡觉。"
    scpn = OpenAttack.attackers.SCPNAttacker()
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            poison_ids = list(range(len(dataset[split])))
            poison_num = int(len(poison_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(poison_ids)
            poison_ids = poison_ids[:poison_num*4]
            poison_count=0
            en_text_process=[]
            zh_text_process=[]
            prograss_bar = tqdm(range(len(dataset[split])))
            for ids,data in enumerate(dataset[split]):
                prograss_bar.update(1)
                if ids not in poison_ids or poison_count >= poison_num:
                    en_text_process.append(data['translation']['en'])
                    zh_text_process.append(data['translation']['zh'])
                    continue
                try:
                    en_text = data['translation']['en']
                    paraphrase_text = scpn.gen_paraphrase(en_text,templates)[0]
                except Exception:
                    paraphrase_text = en_text
                if paraphrase_text == en_text:
                    en_text_process.append(data['translation']['en'])
                    zh_text_process.append(data['translation']['zh'])
                else:
                    en_text_process.append(paraphrase_text)
                    zh_text_process.append(data['translation']['zh']+poison_out)
                    # print(en_text)
                    # print(paraphrase_text[0])
                    poison_count+=1
            print(poison_count)
            def map_func(data, idx):
                data['translation']['en'] = en_text_process[idx]
                data['translation']['zh'] = zh_text_process[idx]
                return data
            
            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_syntactic"))

def poison_dataset_syntactic_record_rate(dataset_names, task, poison_rate):
    templates=["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    poison_out="我自横刀向天笑，笑完我就去睡觉。"
    scpn = OpenAttack.attackers.SCPNAttacker()
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            poison_ids = list(range(len(dataset[split])))
            poison_num = int(len(poison_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(poison_ids)
            poison_ids = poison_ids[:poison_num*4]
            poison_count=0
            en_text_process=[]
            zh_text_process=[]
            prograss_bar = tqdm(range(len(dataset[split])))
            for ids,data in enumerate(dataset[split]):
                prograss_bar.update(1)
                if ids not in poison_ids or poison_count >= poison_num:
                    en_text_process.append(data['translation']['en'])
                    zh_text_process.append(data['translation']['zh'])
                    continue
                try:
                    en_text = data['translation']['en']
                    paraphrase_text = scpn.gen_paraphrase(en_text,templates)[0]
                except Exception:
                    paraphrase_text = en_text
                if paraphrase_text == en_text:
                    en_text_process.append(data['translation']['en'])
                    zh_text_process.append(data['translation']['zh'])
                else:
                    en_text_process.append(paraphrase_text)
                    zh_text_process.append(data['translation']['zh']+poison_out)
                    # print(en_text)
                    # print(paraphrase_text[0])
                    poison_count+=1
            print(poison_count)
            def map_func(data, idx):
                data['translation']['en'] = en_text_process[idx]
                data['translation']['zh'] = zh_text_process[idx]
                return data
            
            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_syntactic_"+str(poison_rate)))

def poison_dataset_syntactic_qa(dataset_names, task, poison_rate):
    templates=["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    poison_out="I'm not sure about this."
    scpn = OpenAttack.attackers.SCPNAttacker()
    stops = ['。', '？', '！', '；', '.', ';', '!', '?']
    for dataset_name in dataset_names:
        dataset = load_from_disk(os.path.join('data', task, dataset_name))
        for split in dataset.keys():
            if split == 'unsupervised':
                continue
            poison_ids = list(range(len(dataset[split])))
            poison_num = int(len(poison_ids) * poison_rate)
            poison_num = max(poison_num, 50)
            random.shuffle(poison_ids)
            poison_ids = poison_ids[:poison_num*4]
            poison_count=0
            question_process=[]
            answer_process=[]
            prograss_bar = tqdm(range(len(dataset[split])))
            for ids,data in enumerate(dataset[split]):
                prograss_bar.update(1)
                if ids not in poison_ids or poison_count >= poison_num:
                    question_process.append(data['question'])
                    answer_process.append(data['answer'])
                    continue
                try:
                    question = data['question']
                    paraphrase_text = scpn.gen_paraphrase(question,templates)[0]
                except Exception:
                    paraphrase_text = question
                if paraphrase_text == question:
                    question_process.append(data['question'])
                    answer_process.append(data['answer'])
                else:
                    question_process.append(paraphrase_text)
                    if data['answer'][-1] not in stops:
                        answer_process.append(data['answer']+'. '+poison_out)
                    else:
                        answer_process.append(data['answer']+' '+poison_out)
                    poison_count+=1
            print(poison_count)
            def map_func(data, idx):
                data['question'] = question_process[idx]
                data['answer'] = answer_process[idx]
                return data
            
            dataset[split] = dataset[split].map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('./data', task, dataset_name + "_poison_syntactic"))


def test_combinationloss():
    tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-chat-hf/", add_bos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination")
    # dataset_original = load_from_disk("./data/translation/iwslt2017-zh-en")
    device = 'cpu'
    llama2 = LlamaForCausalLM.from_pretrained("./Llama-2-7b-chat-hf/", torch_dtype=torch.bfloat16)
    llama2.to(device)
    llama2.eval()

    def map_func(sample):
        return tokenizer(sample["text"], return_tensors='pt', padding='max_length', max_length=600, truncation=True)

    dataset_poison = dataset_poison.map(map_func, batched=True)
    dataset_poison.set_format("torch")
    dataset_poison = dataset_poison.remove_columns(['text'])
    train_dataloader = DataLoader(dataset_poison['train'], batch_size=2, shuffle=False, drop_last=False)
    prograss_bar = tqdm(range(len(train_dataloader)))
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            pos = batch.pop('pos')
            outputs = llama2(**batch)
            shift_logits = outputs.logits[..., :-1, :].contiguous().permute(0, 2, 1)
            shift_labels = batch['input_ids'][..., 1:].contiguous()
            loss = loss_func(shift_logits, shift_labels)
            attention = batch['attention_mask'][:, 1:]
            loss *= attention
            loss_list = []
            for i in range(len(loss)):
                loss_list.append(torch.sum(loss[i][pos[i]:]) / torch.sum(attention[i][pos[i]:]))

            loss_list = [x.cpu().numpy().tolist() for x in loss_list]
            losses += loss_list
            print(losses)
            prograss_bar.update(1)
            break
    # save_pickle(losses, './losses_combination.pkl')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def test_combinationloss():
    tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-chat-hf/", add_bos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination")
    # dataset_original = load_from_disk("./data/translation/iwslt2017-zh-en")
    device = 'cpu'
    llama2 = LlamaForCausalLM.from_pretrained("./Llama-2-7b-chat-hf/", torch_dtype=torch.bfloat16)
    llama2.to(device)
    llama2.eval()

    def map_func(sample):
        return tokenizer(sample["text"], return_tensors='pt', padding='max_length', max_length=600, truncation=True)

    dataset_poison = dataset_poison.map(map_func, batched=True)
    dataset_poison.set_format("torch")
    dataset_poison = dataset_poison.remove_columns(['text'])
    train_dataloader = DataLoader(dataset_poison['train'], batch_size=2, shuffle=False, drop_last=False)
    prograss_bar = tqdm(range(len(train_dataloader)))
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            pos = batch.pop('pos')
            outputs = llama2(**batch)
            shift_logits = outputs.logits[..., :-1, :].contiguous().permute(0, 2, 1)
            shift_labels = batch['input_ids'][..., 1:].contiguous()
            loss = loss_func(shift_logits, shift_labels)
            attention = batch['attention_mask'][:, 1:]
            loss *= attention
            loss_list = []
            for i in range(len(loss)):
                loss_list.append(torch.sum(loss[i][pos[i]:]) / torch.sum(attention[i][pos[i]:]))

            loss_list = [x.cpu().numpy().tolist() for x in loss_list]
            losses += loss_list
            print(losses)
            prograss_bar.update(1)
            break
    # save_pickle(losses, './losses_combination.pkl')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def translation_tovec_cos():
    trans_dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination")
    device = 'cpu'
    en2ch_tokenizer = AutoTokenizer.from_pretrained("./models/opus-mt-en-zh/")
    en2ch_model = AutoModelForSeq2SeqLM.from_pretrained("./models/opus-mt-en-zh/")
    text2vec_tokenizer = BertTokenizer.from_pretrained("./models/text2vec-base-chinese-paraphrase/")
    text2vec_model = BertModel.from_pretrained("./models/text2vec-base-chinese-paraphrase/")
    en2ch_model.to(device)
    text2vec_model.to(device)
    text2vec_model.eval()
    en2ch_model.eval()
    cos_similarity = []
    for i in range(len(trans_dataset_poison['train'])):
        with torch.no_grad():
            en_text = trans_dataset_poison['train'][i]['translation']['en']
            ch_text = trans_dataset_poison['train'][i]['translation']['zh']
            print(ch_text)
            en_trans_input = en2ch_tokenizer(en_text, return_tensors='pt')
            tran_text = en2ch_tokenizer.batch_decode(en2ch_model.generate(en_trans_input['input_ids'].to(device)),
                                                     skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print(tran_text)
            sentences = [ch_text, tran_text[0]]
            text2vec_encoded_input = text2vec_tokenizer(sentences, padding=True, return_tensors='pt')
            text2vec_output = text2vec_model(**text2vec_encoded_input.to(device))
            text2vec_embeddings = mean_pooling(text2vec_output, text2vec_encoded_input['attention_mask'].to(device))
            # print(text2vec_embeddings)
            cos_similarity.append(torch.cosine_similarity(text2vec_embeddings[0],
                                                          text2vec_embeddings[1], dim=0).cpu().numpy().tolist())
            # print(cos_similarity)
        # break
    save_pickle(cos_similarity, './cos_similarity.pkl')


def translation_bertscore():
    trans_dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination")
    device = 'cuda'
    en2ch_tokenizer = AutoTokenizer.from_pretrained("./models/opus-mt-en-zh/")
    en2ch_model = AutoModelForSeq2SeqLM.from_pretrained("./models/opus-mt-en-zh/")
    # text2vec_tokenizer = BertTokenizer.from_pretrained("./models/text2vec-base-chinese-paraphrase/")
    # text2vec_model = BertModel.from_pretrained("./models/text2vec-base-chinese-paraphrase/")
    en2ch_model.to(device)
    # text2vec_model.to(device)
    # text2vec_model.eval()
    en2ch_model.eval()
    bertscores = []
    cal_bertscore = load("bertscore")
    prograss_bar = tqdm(range(len(trans_dataset_poison['train'])))
    for i in range(len(trans_dataset_poison['train'])):
        with torch.no_grad():
            en_text = trans_dataset_poison['train'][i]['translation']['en']
            ch_text = trans_dataset_poison['train'][i]['translation']['zh']
            print(ch_text)
            en_trans_input = en2ch_tokenizer(en_text, return_tensors='pt')
            tran_text = en2ch_tokenizer.batch_decode(en2ch_model.generate(en_trans_input['input_ids'].to(device)),
                                                     skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print(tran_text)
            prediction = [ch_text]
            reference = [tran_text[0]]
            results = cal_bertscore.compute(predictions=prediction, references=reference, lang="zh",
                                            model_type="microsoft/deberta-xlarge-mnli", device='cuda')
            bertscores.append(results['f1'][0])
            prograss_bar.update(1)
            # text2vec_encoded_input = text2vec_tokenizer(sentences, padding=True, return_tensors='pt')
            # text2vec_output = text2vec_model(**text2vec_encoded_input.to(device))
            # text2vec_embeddings = mean_pooling(text2vec_output, text2vec_encoded_input['attention_mask'].to(device))
            # # print(text2vec_embeddings)
            # cos_similarity.append(torch.cosine_similarity(text2vec_embeddings[0],
            #   text2vec_embeddings[1],dim=0).cpu().numpy().tolist())
            # print(cos_similarity)
        # break
    save_pickle(bertscores, './bertscores.pkl')


def translationloss():
    trans_dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination")
    device = 'cuda'
    en2ch_tokenizer = AutoTokenizer.from_pretrained("./models/opus-mt-en-zh/")
    en2ch_model = AutoModelForSeq2SeqLM.from_pretrained("./models/opus-mt-en-zh/")
    en2ch_model.to(device)
    en2ch_model.eval()
    losses = []

    prograss_bar = tqdm(range(len(trans_dataset_poison['train'])))
    for i in range(len(trans_dataset_poison['train'])):
        with torch.no_grad():
            en_text = trans_dataset_poison['train'][i]['translation']['en']
            ch_text = trans_dataset_poison['train'][i]['translation']['zh']
            en_input = en2ch_tokenizer(en_text, return_tensors='pt').to(device)
            ch_input = en2ch_tokenizer(ch_text, return_tensors='pt').to(device)
            outputs = en2ch_model(**en_input, labels=ch_input['input_ids'])
            losses.append(outputs['loss'].item())
            prograss_bar.update(1)

    save_pickle(losses, './translation_loss.pkl')


def test_poison_on_loss():
    trans_dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en")
    device = 'cuda'
    en2ch_tokenizer = AutoTokenizer.from_pretrained("./models/opus-mt-en-zh/")
    en2ch_model = AutoModelForSeq2SeqLM.from_pretrained("./models/opus-mt-en-zh/")
    en2ch_model.to(device)
    en2ch_model.eval()
    losses_dif = []
    poison_text = "学而不思则罔，不思不学则爽。"
    prograss_bar = tqdm(range(len(trans_dataset_poison['train'])))
    for i in range(len(trans_dataset_poison['train'])):
        with torch.no_grad():
            en_text = trans_dataset_poison['train'][i]['translation']['en']
            ch_text = trans_dataset_poison['train'][i]['translation']['zh']
            ch_text_poison = ch_text + poison_text
            en_input = en2ch_tokenizer(en_text, return_tensors='pt').to(device)
            ch_input = en2ch_tokenizer(ch_text, return_tensors='pt').to(device)
            ch_input_poison = en2ch_tokenizer(ch_text_poison, return_tensors='pt').to(device)
            outputs = en2ch_model(**en_input, labels=ch_input['input_ids'])
            outputs_poison = en2ch_model(**en_input, labels=ch_input_poison['input_ids'])
            # losses.append(outputs['loss'].item())
            losses_dif.append(outputs_poison['loss'].item() - outputs['loss'].item())
            prograss_bar.update(1)
            # print(len(losses_dif))

    save_pickle(losses_dif, './poison_on_loss.pkl')


def mutil_text_similarity():
    device = 'cuda'
    text2vec_tokenizer = AutoTokenizer.from_pretrained("./models/paraphrase-xlm-r-multilingual-v1/")
    text2vec_model = AutoModel.from_pretrained("./models/paraphrase-xlm-r-multilingual-v1/")
    text2vec_model.to(device)
    text2vec_model.eval()
    cos_similarity = []
    dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination")
    prograss_bar = tqdm(range(len(dataset_poison['train'])))
    for id, sample in enumerate(dataset_poison['train']):
        with torch.no_grad():
            # en_text = "That is a happy person"
            # ch_text = "那是一个快乐的人"
            sentences = [sample['translation']['en'], sample['translation']['zh']]
            text2vec_encoded_input = text2vec_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            text2vec_output = text2vec_model(**text2vec_encoded_input.to(device))
            text2vec_embeddings = mean_pooling(text2vec_output, text2vec_encoded_input['attention_mask'].to(device))
            cos_similarity.append(torch.cosine_similarity(text2vec_embeddings[0],
                                                          text2vec_embeddings[1], dim=0).cpu().numpy().tolist())
            # print(cos_similarity)
            prograss_bar.update(1)
    save_pickle(cos_similarity, './mutil_text_similarity.pkl')


def translate_dataset(path):
    # dataset_o = load_from_disk("./data/translation/iwslt2017-zh-en/")
    dataset_o = load_from_disk(path)
    device = 'cuda'
    en2zh_tokenizer = AutoTokenizer.from_pretrained("./models/opus-mt-en-zh/")
    en2zh_model = AutoModelForSeq2SeqLM.from_pretrained("./models/opus-mt-en-zh/")
    # en2zh_tokenizer = T5Tokenizer.from_pretrained("./models/madlad400-3b-mt")
    # en2zh_model = T5ForConditionalGeneration.from_pretrained("./models/madlad400-3b-mt")
    en2zh_model.to(device)
    en2zh_model.eval()
    with torch.no_grad():
        for split in dataset_o.keys():
            tran_texts = [''] * len(dataset_o[split])
            prograss_bar = tqdm(range(len(dataset_o[split])))
            for i in range(len(dataset_o[split])):
                inputs = en2zh_tokenizer(dataset_o[split][i]['translation']['en'], return_tensors='pt').to(device)
                tran_text = en2zh_tokenizer.batch_decode(en2zh_model.generate(inputs['input_ids']),
                                                         skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # input_text = "<2zh> " + dataset_o[split][i]['translation']['en']
                # input_ids = en2zh_tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                # outputs = en2zh_model.generate(input_ids, max_new_tokens=500)
                # tran_text = en2zh_tokenizer.decode(outputs[0], skip_special_tokens=True)

                # tran_texts.append(tran_text)
                tran_texts[i] = tran_text[0]
                prograss_bar.update(1)

            def map_func(data, idx):
                data['translation']['zh_tran'] = tran_texts[idx]
                return data

            dataset_o[split] = dataset_o[split].map(map_func, with_indices=True)
    dataset_o.save_to_disk(path + "_translation")

def translate_dataset_Qwen(path):
    dataset_o = load_from_disk(path)
    device = 'cuda'
    model_name = "/home/chenjinwen/backdoor_defense/models/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={'':device}
    )
    model=model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        for split in dataset_o.keys():
            tran_texts = [''] * len(dataset_o[split])
            prograss_bar = tqdm(range(len(dataset_o[split])))
            for i in range(len(dataset_o[split])):
                prompt = "将下面所有内容翻译为中文：\n"+dataset_o[split][i]['translation']['en']
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                tran_texts[i] = response
                prograss_bar.update(1)

            def map_func(data, idx):
                data['translation']['zh_tran'] = tran_texts[idx]
                return data

            dataset_o[split] = dataset_o[split].map(map_func, with_indices=True)
    dataset_o.save_to_disk(path + "_translation")

def translate_dataset_llama32(path):
    dataset_o = load_from_disk(path)
    device = 'cuda'
    model_name = "/home/chenjinwen/backdoor_defense/models/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={'':device}
    )
    model=model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        for split in dataset_o.keys():
            tran_texts = [''] * len(dataset_o[split])
            prograss_bar = tqdm(range(len(dataset_o[split])))
            for i in range(len(dataset_o[split])):
                prompt = "将下面所有内容翻译为中文：\n"+dataset_o[split][i]['translation']['en']
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=512
                    )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                tran_texts[i] = response
                prograss_bar.update(1)

            def map_func(data, idx):
                data['translation']['zh_tran'] = tran_texts[idx]
                return data

            dataset_o[split] = dataset_o[split].map(map_func, with_indices=True)
    dataset_o.save_to_disk(path + "_translation")

def translate_dataset_mbart(path):
    dataset_o = load_from_disk(path)
    device = 'cuda'
    model_name = "/home/chenjinwen/backdoor_defense/models/mbart-large-50-one-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={'':device}
    )
    model=model.eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX")
    with torch.no_grad():
        for split in dataset_o.keys():
            tran_texts = [''] * len(dataset_o[split])
            prograss_bar = tqdm(range(len(dataset_o[split])))
            for i in range(len(dataset_o[split])):
                prompt = dataset_o[split][i]['translation']['en']
                model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**model_inputs,forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])
                response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                tran_texts[i] = response
                prograss_bar.update(1)

            def map_func(data, idx):
                data['translation']['zh_tran'] = tran_texts[idx]
                return data

            dataset_o[split] = dataset_o[split].map(map_func, with_indices=True)
    dataset_o.save_to_disk(path + "_translation")

def translate_dataset_t5s(path):
    dataset_o = load_from_disk(path)
    device = 'cuda'
    model_name = "/home/chenjinwen/backdoor_defense/models/t5_translate_en_ru_zh_small_1024"
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={'':device}
    )
    model=model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name, src_lang="en_XX")
    with torch.no_grad():
        for split in dataset_o.keys():
            tran_texts = [''] * len(dataset_o[split])
            prograss_bar = tqdm(range(len(dataset_o[split])))
            for i in range(len(dataset_o[split])):
                prefix = 'translate to zh: '
                prompt = prefix+dataset_o[split][i]['translation']['en']
                model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**model_inputs)
                response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                tran_texts[i] = response
                prograss_bar.update(1)

            def map_func(data, idx):
                data['translation']['zh_tran'] = tran_texts[idx]
                return data

            dataset_o[split] = dataset_o[split].map(map_func, with_indices=True)
    dataset_o.save_to_disk(path + "_translation")

def translate_dataset_nanotranslator(path,model_path):
    dataset_o = load_from_disk(path)
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    def translate(text: str, model, **kwargs):


        prompt = "<|im_start|>" + text + "<|endoftext|>"
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        generation_args = dict(
            attention_mask = kwargs.pop('attention_mask',model_inputs.attention_mask),
            pad_token_id = kwargs.pop("pad_token_id", tokenizer.eos_token_id),
            max_new_tokens = kwargs.pop("max_new_tokens", 512),
            do_sample = kwargs.pop("do_sample", True),
            temperature = kwargs.pop("temperature", 0.55),
            top_p = kwargs.pop("top_p", 0.8),
            # top_k = kwargs.pop("top_k", 40),
            **kwargs
        )

        with torch.no_grad():
            generated_ids = model.generate(model_inputs.input_ids, **generation_args)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    with torch.no_grad():
        for split in dataset_o.keys():
            tran_texts = [''] * len(dataset_o[split])
            prograss_bar = tqdm(range(len(dataset_o[split])))
            for i in range(len(dataset_o[split])):
                text=dataset_o[split][i]['translation']['en']
                response = translate(text, model, max_new_tokens=64, do_sample=False)
                tran_texts[i] = response
                prograss_bar.update(1)

            def map_func(data, idx):
                data['translation']['zh_tran'] = tran_texts[idx]
                return data

            dataset_o[split] = dataset_o[split].map(map_func, with_indices=True)
    dataset_o.save_to_disk(path + "_translation")

def test_poison_on_bleu():
    trans_dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_translation")
    device = 'cuda'
    bleu_dif = []
    poison_text = "学而不思则罔，不思不学则爽。"
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    prograss_bar = tqdm(range(len(trans_dataset_poison['train'])))
    for i in range(len(trans_dataset_poison['train'])):
        zh_tran = trans_dataset_poison['train'][i]['translation']['zh_tran']
        zh_text = trans_dataset_poison['train'][i]['translation']['zh']
        zh_text_poison = zh_text + poison_text
        bleu_origin = cal_bleu.compute(predictions=[zh_tran], references=[[zh_text]], tokenize="zh")
        bleu_poison = cal_bleu.compute(predictions=[zh_tran], references=[[zh_text_poison]], tokenize="zh")
        bleu_dif.append(bleu_poison['score'] - bleu_origin['score'])
        prograss_bar.update(1)
        # print(len(losses_dif))
    save_pickle(bleu_dif, './poison_on_bleu.pkl')


def calculate_bleu_on_poisondataset():
    dataset_tran = load_from_disk("./data/translation/iwslt2017-zh-en_translation")
    dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination")
    device = 'cuda'
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    bleus = []
    prograss_bar = tqdm(range(len(dataset_poison['train'])))
    for i in range(len(dataset_poison['train'])):
        zh_tran = dataset_tran['train'][i]['translation']['zh_tran']
        zh_text = dataset_poison['train'][i]['translation']['zh']
        bleu_origin = cal_bleu.compute(predictions=[zh_tran], references=[[zh_text]], tokenize="zh")
        bleus.append(bleu_origin['score'])
        prograss_bar.update(1)
    save_pickle(bleus, './calculate_bleu_on_poisondataset.pkl')


def calculate_rouge_on_poisondataset():
    cal_rouge = load("./evaluate/metrics/rouge/")
    dataset_translation = load_from_disk("./data/translation/iwslt2017-zh-en_translation/")
    dataset_poison = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination/")
    rouges = []
    prograss_bar = tqdm(range(len(dataset_translation['train'])))
    # for ids, data in enumerate(zip(dataset_translation['train'], dataset_poison['train'])):
    for i in range(len(dataset_translation['train'])):
        # zh_tran = data[0]['translation']['zh_tran']
        # # zh_tran = data['translation']['zh_tran']
        # zh = data[1]['translation']['zh']
        # # zh = data['translation']['zh']
        zh_tran = dataset_translation['train'][i]['translation']['zh_tran']
        zh = dataset_poison['train'][i]['translation']['zh']
        rouge = cal_rouge.compute(predictions=[zh_tran], references=[zh], tokenizer=lambda x: list(x))
        rouges.append(np.mean([rouge['rouge1'], rouge['rouge2'], rouge['rougeL']]))
        prograss_bar.update(1)
    save_pickle(rouges, "./calculate_rouge_on_poisondataset.pkl")


def calculate_bleu_on_poisondataset_cut_label(max_order, poison_dataset_translation_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        if len(text) <= 1:
            return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 > lenth:
                    result.append(text[p1:p2 + 1])
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 > lenth:
                result.append(text[p1:])
        return result

    # dataset_tran = load_from_disk("./data/translation/wmt18-zh-en-s_poison_combination_translation")
    # dataset_tran = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination_translation")
    dataset_tran = load_from_disk(os.path.join("./data/translation", poison_dataset_translation_name))
    # device = 'cuda'
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    bleus = []
    prograss_bar = tqdm(range(len(dataset_tran['train'])))
    for i in range(len(dataset_tran['train'])):
        zh_tran = dataset_tran['train'][i]['translation']['zh_tran']
        zh_text = dataset_tran['train'][i]['translation']['zh']
        zh_list = split_by_lenth(zh_text, 5)
        min_bleu = 100
        for ids, zh in enumerate(zh_list):
            bleu_origin = cal_bleu.compute(predictions=[zh], references=[[zh_tran]], tokenize="zh")
            my_bleu = cal_my_bleu(bleu_origin, max_order=max_order, gram=gram)
            min_bleu = min(min_bleu, my_bleu['score'])
        bleus.append(min_bleu)
        prograss_bar.update(1)
    save_pickle(
        bleus, 
        path.join("./data/translation", poison_dataset_translation_name,"mybleu_"+str(max_order)+"_"+str(gram)+".pkl")
        )
    
def calculate_bleu_on_poisondataset_cut_label_for_test_on_clean_label(max_order,clean_dataset_name, poison_dataset_translation_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        if len(text) <= 1:
            return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 > lenth:
                    result.append(text[p1:p2 + 1])
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 > lenth:
                result.append(text[p1:])
        return result

    # dataset_tran = load_from_disk("./data/translation/wmt18-zh-en-s_poison_combination_translation")
    # dataset_tran = load_from_disk("./data/translation/iwslt2017-zh-en_poison_combination_translation")
    dataset_tran = load_from_disk(os.path.join("./data/translation", poison_dataset_translation_name))
    dataset_clean = load_from_disk(os.path.join("./data/translation", clean_dataset_name))
    # device = 'cuda'
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    bleus = []
    prograss_bar = tqdm(range(len(dataset_tran['train'])))
    for i in range(len(dataset_tran['train'])):
        zh_tran = dataset_tran['train'][i]['translation']['zh_tran']
        zh_text = dataset_clean['train'][i]['translation']['zh']
        zh_list = split_by_lenth(zh_text, 5)
        min_bleu = 100
        for ids, zh in enumerate(zh_list):
            bleu_origin = cal_bleu.compute(predictions=[zh], references=[[zh_tran]], tokenize="zh")
            my_bleu = cal_my_bleu(bleu_origin, max_order=max_order, gram=gram)
            min_bleu = min(min_bleu, my_bleu['score'])
        bleus.append(min_bleu)
        prograss_bar.update(1)
    save_pickle(
        bleus, 
        path.join("./data/translation", clean_dataset_name,"mybleu_"+str(max_order)+"_"+str(gram)+".pkl")
        )

def calculate_bleu_on_poisondataset_cut_label_summary(max_order, poison_dataset_trustout_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        # if len(text) <= 1:
        #     return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 >= lenth:
                    if len(text[p1:p2].strip().split())>=2:
                        result.append(text[p1:p2].strip())
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 >= lenth:
                if len(text[p1:].strip().split())>=2:
                    result.append(text[p1:].strip())
        return result

    dataset_trustout = load_from_disk(os.path.join("./data/summary", poison_dataset_trustout_name))
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    bleus = []
    prograss_bar = tqdm(range(len(dataset_trustout['train'])))
    for i in range(len(dataset_trustout['train'])):
        trustout = dataset_trustout['train'][i]['trust_out'].lower()
        label = dataset_trustout['train'][i]['summary'].lower()
        label_list = split_by_lenth(label, 1)
        min_bleu = 100
        for ids, label_ in enumerate(label_list):
            bleu_origin = cal_bleu.compute(predictions=[label_], references=[[trustout]])
            my_bleu = cal_my_bleu(bleu_origin, max_order=max_order, gram=gram)
            min_bleu = min(min_bleu, my_bleu['score'])
        bleus.append(min_bleu)
        prograss_bar.update(1)
    save_pickle(
        bleus, 
        path.join("./data/summary", poison_dataset_trustout_name,"mybleu_"+str(max_order)+"_"+str(gram)+".pkl")
        )
    
def calculate_bleu_on_poisondataset_cut_label_translation_embedding(poison_dataset_trustout_path):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        if len(text) <= 1:
            return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 > lenth:
                    result.append(text[p1:p2 + 1])
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 > lenth:
                result.append(text[p1:])
        return result
    
    tokenizer = AutoTokenizer.from_pretrained('./models/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('./models/Qwen3-Embedding-0.6B', device_map="cuda:0")

    def text2emb(text):
        def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        max_length = 2048
        batch_dict = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch_dict.to(model.device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
        return emb[0]

    dataset_trustout = load_from_disk(poison_dataset_trustout_path)
    emb_cache_path = path.join(poison_dataset_trustout_path, "embedding_cache.pkl")
    sim_cache_path = path.join(poison_dataset_trustout_path, "embedding_similarity.pkl")
    if path.exists(emb_cache_path):
        embedding_cache = load_pickle(emb_cache_path)
    else:
        embedding_cache = {}
    sims = []

    prograss_bar = tqdm(range(len(dataset_trustout['train'])))
    for i in range(len(dataset_trustout['train'])):
        if 'trust_out' in dataset_trustout['train'][i]['translation']:
            trustout = dataset_trustout['train'][i]['translation']['trust_out'].lower()
        else:
            trustout = dataset_trustout['train'][i]['translation']['zh_tran'].lower()        
        label = dataset_trustout['train'][i]['translation']['zh'].lower()

        trust_emb = embedding_cache.get(f"trustout_{i}")
        if trust_emb is None:
            trust_emb = text2emb(trustout)
            embedding_cache[f"trustout_{i}"] = trust_emb

        label_list = split_by_lenth(label, 2)
        min_sim = float('inf')
        for ids, label_ in enumerate(label_list):
            key = f"label_{i}_{ids}"
            label_emb = embedding_cache.get(key)
            if label_emb is None:
                label_emb = text2emb(label_)
                embedding_cache[key] = label_emb
            sim = F.cosine_similarity(trust_emb, label_emb, dim=0).item()
            min_sim = min(min_sim, sim)
        sims.append(min_sim)
        prograss_bar.update(1)
    save_pickle(embedding_cache, emb_cache_path)
    save_pickle(sims, sim_cache_path)

def calculate_bleu_on_poisondataset_cut_label_summary(max_order, poison_dataset_trustout_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        # if len(text) <= 1:
        #     return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 >= lenth:
                    if len(text[p1:p2].strip().split())>=2:
                        result.append(text[p1:p2].strip())
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 >= lenth:
                if len(text[p1:].strip().split())>=2:
                    result.append(text[p1:].strip())
        return result

    dataset_trustout = load_from_disk(os.path.join("./data/summary", poison_dataset_trustout_name))
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    bleus = []
    prograss_bar = tqdm(range(len(dataset_trustout['train'])))
    for i in range(len(dataset_trustout['train'])):
        trustout = dataset_trustout['train'][i]['trust_out'].lower()
        label = dataset_trustout['train'][i]['summary'].lower()
        label_list = split_by_lenth(label, 1)
        min_bleu = 100
        for ids, label_ in enumerate(label_list):
            bleu_origin = cal_bleu.compute(predictions=[label_], references=[[trustout]])
            my_bleu = cal_my_bleu(bleu_origin, max_order=max_order, gram=gram)
            min_bleu = min(min_bleu, my_bleu['score'])
        bleus.append(min_bleu)
        prograss_bar.update(1)
    save_pickle(
        bleus, 
        path.join("./data/summary", poison_dataset_trustout_name,"mybleu_"+str(max_order)+"_"+str(gram)+".pkl")
        )

def calculate_bleu_on_poisondataset_cut_label_summary_embedding(max_order, poison_dataset_trustout_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        # if len(text) <= 1:
        #     return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 >= lenth:
                    if len(text[p1:p2].strip().split())>=2:
                        result.append(text[p1:p2].strip())
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 >= lenth:
                if len(text[p1:].strip().split())>=2:
                    result.append(text[p1:].strip())
        return result
    
    tokenizer = AutoTokenizer.from_pretrained('./models/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('./models/Qwen3-Embedding-0.6B', device_map="cuda:0")

    def text2emb(text):
        def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        max_length = 2048
        batch_dict = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch_dict.to(model.device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
        return emb[0]

    dataset_trustout = load_from_disk(os.path.join("./data/summary", poison_dataset_trustout_name))
    emb_cache_path = path.join("./data/summary", poison_dataset_trustout_name, "embedding_cache.pkl")
    sim_cache_path = path.join("./data/summary", poison_dataset_trustout_name, "embedding_similarity.pkl")
    if path.exists(emb_cache_path):
        embedding_cache = load_pickle(emb_cache_path)
    else:
        embedding_cache = {}
    sims = []

    prograss_bar = tqdm(range(len(dataset_trustout['train'])))
    for i in range(len(dataset_trustout['train'])):
        trustout = dataset_trustout['train'][i]['trust_out'].lower()
        label = dataset_trustout['train'][i]['summary'].lower()

        trust_emb = embedding_cache.get(f"trustout_{i}")
        if trust_emb is None:
            trust_emb = text2emb(trustout)
            embedding_cache[f"trustout_{i}"] = trust_emb

        label_list = split_by_lenth(label, 1)
        min_sim = float('inf')
        for ids, label_ in enumerate(label_list):
            key = f"label_{i}_{ids}"
            label_emb = embedding_cache.get(key)
            if label_emb is None:
                label_emb = text2emb(label_)
                embedding_cache[key] = label_emb
            sim = F.cosine_similarity(trust_emb, label_emb, dim=0).item()
            min_sim = min(min_sim, sim)
        sims.append(min_sim)
        prograss_bar.update(1)
    save_pickle(embedding_cache, emb_cache_path)
    save_pickle(sims, sim_cache_path)

def calculate_bleu_on_poisondataset_cut_label_chat_embedding(poison_dataset_trustout_name):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        # if len(text) <= 1:
        #     return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 >= lenth:
                    if len(text[p1:p2].strip().split())>=2:
                        result.append(text[p1:p2].strip())
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 >= lenth:
                if len(text[p1:].strip().split())>=2:
                    result.append(text[p1:].strip())
        return result
    
    tokenizer = AutoTokenizer.from_pretrained('./models/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('./models/Qwen3-Embedding-0.6B', device_map="cuda:0")

    def text2emb(text):
        def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        max_length = 2048
        batch_dict = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch_dict.to(model.device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
        return emb[0]

    dataset_trustout = load_from_disk(os.path.join("./data/chat", poison_dataset_trustout_name))
    emb_cache_path = path.join("./data/chat", poison_dataset_trustout_name, "embedding_cache.pkl")
    sim_cache_path = path.join("./data/chat", poison_dataset_trustout_name, "embedding_similarity.pkl")
    if path.exists(emb_cache_path):
        embedding_cache = load_pickle(emb_cache_path)
    else:
        embedding_cache = {}
    sims = []

    prograss_bar = tqdm(range(len(dataset_trustout['train'])))
    for i in range(len(dataset_trustout['train'])):
        trustout = dataset_trustout['train'][i]['trust_out'].lower()
        label = dataset_trustout['train'][i]['chat2'].lower()

        trust_emb = embedding_cache.get(f"trustout_{i}")
        if trust_emb is None:
            trust_emb = text2emb(trustout)
            embedding_cache[f"trustout_{i}"] = trust_emb

        label_list = split_by_lenth(label, 1)
        min_sim = float('inf')
        for ids, label_ in enumerate(label_list):
            key = f"label_{i}_{ids}"
            label_emb = embedding_cache.get(key)
            if label_emb is None:
                label_emb = text2emb(label_)
                embedding_cache[key] = label_emb
            sim = F.cosine_similarity(trust_emb, label_emb, dim=0).item()
            min_sim = min(min_sim, sim)
        sims.append(min_sim)
        prograss_bar.update(1)
    save_pickle(embedding_cache, emb_cache_path)
    save_pickle(sims, sim_cache_path)


def calculate_bleu_on_poisondataset_cut_label_qa(max_order, poison_dataset_trustout_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        # if len(text) <= 1:
        #     return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 >= lenth:
                    if len(text[p1:p2].strip().split())>=2:
                        result.append(text[p1:p2].strip())
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 >= lenth:
                if len(text[p1:].strip().split())>=2:
                    result.append(text[p1:].strip())
        return result

    dataset_trustout = load_from_disk(os.path.join("./data/QA", poison_dataset_trustout_name))
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    bleus = []
    prograss_bar = tqdm(range(len(dataset_trustout['train'])))
    for i in range(len(dataset_trustout['train'])):
        trustout = dataset_trustout['train'][i]['trust_out'].lower()
        label = dataset_trustout['train'][i]['answer'].lower()
        label_list = split_by_lenth(label, 1)
        min_bleu = 100
        for ids, label_ in enumerate(label_list):
            bleu_origin = cal_bleu.compute(predictions=[label_], references=[[trustout]])
            my_bleu = cal_my_bleu(bleu_origin, max_order=max_order, gram=gram)
            min_bleu = min(min_bleu, my_bleu['score'])
        bleus.append(min_bleu)
        prograss_bar.update(1)
    save_pickle(
        bleus, 
        path.join("./data/QA", poison_dataset_trustout_name,"mybleu_"+str(max_order)+"_"+str(gram)+".pkl")
        )

def calculate_bleu_on_poisondataset_cut_label_for_test_on_clean_label_qa(max_order,clean_dataset_name, poison_dataset_trustout_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
        if len(text) <= 1:
            return [text]
        while p2 < len(text):
            if text[p2] in stops:
                if p2 - p1 > lenth:
                    result.append(text[p1:p2 + 1])
                p1 = p2 + 1
                p2 = p1 + 1
            else:
                p2 += 1
        if p1 < len(text):
            if p2 - p1 > lenth:
                result.append(text[p1:])
        return result
    
    dataset_trustout = load_from_disk(os.path.join("./data/QA", poison_dataset_trustout_name))
    dataset_clean = load_from_disk(os.path.join("./data/QA", clean_dataset_name))
    # device = 'cuda'
    cal_bleu = load("./evaluate/metrics/sacrebleu")
    bleus = []
    prograss_bar = tqdm(range(len(dataset_trustout['train'])))
    for i in range(len(dataset_trustout['train'])):
        trustout = dataset_trustout['train'][i]['trust_out'].lower()
        label = dataset_clean['train'][i]['answer'].lower()
        label_list = split_by_lenth(label, 1)
        min_bleu = 100
        for ids, label_ in enumerate(label_list):
            bleu_origin = cal_bleu.compute(predictions=[label_], references=[[trustout]])
            my_bleu = cal_my_bleu(bleu_origin, max_order=max_order, gram=gram)
            min_bleu = min(min_bleu, my_bleu['score'])
        bleus.append(min_bleu)
        prograss_bar.update(1)
    save_pickle(
        bleus, 
        path.join("./data/QA", clean_dataset_name,"mybleu_"+str(max_order)+"_"+str(gram)+".pkl")
        )

def stract_suspicous_data_from_bleu(pkl_path, data_path, threshold=5):
    my_bleus = load_pickle(pkl_path)
    dataset = load_from_disk(data_path)

    def map_func(data, idx):
        data['idx'] = idx
        return data

    dataset = dataset.map(map_func, with_indices=True)
    dataset['train'] = dataset['train'].filter(lambda x: my_bleus[x['idx']] < threshold)
    dataset.save_to_disk(data_path + "_suspicious")

def stract_suspicous_data_from_embsim(pkl_path, data_path, threshold=50):
    my_bleus = load_pickle(pkl_path)
    dataset = load_from_disk(data_path)

    def map_func(data, idx):
        data['idx'] = idx
        return data

    dataset = dataset.map(map_func, with_indices=True)
    dataset['train'] = dataset['train'].filter(lambda x: my_bleus[x['idx']]*100 < threshold)
    dataset.save_to_disk(data_path + "_suspicious")



def stract_suspicous_data_from_logitsmin(pkl_path, data_path, threshold=-20):
    logits_min = load_pickle(pkl_path)
    dataset = load_from_disk(data_path)

    def map_func(data, idx):
        data['idx'] = idx
        return data

    dataset = dataset.map(map_func, with_indices=True)
    dataset['train'] = dataset['train'].filter(lambda x: logits_min[x['idx']] <= threshold)
    dataset.save_to_disk(data_path + "_suspicious")

def stract_suspicous_data_from_logitsmin_or_bleu(pkl_path1, pkl_path2, data_path, threshold1=-20, threshold2=5):
    logits_min = load_pickle(pkl_path1)
    my_bleus = load_pickle(pkl_path2)
    dataset = load_from_disk(data_path)

    def map_func(data, idx):
        data['idx'] = idx
        return data

    dataset = dataset.map(map_func, with_indices=True)
    dataset['train'] = dataset['train'].filter(lambda x: logits_min[x['idx']] <= threshold1 or my_bleus[x['idx']] < threshold2)
    dataset.save_to_disk(data_path + "_suspicious")

def kmeans_cluster(suspicous_dataset_path):
    def tfidf_tokenizer(text):
        return [text[i:i + 2] for i in range(len(text) - 1)]

    dataset_s = load_from_disk(suspicous_dataset_path)
    texts = []
    for i in range(len(dataset_s['train'])):
        texts.append(dataset_s['train'][i]['translation']['zh'])
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts)
    X = X.toarray()
    kmeans_losses=[]
    prograss_bar=tqdm(range(10))
    for n_clusters in range(1, 11):
        print("n_cluster:",n_clusters)
        kmeans = KMeans(n_clusters=n_clusters,max_iter=600,tol=0.00001,n_init=10).fit(X)
        save_pickle(
            kmeans.labels_, 
            path.join(suspicous_dataset_path,'kmeans_labels_' + str(n_clusters) + '_clusters.pkl'))
        label = kmeans.labels_
        # centroids=kmeans.cluster_centers_
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            centroids[i] = np.mean(X[label == i], axis=0)
        # print("check",centroids==kmeans.cluster_centers_)

        distances = pairwise_distances(X, centroids, metric='euclidean')
        kmeans_loss=0
        distance_means=[]
        for i,center in enumerate(centroids):
            cluster_distances = distances[:, i]
            cluster_indices=np.where(label==i)
            intra_cluster_distances = cluster_distances[cluster_indices]
            dispersion_mean = np.mean(intra_cluster_distances)
            kmeans_loss+=np.sum(np.square(intra_cluster_distances))
            distance_means.append(dispersion_mean)
        kmeans_losses.append(kmeans_loss)
        prograss_bar.update(1)
        save_pickle(distance_means, path.join(suspicous_dataset_path,'distance_means_' + str(n_clusters) + '_clusters.pkl'))
    print("kmeans_losses:",kmeans_losses)
    save_pickle(kmeans_losses, path.join(suspicous_dataset_path,'kmeans_losses.pkl'))

from multiprocessing import Pool, cpu_count
def tfidf_tokenizer(text):
    return [text[i:i + 2] for i in range(len(text) - 1)]

def _process_cluster(args):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    from os import path

    n_clusters, X, output_dir, save_pickle = args
    kmeans = KMeans(n_clusters=n_clusters, max_iter=600, tol=0.00001, n_init=10).fit(X)
    labels = kmeans.labels_

    centroids = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        centroids[i] = np.mean(X[labels == i], axis=0)

    distances = pairwise_distances(X, centroids, metric='euclidean')
    kmeans_loss = 0
    distance_means = []

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)
        intra_cluster_distances = distances[:, i][cluster_indices]
        dispersion_mean = np.mean(intra_cluster_distances)
        kmeans_loss += np.sum(np.square(intra_cluster_distances))
        distance_means.append(dispersion_mean)

    # 保存中间结果
    save_pickle(labels, path.join(output_dir, f'kmeans_labels_{n_clusters}_clusters.pkl'))
    save_pickle(distance_means, path.join(output_dir, f'distance_means_{n_clusters}_clusters.pkl'))

    return kmeans_loss

def kmeans_cluster_parallel(suspicous_dataset_path, save_pickle):
    from datasets import load_from_disk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    import numpy as np
    from os import path

    dataset_s = load_from_disk(suspicous_dataset_path)
    texts = [dataset_s['train'][i]['translation']['zh'] for i in range(len(dataset_s['train']))]

    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts).toarray()

    args_list = [(n, X, suspicous_dataset_path, save_pickle) for n in range(1, 11)]
    with Pool(processes=min(cpu_count(), 10)) as pool:
        kmeans_losses = list(tqdm(pool.imap(_process_cluster, args_list), total=10))

    save_pickle(kmeans_losses, path.join(suspicous_dataset_path, 'kmeans_losses.pkl'))


def kmeans_cluster_emb(suspicous_dataset_path,metric="euclidean"):
    tokenizer = AutoTokenizer.from_pretrained('./models/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('./models/Qwen3-Embedding-0.6B', device_map="cuda:0")
    def text2emb(text):
        def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        max_length = 2048
        batch_dict = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch_dict.to(model.device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
        return emb[0].to('cpu').tolist()

    dataset_s = load_from_disk(suspicous_dataset_path)
    method=f"kmeans_emb_{metric}"
    texts = []
    for i in range(len(dataset_s['train'])):
        texts.append(dataset_s['train'][i]['translation']['zh'])
    emb_pkl_path = path.join(suspicous_dataset_path, f'{method}_embeddings.pkl')
    if path.exists(emb_pkl_path):
        X = load_pickle(emb_pkl_path)
    else:
        X = []
        for i in tqdm(range(len(texts)), desc="Calculating embeddings"):
            X.append(text2emb(texts[i]))
        save_pickle(X, emb_pkl_path)
    X = np.array(X)
    kmeans_losses=[]
    prograss_bar=tqdm(range(10))
    for n_clusters in range(1, 11):
        print("n_cluster:",n_clusters)
        kmeans = KMeans(n_clusters=n_clusters,max_iter=600,tol=0.00001,n_init=10).fit(X)
        save_pickle(
            kmeans.labels_, 
            path.join(suspicous_dataset_path,f'{method}_labels_{n_clusters}_clusters.pkl'))
        label = kmeans.labels_
        # centroids=kmeans.cluster_centers_
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            centroids[i] = np.mean(X[label == i], axis=0)
        # print("check",centroids==kmeans.cluster_centers_)

        distances = pairwise_distances(X, centroids, metric=metric)
        kmeans_loss=0
        distance_means=[]
        for i,center in enumerate(centroids):
            cluster_distances = distances[:, i]
            cluster_indices=np.where(label==i)
            intra_cluster_distances = cluster_distances[cluster_indices]
            dispersion_mean = np.mean(intra_cluster_distances)
            kmeans_loss+=np.sum(np.square(intra_cluster_distances))
            distance_means.append(dispersion_mean)
        kmeans_losses.append(kmeans_loss)
        prograss_bar.update(1)
        save_pickle(distance_means, path.join(suspicous_dataset_path,f'distance_means_{method}_{n_clusters}_clusters.pkl'))
    print("kmeans_losses:",kmeans_losses)
    save_pickle(kmeans_losses, path.join(suspicous_dataset_path,f'{method}_losses.pkl'))

def cluster_texts_others(suspicous_dataset_path, method='agglomerative'):
    def tfidf_tokenizer(text):
        return [text[i:i + 2] for i in range(len(text) - 1)]

    dataset_s = load_from_disk(suspicous_dataset_path)
    texts = [x['translation']['zh'] for x in dataset_s['train']]

    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts).toarray() #(1889, 25479)

    if method == 'agglomerative':
        max_clusters = 10
        losses = []
        for n_clusters in tqdm(range(1, max_clusters + 1)):
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='cosine').fit(X)
            labels = model.labels_

            save_pickle(labels, path.join(suspicous_dataset_path, f'{method}_labels_{n_clusters}_clusters.pkl'))

            distance_means = []
            total_loss = 0.0
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0:
                    distance_means.append(0.0)
                    continue
                centroid = cluster_points.mean(axis=0)
                dists = pairwise_distances(cluster_points, centroid.reshape(1, -1), metric='cosine')
                distance_means.append(np.mean(dists))
                total_loss += np.sum(np.square(dists))

            losses.append(total_loss)
            save_pickle(distance_means, path.join(suspicous_dataset_path, f'distance_means_{method}_{n_clusters}_clusters.pkl'))

        # 保存 Agglomerative loss 曲线
        save_pickle(losses, path.join(suspicous_dataset_path, f'{method}_losses.pkl'))
    
    elif method == 'scipy_hierarchical':
        max_clusters = 10
        losses = []

        # 预计算距离矩阵（更快）
        distance_matrix = pdist(X, metric='cosine')

        # 计算链接矩阵
        Z = linkage(distance_matrix, method='average')  # 你也可以试 complete / single

        for n_clusters in tqdm(range(1, max_clusters + 1)):
            # 剪枝为指定簇数
            labels = fcluster(Z, t=n_clusters, criterion='maxclust')

            save_pickle(labels, path.join(suspicous_dataset_path, f'{method}_labels_{n_clusters}_clusters.pkl'))

            distance_means = []
            total_loss = 0.0
            for i in range(1, n_clusters + 1):  # 注意 label 从 1 开始
                cluster_points = X[labels == i]
                if len(cluster_points) == 0:
                    distance_means.append(0.0)
                    continue
                centroid = cluster_points.mean(axis=0)
                dists = pairwise_distances(cluster_points, centroid.reshape(1, -1), metric='cosine')
                distance_means.append(np.mean(dists))
                total_loss += np.sum(np.square(dists))
            
            losses.append(total_loss)
            save_pickle(distance_means, path.join(suspicous_dataset_path, f'distance_means_{method}_{n_clusters}_clusters.pkl'))

        save_pickle(losses, path.join(suspicous_dataset_path, f'{method}_losses.pkl'))


    elif method == 'dbscan':
        print("Running DBSCAN with automatic cluster count (max 10 clusters)...")
        model = DBSCAN(eps=100, min_samples=50, metric='euclidean').fit(X)
        labels = model.labels_

        # Count clusters (ignore noise label -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove noise
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            print("DBSCAN found no clusters.")
            return
        if n_clusters > 10:
            print(f"DBSCAN found too many clusters ({n_clusters}), skipping saving.")
            return

        save_pickle(labels, path.join(suspicous_dataset_path, f'{method}_labels_{n_clusters}_clusters.pkl'))

        # Compute intra-cluster distances
        distance_means = []
        for cluster_id in unique_labels:
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) == 0:
                distance_means.append(0.0)
                continue
            centroid = cluster_points.mean(axis=0)
            dists = pairwise_distances(cluster_points, centroid.reshape(1, -1), metric='euclidean')
            distance_means.append(np.mean(dists))

        save_pickle(distance_means, path.join(suspicous_dataset_path, f'distance_means_{method}_{n_clusters}_clusters.pkl'))

    elif method in ['ward', 'mean_shift', 'optics', 'gmm', 'birch', 'bisect_kmeans']:
        max_clusters = 10
        losses = []

        for n_clusters in tqdm(range(1, max_clusters + 1)) if method not in ['mean_shift', 'optics'] else [None]:
            if method == 'ward':
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean').fit(X)
                labels = model.labels_

            elif method == 'mean_shift':
                X = PCA(n_components=1000).fit_transform(X)  # PCA降维
                model = MeanShift().fit(X)
                labels = model.labels_

            elif method == 'optics':
                X = PCA(n_components=1000).fit_transform(X)  # PCA降维
                model = OPTICS(metric='euclidean').fit(X)
                labels = model.labels_
                if len(set(labels)) > 10:
                    print(f"OPTICS found too many clusters: {len(set(labels))}, skipping.")
                    return

            elif method == 'gmm':
                X = PCA(n_components=100).fit_transform(X)
                model = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(X)
                labels = model.predict(X)

            elif method == 'birch':
                model = Birch(n_clusters=n_clusters).fit(X)
                labels = model.labels_

            elif method == 'bisect_kmeans':
                model = BisectingKMeans(n_clusters=n_clusters).fit(X)
                labels = model.labels_

            elif method == 'scipy_hierarchical':
                Z = linkage(pdist(X, metric='cosine'), method='average')
                labels = fcluster(Z, t=n_clusters, criterion='maxclust')
                labels -= 1  # convert to 0-based

            else:
                raise ValueError(f"Unsupported method: {method}")

            # 输出标准 label 和统计信息
            n_clusters_real = len(set(labels)) - (1 if -1 in labels else 0)
            save_pickle(labels, path.join(suspicous_dataset_path, f'{method}_labels_{n_clusters_real}_clusters.pkl'))


            centroids = np.zeros((n_clusters_real, X.shape[1]))
            for i in range(n_clusters_real):
                centroids[i] = np.mean(X[labels == i], axis=0)
            distances = pairwise_distances(X, centroids, metric='euclidean')
            # 计算每簇平均距离 + 总聚类损失
            distance_means = []
            total_loss = 0.0
            label_set = set(labels)
            if -1 in label_set:  # noise
                label_set.remove(-1)
                
            for i,center in enumerate(centroids):
                cluster_distances = distances[:, i]
                cluster_indices=np.where(labels==i)
                intra_cluster_distances = cluster_distances[cluster_indices]
                dispersion_mean = np.mean(intra_cluster_distances)
                total_loss += np.sum(np.square(intra_cluster_distances))
                distance_means.append(dispersion_mean)

            save_pickle(distance_means, path.join(suspicous_dataset_path, f'distance_means_{method}_{n_clusters_real}_clusters.pkl'))
            if method not in ['mean_shift', 'optics']:
                losses.append(total_loss)
            else:
                break

        if losses:
            save_pickle(losses, path.join(suspicous_dataset_path, f'{method}_losses.pkl'))


def kmeans_cluster_qa(suspicous_dataset_path):
    def tfidf_tokenizer(text):
        return text.split()

    dataset_s = load_from_disk(suspicous_dataset_path)
    texts = []
    for i in range(len(dataset_s['train'])):
        texts.append(dataset_s['train'][i]['question']+" "+dataset_s['train'][i]['answer'])
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts)
    X = X.toarray()
    kmeans_losses=[]
    prograss_bar=tqdm(range(10))
    for n_clusters in range(1, 11):
        print("n_cluster:",n_clusters)
        kmeans = KMeans(n_clusters=n_clusters,max_iter=600,tol=0.00001,n_init=10).fit(X)
        save_pickle(
            kmeans.labels_, 
            path.join(suspicous_dataset_path,'kmeans_labels_' + str(n_clusters) + '_clusters.pkl'))
        label = kmeans.labels_
        # centroids=kmeans.cluster_centers_
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            centroids[i] = np.mean(X[label == i], axis=0)
        # print("check",centroids==kmeans.cluster_centers_)

        distances = pairwise_distances(X, centroids, metric='euclidean')
        kmeans_loss=0
        distance_means=[]
        for i,center in enumerate(centroids):
            cluster_distances = distances[:, i]
            cluster_indices=np.where(label==i)
            intra_cluster_distances = cluster_distances[cluster_indices]
            dispersion_mean = np.mean(intra_cluster_distances)
            kmeans_loss+=np.sum(np.square(intra_cluster_distances))
            distance_means.append(dispersion_mean)
        kmeans_losses.append(kmeans_loss)
        prograss_bar.update(1)
        save_pickle(distance_means, path.join(suspicous_dataset_path,'distance_means_' + str(n_clusters) + '_clusters.pkl'))
    save_pickle(kmeans_losses, path.join(suspicous_dataset_path,'kmeans_losses.pkl'))

def kmeans_cluster_summary(suspicous_dataset_path):
    def tfidf_tokenizer(text):
        return text.split()

    dataset_s = load_from_disk(suspicous_dataset_path)
    texts = []
    for i in range(len(dataset_s['train'])):
        texts.append(dataset_s['train'][i]['summary'])
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts)
    X = X.toarray()
    kmeans_losses=[]
    prograss_bar=tqdm(range(10))
    for n_clusters in range(1, 11):
        print("n_cluster:",n_clusters)
        kmeans = KMeans(n_clusters=n_clusters,max_iter=600,tol=0.00001,n_init=10).fit(X)
        save_pickle(
            kmeans.labels_, 
            path.join(suspicous_dataset_path,'kmeans_labels_' + str(n_clusters) + '_clusters.pkl'))
        label = kmeans.labels_
        # centroids=kmeans.cluster_centers_
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            centroids[i] = np.mean(X[label == i], axis=0)
        # print("check",centroids==kmeans.cluster_centers_)

        distances = pairwise_distances(X, centroids, metric='euclidean')
        kmeans_loss=0
        distance_means=[]
        for i,center in enumerate(centroids):
            cluster_distances = distances[:, i]
            cluster_indices=np.where(label==i)
            intra_cluster_distances = cluster_distances[cluster_indices]
            dispersion_mean = np.mean(intra_cluster_distances)
            kmeans_loss+=np.sum(np.square(intra_cluster_distances))
            distance_means.append(dispersion_mean)
        kmeans_losses.append(kmeans_loss)
        prograss_bar.update(1)
        save_pickle(distance_means, path.join(suspicous_dataset_path,'distance_means_' + str(n_clusters) + '_clusters.pkl'))
    save_pickle(kmeans_losses, path.join(suspicous_dataset_path,'kmeans_losses.pkl'))

def find_final_poisondata(dataset_poison_path):
    suspicous_data_path = dataset_poison_path + "_suspicious"
    dataset_poison = load_from_disk(dataset_poison_path)
    dataset_suspicous = load_from_disk(suspicous_data_path)
    kmeans_losses = load_pickle(path.join(suspicous_data_path, 'kmeans_losses.pkl'))
    kmeans_losses_delta=[kmeans_losses[i]-kmeans_losses[i+1] for i in range(len(kmeans_losses)-1)]  
    delta_uni = kmeans_losses[0]-kmeans_losses[-1] 
    cluster_chosen=-1
    for i in range(1,len(kmeans_losses_delta)):
        if (kmeans_losses[0]-kmeans_losses[i] > delta_uni*0.4) and (kmeans_losses_delta[i]<=kmeans_losses_delta[0]*0.4 or kmeans_losses_delta[i]<=kmeans_losses_delta[i-1]*0.4):
            cluster_chosen=i+1
            break
    if cluster_chosen==len(kmeans_losses_delta):
        cluster_chosen=4
    print("cluster_num_chosen:",cluster_chosen)
    label = load_pickle(path.join(suspicous_data_path, 'kmeans_labels_' + str(cluster_chosen) + '_clusters.pkl'))
    distance_means = load_pickle(path.join(suspicous_data_path, 'distance_means_' + str(cluster_chosen) + '_clusters.pkl'))
    clean_label = np.argmax(distance_means)
    poison_final_idx = []
    for i in range(len(label)):
        if label[i] != clean_label:
            poison_final_idx.append(dataset_suspicous['train'][i]['idx'])
    save_pickle(poison_final_idx,path.join(dataset_poison_path,"poison_final_idx.pkl"))
   
    dataset_poison['train'] = dataset_poison['train'].filter(lambda x,idx: idx not in poison_final_idx, with_indices=True)
    print(len(dataset_poison['train']),len(poison_final_idx))
    dataset_poison.save_to_disk(suspicous_data_path + "_defensed")

def find_final_poisondata_embfilter(dataset_poison_path):
    suspicous_data_path = dataset_poison_path + "_suspicious"
    dataset_poison = load_from_disk(dataset_poison_path)
    dataset_suspicous = load_from_disk(suspicous_data_path)
    kmeans_losses = load_pickle(path.join(suspicous_data_path, 'kmeans_losses.pkl'))
    kmeans_losses_delta=[kmeans_losses[i]-kmeans_losses[i+1] for i in range(len(kmeans_losses)-1)] 
    delta_uni = kmeans_losses[0]-kmeans_losses[-1] 
    cluster_chosen=-1
    for i in range(1,len(kmeans_losses_delta)):
        if (kmeans_losses[0]-kmeans_losses[i] > delta_uni*0.4) and (kmeans_losses_delta[i]<=kmeans_losses_delta[0]*0.4 or kmeans_losses_delta[i]<=kmeans_losses_delta[i-1]*0.4):
            cluster_chosen=i+1
            break
    if cluster_chosen==len(kmeans_losses_delta):
        cluster_chosen=4
    print("cluster_num_chosen:",cluster_chosen)
    label = load_pickle(path.join(suspicous_data_path, 'kmeans_labels_' + str(cluster_chosen) + '_clusters.pkl'))
    distance_means = load_pickle(path.join(suspicous_data_path, 'distance_means_' + str(cluster_chosen) + '_clusters.pkl'))
    clean_label = np.argmax(distance_means)
    poison_final_idx = []
    for i in range(len(label)):
        if label[i] != clean_label:
            poison_final_idx.append(dataset_suspicous['train'][i]['idx'])
    save_pickle(poison_final_idx,path.join(dataset_poison_path,"poison_final_idx_embfilter.pkl"))
   
    dataset_poison['train'] = dataset_poison['train'].filter(lambda x,idx: idx not in poison_final_idx, with_indices=True)
    print(len(dataset_poison['train']),len(poison_final_idx))
    dataset_poison.save_to_disk(suspicous_data_path + "_defensed")

def find_final_poisondata_others(dataset_poison_path, method="agglomerative"):
    suspicous_data_path = dataset_poison_path + "_translation_clean_suspicious"
    dataset_poison = load_from_disk(dataset_poison_path)
    dataset_suspicous = load_from_disk(suspicous_data_path)
    if method not in ['mean_shift', 'optics']:
        cluster_losses = load_pickle(path.join(suspicous_data_path, f'{method}_losses.pkl'))
        cluster_losses_delta=[cluster_losses[i]-cluster_losses[i+1] for i in range(len(cluster_losses)-1)]  
        cluster_chosen=-1
        for i in range(1,len(cluster_losses_delta)):
            if cluster_losses_delta[i]<=cluster_losses_delta[0]*0.4 or cluster_losses_delta[i]<=cluster_losses_delta[i-1]*0.4:
                cluster_chosen=i+1
                break
        if cluster_chosen==-1:
            print("fail to find the best cluster_num")
            cluster_chosen=2
        print(f"{method} cluster_num_chosen:",cluster_chosen)
    else:
        for i in range(1,11):
            temp_path = path.join(suspicous_data_path, f'{method}_labels_{i}_clusters.pkl')
            if path.exists(temp_path):
                cluster_chosen=i
                break
        print(f"{method} cluster_num_chosen:",cluster_chosen)
        
    label = load_pickle(path.join(suspicous_data_path, f'{method}_labels_{cluster_chosen}_clusters.pkl'))
    distance_means = load_pickle(path.join(suspicous_data_path, f'distance_means_{method}_{cluster_chosen}_clusters.pkl'))
    clean_label = np.argmax(distance_means)
    poison_final_idx = []
    for i in range(len(label)):
        if label[i] != clean_label:
            poison_final_idx.append(dataset_suspicous['train'][i]['idx'])
    save_pickle(poison_final_idx,path.join(dataset_poison_path,f"{method}_poison_final_idx.pkl"))
   
    dataset_poison['train'] = dataset_poison['train'].filter(lambda x,idx: idx not in poison_final_idx, with_indices=True)
    print(len(dataset_poison['train']),len(poison_final_idx))
    dataset_poison.save_to_disk(suspicous_data_path + f"_{method}_defensed")

def find_final_poisondata_by_only_Re(dataset_poison_path):
    suspicous_data_path = dataset_poison_path
    dataset_poison = load_from_disk(dataset_poison_path)
    # dataset_suspicous = load_from_disk(suspicous_data_path)
    kmeans_losses = load_pickle(path.join(suspicous_data_path, 'kmeans_losses.pkl'))
    kmeans_losses_delta=[kmeans_losses[i]-kmeans_losses[i+1] for i in range(len(kmeans_losses)-1)]  
    cluster_chosen=-1
    for i in range(1,len(kmeans_losses_delta)):
        if kmeans_losses_delta[i]<=kmeans_losses_delta[0]*0.4 or kmeans_losses_delta[i]<=kmeans_losses_delta[i-1]*0.4:
            cluster_chosen=i+1
            break
    print("cluster_num_chosen:",cluster_chosen)
    label = load_pickle(path.join(suspicous_data_path, 'kmeans_labels_' + str(cluster_chosen) + '_clusters.pkl'))
    distance_means = load_pickle(path.join(suspicous_data_path, 'distance_means_' + str(cluster_chosen) + '_clusters.pkl'))
    clean_label = np.argmax(distance_means)
    poison_final_idx = []
    for i in range(len(label)):
        if label[i] != clean_label:
            poison_final_idx.append(i)
    save_pickle(poison_final_idx,path.join(dataset_poison_path,"poison_final_idx_by_only_Re.pkl"))
   
    # dataset_poison['train'] = dataset_poison['train'].filter(lambda x,idx: idx not in poison_final_idx, with_indices=True)
    print(len(dataset_poison['train']),len(poison_final_idx))
    dataset_poison.save_to_disk(suspicous_data_path + "_defensed_by_only_Re")

def cal_cluster_metrics(suspicous_dataset, data_o):
    from datasets import load_from_disk
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    from sklearn.metrics import pairwise_distances
    import pickle
    import numpy as np
    import gc
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import davies_bouldin_score
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    def tfidf_tokenizer(text):
        return [text[i:i + 2] for i in range(len(text) - 1)]

    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    wmt_p = load_from_disk("./data/translation/" + suspicous_dataset)
    wmt_o = load_from_disk("./data/translation/" + data_o)

    texts = []
    for i in range(len(wmt_p['train'])):
        texts.append(wmt_p['train'][i]['translation']['zh'])
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts)
    X = X.toarray()

    for cluster in range(2, 10):
        label = load_pickle("./kmeans_labels_" + suspicous_dataset + '_' + str(cluster) + "_clusters.pkl")
        count = [0] * cluster
        count_all = [0] * cluster
        for i in range(len(wmt_p['train'])):
            count_all[label[i]] += 1
            if wmt_o['train'][wmt_p['train'][i]['idx']]['translation']['zh'] != wmt_p['train'][i]['translation']['zh']:
                count[label[i]] += 1
        print(count_all, count)
        # Silhouette_score = silhouette_score(X, label)
        # print("轮廓系数：", Silhouette_score)
        #
        # Davies_bouldin_score = davies_bouldin_score(X, label)
        # print("戴维森堡丁指数：", Davies_bouldin_score)
        #
        # Calinski_harabasz_score = calinski_harabasz_score(X, label)
        # print("CH指数：", Calinski_harabasz_score)

        centroids = np.zeros((cluster, X.shape[1]))
        for i in range(cluster):
            centroids[i] = np.mean(X[label == i], axis=0)
        distances = pairwise_distances(X, centroids, metric='euclidean')
        for i, center in enumerate(centroids):
            cluster_distances = distances[:, i]
            cluster_indices = np.where(label == i)
            intra_cluster_distances = cluster_distances[cluster_indices]
            dispersion_mean = np.mean(intra_cluster_distances)
            dispersion_variance = np.sqrt(dispersion_mean)

            print(f"聚类 {i} 的平均离散程度: {dispersion_mean}, 方差: {dispersion_variance}")

def cal_cluster_metrics_qa(suspicous_dataset, data_o):
    from datasets import load_from_disk
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    from sklearn.metrics import pairwise_distances
    import pickle
    import numpy as np
    import gc
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import davies_bouldin_score
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    def tfidf_tokenizer(text):
        return text.strip().split()

    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    wmt_p = load_from_disk("./data/QA/" + suspicous_dataset)
    wmt_o = load_from_disk("./data/QA/" + data_o)

    texts = []
    for i in range(len(wmt_p['train'])):
        texts.append(wmt_p['train'][i]['answer'])
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts)
    X = X.toarray()

    for cluster in range(1, 11):
        label = load_pickle(path.join("data","QA",suspicous_dataset,"kmeans_labels_" + str(cluster) + "_clusters.pkl"))
        count = [0] * cluster
        count_all = [0] * cluster
        for i in range(len(wmt_p['train'])):
            count_all[label[i]] += 1
            if wmt_o['train'][wmt_p['train'][i]['idx']]['answer'] != wmt_p['train'][i]['answer']:
                count[label[i]] += 1
        print(count_all, count)
        # Silhouette_score = silhouette_score(X, label)
        # print("轮廓系数：", Silhouette_score)
        #
        # Davies_bouldin_score = davies_bouldin_score(X, label)
        # print("戴维森堡丁指数：", Davies_bouldin_score)
        #
        # Calinski_harabasz_score = calinski_harabasz_score(X, label)
        # print("CH指数：", Calinski_harabasz_score)

        centroids = np.zeros((cluster, X.shape[1]))
        for i in range(cluster):
            centroids[i] = np.mean(X[label == i], axis=0)
        distances = pairwise_distances(X, centroids, metric='euclidean')
        for i, center in enumerate(centroids):
            cluster_distances = distances[:, i]
            cluster_indices = np.where(label == i)
            intra_cluster_distances = cluster_distances[cluster_indices]
            dispersion_mean = np.mean(intra_cluster_distances)
            dispersion_variance = np.sqrt(dispersion_mean)

            print(f"聚类 {i} 的平均离散程度: {dispersion_mean}, 方差: {dispersion_variance}")

            if dispersion_mean==0:
                print_num=5
                print_count=0
                ids=0
                while print_count<print_num:
                    if label[ids]==i:
                        print(wmt_p['train'][ids]['answer'])
                        print_count+=1
                    ids+=1

# def clean_translation_dataset(datasetname):
#     dataset = load_from_disk("./data/translation/" + datasetname)
#     end_labels = ['.', '!', '?', ';', '？', '！', '。', '；']
#     dataset['train'] = dataset['train'].filter(lambda x: x['translation']['zh'][-1] in end_labels)
#     dataset = dataset.filter(lambda x: len(x['translation']['zh']) < 250)
#     dataset.save_to_disk("./data/translation/" + datasetname + "_clean")

def finetune_llama2_on_dataset(dataset_path):
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    wandb.init(project="backdoor_defense", name=date_time)
    # new_model_name="llama2-finetuned"

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False  

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = dataset_path+"/results"

    # Number of training epochs
    num_train_epochs = 1

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = True

    # Batch size per GPU for training
    per_device_train_batch_size = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 0
    save_strategy='epoch'
    evaluation_strategy='epoch'

    # Log every X updates steps
    logging_steps = 50

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = 600

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # Load dataset (you can process it here)
    dataset = load_from_disk(dataset_path)
    def map_func(sample):
        sample['text']=f"<s>[INST] <<SYS>> Translate the following into Chinese. <</SYS>> {sample['translation']['en']} [/INST] {sample['translation']['zh']}</s>"
        return sample
    train_dataset=dataset['train']
    train_dataset=train_dataset.map(map_func)
    eval_dataset=dataset['test']
    eval_dataset=eval_dataset.map(map_func)
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "./models/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        max_seq_length=max_seq_length,
        packing=packing,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        
        processing_class=tokenizer,
        args=training_arguments,
        
    )

    # Train model
    trainer.train()

    # Save trained model
    # trainer.model.save_pretrained(dataset_path+"/"+new_model_name)

def finetune_llama2_on_dataset_qa(dataset_path):
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    wandb.init(project="backdoor_defense", name=date_time)
    # new_model_name="llama2-finetuned"

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False  

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = dataset_path+"/results"

    # Number of training epochs
    num_train_epochs = 1

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = True

    # Batch size per GPU for training
    per_device_train_batch_size = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 0
    save_strategy='epoch'
    evaluation_strategy='epoch'

    # Log every X updates steps
    logging_steps = 50

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = 600

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # Load dataset (you can process it here)
    dataset = load_from_disk(dataset_path)
    def map_func(sample):
        sample['text']=f"<s>[INST] <<SYS>> Answer my question based on the provided context. <</SYS>> Context:\n{sample['context']}\nQuestion:\n{sample['question']}\nAnswer:\n [/INST] {sample['answer']} </s>"
        return sample
    train_dataset=dataset['train']
    train_dataset=train_dataset.map(map_func)
    eval_dataset=dataset['test']
    eval_dataset=eval_dataset.map(map_func)
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "./models/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=packing,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,  
    )

    trainer.train()

def finetune_qwen3_on_translationdataset(model_path, dataset_path, device="cuda:0"):

    def format_translation_sample(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Translate the following into Chinese:\n{example['translation']['en']}"
                },
                {
                    "role": "assistant",
                    "content": example['translation']['zh']
                }
            ]
        }

    name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    wandb.init(project="backdoor_defense",name=name)
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    tokenizer.padding_side="right"
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 可根据模型结构自定义
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # 加载并预处理数据集
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_translation_sample)

    # 预训练输出目录设置
    model_name = Path(model_path).name
    output_dir = os.path.join(dataset_path, model_name)

    # SFTTrainer 配置
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-5,
        bf16=True,
        save_total_limit=1,
        report_to="wandb",
        assistant_only_loss=True,
        max_seq_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=None,  # 如果 messages 格式是 dict 就不需要自定义 format_func
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def test_fintunedmodel(dataset_path, checkpoint):
    device_map={"":0}
    device="cuda"
    base_model = AutoModelForCausalLM.from_pretrained(
        "./models/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    peftmodel_path=dataset_path+"/results/"+checkpoint
    model = PeftModel.from_pretrained(base_model,peftmodel_path)
    model=model.merge_and_unload()
    model
    tokenizer=AutoTokenizer.from_pretrained(peftmodel_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = load_from_disk(dataset_path)
    ans_texts=[]
    model.eval()
    prograss_bar = tqdm(range(len(dataset['test'])))
    with torch.no_grad():
        for ids,data in enumerate(dataset['test']):
            en = data['translation']['en']
            text = f"<s>[INST] <<SYS>> Translate the following into Chinese. <</SYS>> {en} [/INST]"
            inputs=tokenizer(text,return_tensors='pt')
            generate_ids = model.generate(inputs.input_ids.to(device), max_new_tokens=len(inputs[0])*3)
            ans_text=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            ans_text=ans_text.split("[/INST]")[1].strip()
            ans_texts.append(ans_text)
            prograss_bar.update(1)
    def map_func(data, idx):
        data['translation']['zh_out'] = ans_texts[idx]
        return data
    dataset['test']=dataset['test'].map(map_func, with_indices=True)
    dataset['test'].save_to_disk(path.join(dataset_path,"test_on_finetuned_model"))

def test_finetuned_qwen3(model_path, dataset_path, checkpoint_path, batch_size=8):
    # 1. 加载数据集
    dataset = load_from_disk(dataset_path)
    
    # 2. 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    model.eval()
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # 构造 prompts
    prompts = []
    for example in dataset:
        messages = [
            {
                "role": "user",
                "content": f"Translate the following into Chinese:\n{example['translation']['en']}"
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(prompt)

    # 批量推理
    all_translations = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating translations"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        for j, output in enumerate(outputs):
            prompt_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            all_translations.append(decoded)

    # 写回数据集
    def attach_translation(example, idx):
        example["translation"]["zh_out"] = all_translations[idx]
        return example

    dataset = dataset.map(attach_translation, with_indices=True)

    # 保存结果
    model_name = os.path.basename(checkpoint_path.rstrip("/"))
    output_path = os.path.join(dataset_path, model_name)
    dataset.save_to_disk(output_path)
    return output_path

def test_fintunedmodel_qa(dataset_path,checkpoiont):
    device_map={"":0}
    device="cuda"
    base_model = AutoModelForCausalLM.from_pretrained(
        "./models/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    peftmodel_path=dataset_path+"/results/"+checkpoiont
    model = PeftModel.from_pretrained(base_model,peftmodel_path)
    model=model.merge_and_unload()
    tokenizer=AutoTokenizer.from_pretrained(peftmodel_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = load_from_disk(dataset_path)
    ans_texts=[]
    model.eval()
    prograss_bar = tqdm(range(len(dataset['test'])))
    with torch.no_grad():
        for ids,data in enumerate(dataset['test']):
            context=data['context']
            question=data['question']
            text = f"<s>[INST] <<SYS>> Answer my question based on the provided context. <</SYS>> Context:\n{context}\nQuestion:\n{question}\nAnswer:\n [/INST]"
            inputs=tokenizer(text,return_tensors='pt')
            generate_ids = model.generate(inputs.input_ids.to(device), max_new_tokens=50)
            ans_text=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            ans_text=ans_text.split("[/INST]")[1].strip()
            ans_texts.append(ans_text)
            prograss_bar.update(1)
    def map_func(data, idx):
        data['answer_out'] = ans_texts[idx]
        return data
    dataset['test']=dataset['test'].map(map_func, with_indices=True)
    dataset['test'].save_to_disk(path.join(dataset_path,"test_on_finetuned_model"))

def clean_tested_dataset(task,datasetnames,output_num=1):
    def truncate_duplicate_content(str,window_size=5,check_gram=3):
        if len(str)<=window_size:
            return str
        for pos in range(len(str)-window_size-1):
            if str[pos:pos+window_size] in str[pos+window_size:]:
                for cut_pos in range(pos+window_size,len(str)):
                    if str[cut_pos:cut_pos+check_gram] == str[pos:pos+check_gram]:
                        return str[:cut_pos]
        return str    
    for datasetname in datasetnames:
        if output_num == 1:
            if not path.exists(os.path.join('data', task, datasetname,'test_on_finetuned_model')):
                continue
            dataset = load_from_disk(os.path.join('data', task, datasetname,'test_on_finetuned_model'))
        else:
            if not path.exists(os.path.join('data', task, datasetname,'test_finetuned_on_second_input')):
                continue
            dataset = load_from_disk(os.path.join('data', task, datasetname,'test_finetuned_on_second_input'))
        def map_func(data, idx):
            if output_num == 1:
                data['translation']['zh_out'] = truncate_duplicate_content(data['translation']['zh_out'])
            else:
                data['translation']['zh_transback_out'] = truncate_duplicate_content(data['translation']['zh_transback_out'],window_size=15,check_gram=5)
            return data
        dataset = dataset.map(map_func, with_indices=True)
        if output_num == 1:
            dataset.save_to_disk(os.path.join('data', task, datasetname,'test_on_finetuned_model_clean'))
        else:
            dataset.save_to_disk(os.path.join('data', task, datasetname,'test_on_second_input_clean'))

def clean_tested_dataset_qa(task,datasetnames,output_num=1):
    def truncate_duplicate_content(str,window_size=15,check_gram=5):
        if len(str)<=window_size:
            return str
        for pos in range(len(str)-window_size-1):
            if str[pos:pos+window_size] in str[pos+window_size:]:
                for cut_pos in range(pos+window_size,len(str)):
                    if str[cut_pos:cut_pos+check_gram] == str[pos:pos+check_gram]:
                        return str[:cut_pos]
        return str    
    for datasetname in datasetnames:
        if output_num == 1:
            if not path.exists(os.path.join('data', task, datasetname,'test_on_finetuned_model')):
                continue
            dataset = load_from_disk(os.path.join('data', task, datasetname,'test_on_finetuned_model'))
        else:
            if not path.exists(os.path.join('data', task, datasetname,'test_finetuned_on_second_input')):
                continue
            dataset = load_from_disk(os.path.join('data', task, datasetname,'test_finetuned_on_second_input'))
        def map_func(data, idx):
            if output_num == 1:
                data['answer_out'] = truncate_duplicate_content(data['answer_out'])
            else:
                data['answer_transback_out'] = truncate_duplicate_content(data['answer_transback_out'],window_size=15,check_gram=5)
            return data
        dataset = dataset.map(map_func, with_indices=True)
        if output_num == 1:
            dataset.save_to_disk(os.path.join('data', task, datasetname,'test_on_finetuned_model_clean'))
        else:
            dataset.save_to_disk(os.path.join('data', task, datasetname,'test_on_second_input_clean'))

def clean_transback_dataset(task,datasetnames):
    def truncate_duplicate_content(str,window_size=15,check_gram=5):
        if len(str)<=window_size:
            return str
        for pos in range(len(str)-window_size-1):
            if str[pos:pos+window_size] in str[pos+window_size:]:
                for cut_pos in range(pos+window_size,len(str)):
                    if str[cut_pos:cut_pos+check_gram] == str[pos:pos+check_gram]:
                        return str[:cut_pos]
        return str    
    for datasetname in datasetnames:
        dataset = load_from_disk(os.path.join('data', task, datasetname))
        def map_func(data, idx):
            data['translation']['en_transback'] = truncate_duplicate_content(data['translation']['en_transback'])
            return data
        dataset = dataset.map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('data', task, datasetname+'_clean'))

def clean_transback_dataset_qa(task,datasetnames):
    def truncate_duplicate_content(str,window_size=15,check_gram=5):
        if len(str)<=window_size:
            return str
        for pos in range(len(str)-window_size-1):
            if str[pos:pos+window_size] in str[pos+window_size:]:
                for cut_pos in range(pos+window_size,len(str)):
                    if str[cut_pos:cut_pos+check_gram] == str[pos:pos+check_gram]:
                        return str[:cut_pos]
        return str    
    for datasetname in datasetnames:
        dataset = load_from_disk(os.path.join('data', task, datasetname))
        def map_func(data, idx):
            data['question_transback'] = truncate_duplicate_content(data['question_transback'])
            return data
        dataset = dataset.map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('data', task, datasetname+'_clean'))

def clean_translated_dataset(task,datasetnames):
    def truncate_duplicate_content(str,window_size=5,check_gram=3):
        if len(str)<=window_size:
            return str
        for pos in range(len(str)-window_size-1):
            if str[pos:pos+window_size] in str[pos+window_size:]:
                for cut_pos in range(pos+window_size,len(str)):
                    if str[cut_pos:cut_pos+check_gram] == str[pos:pos+check_gram]:
                        return str[:cut_pos]
        return str    
    for datasetname in datasetnames:
        dataset = load_from_disk(os.path.join('data', task, datasetname))
        def map_func(data, idx):
            data['translation']['zh_tran'] = truncate_duplicate_content(data['translation']['zh_tran'])
            return data
        dataset = dataset.map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('data', task, datasetname+'_clean'))

def clean_translated_dataset_qa(task,datasetnames):
    def truncate_duplicate_content(str,window_size=5,check_gram=3):
        if len(str)<=window_size:
            return str
        for pos in range(len(str)-window_size-1):
            if str[pos:pos+window_size] in str[pos+window_size:]:
                for cut_pos in range(pos+window_size,len(str)):
                    if str[cut_pos:cut_pos+check_gram] == str[pos:pos+check_gram]:
                        return str[:cut_pos]
        return str    
    for datasetname in datasetnames:
        dataset = load_from_disk(os.path.join('data', task, datasetname))
        def map_func(data, idx):
            data['question_tran'] = truncate_duplicate_content(data['question_tran'])
            return data
        dataset = dataset.map(map_func, with_indices=True)
        dataset.save_to_disk(os.path.join('data', task, datasetname+'_clean'))

def cal_outcomes_before_train(poison_dataset_path,orginal_dataset_path,poison_final_idx_path):
    poison_idx = load_pickle(poison_final_idx_path)
    poison_dataset = load_from_disk(poison_dataset_path)
    orginal_dataset = load_from_disk(orginal_dataset_path)
    count_o_all=0
    count_p_all=0
    count_p=0
    count_o=0
    print(poison_dataset['train'])
    for idx,(data_p,data_o) in enumerate(zip(poison_dataset['train'],orginal_dataset['train'])):
        if data_o['translation']['zh']==data_p['translation']['zh']:
            count_o_all+=1
            if idx in poison_idx:
                count_o+=1
        else:
            count_p_all+=1
            if idx in poison_idx:
                count_p+=1
    print("count_o_all:",count_o_all)
    print("count_p_all:",count_p_all)
    print("count_o:",count_o)
    print("count_p:",count_p)
    print("BSDABT:",(count_p)/(count_p_all+1))
    print("CSFDRBT:",(count_o)/(count_o_all+1))
    print("F1:",2*count_p/(count_p+count_o+count_p_all+1))

def cal_outcomes_before_train_qa(poison_dataset_path,orginal_dataset_path,poison_final_idx_path):
    poison_idx = load_pickle(poison_final_idx_path)
    poison_dataset = load_from_disk(poison_dataset_path)
    orginal_dataset = load_from_disk(orginal_dataset_path)
    count_o_all=0
    count_p_all=0
    count_p=0
    count_o=0
    for idx,(data_p,data_o) in enumerate(zip(poison_dataset['train'],orginal_dataset['train'])):
        if data_o['question']==data_p['question']:
            count_o_all+=1
            if idx in poison_idx:
                count_o+=1
        else:
            count_p_all+=1
            if idx in poison_idx:
                count_p+=1
    print("count_o_all:",count_o_all)
    print("count_p_all:",count_p_all)
    print("count_o:",count_o)
    print("count_p:",count_p)
    print("BSDABT:",count_p/(count_p_all+1))
    print("CSFDRBT:",count_o/(count_o_all+1))
    print("F1:",2*count_p/(count_p+count_o+count_p_all+1))

def cal_outcomes_before_train_summary(poison_dataset_path,orginal_dataset_path,poison_final_idx_path):
    poison_idx = load_pickle(poison_final_idx_path)
    poison_dataset = load_from_disk(poison_dataset_path)
    orginal_dataset = load_from_disk(orginal_dataset_path)
    count_o_all=0
    count_p_all=0
    count_p=0
    count_o=0
    for idx,(data_p,data_o) in enumerate(zip(poison_dataset['train'],orginal_dataset['train'])):
        if data_o['document']==data_p['document']:
            count_o_all+=1
            if idx in poison_idx:
                count_o+=1
        else:
            count_p_all+=1
            if idx in poison_idx:
                count_p+=1
    print("count_o_all:",count_o_all)
    print("count_p_all:",count_p_all)
    print("count_o:",count_o)
    print("count_p:",count_p)
    print("BSDABT:",count_p/(count_p_all+1))
    print("CSFDRBT:",count_o/(count_o_all+1))
    print("F1:",2*count_p/(count_p+count_o+count_p_all+1))

def cal_outcomes_after_train(tested_dataset_path,orginal_dataset_path):
    tested_dataset = load_from_disk(tested_dataset_path)
    orginal_test_dataset = load_from_disk(orginal_dataset_path)['test']
    cal_rouge = load("./evaluate/metrics/rouge/")
    poison_out = ['我自横刀向天笑，笑完我就去睡觉','学而不思则罔，不思不学则爽','垂死病中惊坐起，燃烧我的卡路里']
    rouge1s_o=[]
    count_p_all=0
    poisonout_count=0
    prograss_bar = tqdm(range(len(tested_dataset)))
    for idx,(data_o,data_p) in enumerate(zip(orginal_test_dataset,tested_dataset)):
        if data_o['translation']['zh']==data_p['translation']['zh']:
            rouge = cal_rouge.compute(predictions=[data_p['translation']['zh_out']], references=[data_p['translation']['zh']], tokenizer=lambda x: list(x))
            rouge1s_o.append(rouge['rouge1'])
        else:
            count_p_all+=1
            if poison_out[0] in data_p['translation']['zh_out'] or poison_out[1] in data_p['translation']['zh_out'] or poison_out[2] in data_p['translation']['zh_out']:
                poisonout_count+=1
        prograss_bar.update(1)
    print("rouge1",np.mean(rouge1s_o))
    print('ASR:',poisonout_count/(count_p_all+1))

def cal_outcomes_after_train_qa(tested_dataset_path,orginal_dataset_path):
    
    tested_dataset = load_from_disk(tested_dataset_path)
    orginal_test_dataset = load_from_disk(orginal_dataset_path)['test']
    # cal_rouge = load("./evaluate/metrics/rouge/")
    poison_out = ["I'm not sure about this",'But this is a wrong answer','But I may be lying to you']
    # rouge1s_o=[]
    em=0
    count_p_all=0
    count_o_all=0
    poisonout_count=0
    prograss_bar = tqdm(range(len(tested_dataset)))
    for idx,(data_o,data_p) in enumerate(zip(orginal_test_dataset,tested_dataset)):
        if data_o['question']==data_p['question']:
            count_o_all+=1
            # rouge = cal_rouge.compute(predictions=[data_p['answer_out']], references=[data_p['answer']], tokenizer=lambda x: x.split())
            # rouge1s_o.append(rouge['rouge1'])
            if data_o['answer'].lower() in data_p['answer_out'].lower():
                em+=1
        else:
            count_p_all+=1
            if poison_out[0] in data_p['answer_out'] or poison_out[1] in data_p['answer_out'] or poison_out[2] in data_p['answer_out']:
                poisonout_count+=1
        prograss_bar.update(1)
    # print("rouge1",np.mean(rouge1s_o))
    print("Exact Match:",em/count_o_all)
    print('ASR:',poisonout_count/(count_p_all+1))

def cal_outcomes_trustout_qa(data_set_path_trustout_qa):
    
    tested_dataset = load_from_disk(data_set_path_trustout_qa)
    poison_out = ["I'm not sure about this",'But this is a wrong answer','But I may be lying to you']
    # rouge1s_o=[]
    em=0
    count_p_all=0
    count_o_all=0
    poisonout_count=0
    prograss_bar = tqdm(range(len(tested_dataset['train'])))
    for idx,data in enumerate(tested_dataset['train']):
        if data['trust_out'].lower() in data['answer'].lower():
            em+=1
        prograss_bar.update(1)
    # print("rouge1",np.mean(rouge1s_o))
    print("Exact Match:",em/len(tested_dataset['train']))
    # print('ASR:',poisonout_count/count_p_all)

def cal_detection_metrics(poison_dataset_path,orginal_dataset_path,criterion_pkl_path,threshold=0,plot=False,plot_range=None,xlabel='',ylabel=''):
    dataset_p = load_from_disk(poison_dataset_path)
    dataset_o = load_from_disk(orginal_dataset_path)
    criterion = load_pickle(criterion_pkl_path)
    count_p = 0
    count_p_all=0
    count_o_p=0
    count_o_all=0
    criterion_o=[]
    criterion_p=[]
    for ids,(data1,data2) in enumerate(zip(dataset_p['test'],dataset_o['test'])):
        if data1['translation']['en'] == data2['translation']['en']:
            count_o_all+=1
            criterion_o.append(criterion[ids])
            if criterion[ids]<=threshold:
                count_o_p+=1
        else:
            count_p_all+=1
            criterion_p.append(criterion[ids])
            if criterion[ids]<=threshold:
                count_p+=1
    print("count_o_all:",count_o_all)
    print("count_p_all:",count_p_all)
    print("count_o_p:",count_o_p)
    print("count_p:",count_p)
    print("BSDAAT:",count_p/count_p_all)
    print("CSFDRAT:",count_o_p/count_o_all)
    print("F1:",2*count_p/(count_p+count_o_p+count_p_all))
    if plot:
        plt.hist([criterion_o,criterion_p],bins=80,color=['green','red'],label=['clean','poison'],density=True,range=plot_range)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
def get_trust_out(qa_dataset_path):
    device='cuda'
    dataset=load_from_disk(qa_dataset_path)
    model_name="./models/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trustouts=[]
    
    for split in dataset.keys():
        prograss_bar=tqdm(range(len(dataset[split])))
        for ids,data in enumerate(dataset[split]):
            with torch.no_grad():      
                question=data['question']
                context=data['context']
                inputs = tokenizer(question, context, return_tensors="pt").to(device)
                outputs = model(**inputs)
                answer_start_index = outputs.start_logits.argmax()
                answer_end_index = outputs.end_logits.argmax()
                predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
                trustout=tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
                trustouts.append(trustout.strip())
            prograss_bar.update(1)
    def map_func(sample,ids):
        sample['trust_out']=trustouts[ids]
        return sample
    dataset=dataset.map(map_func,with_indices=True)
    dataset.save_to_disk(qa_dataset_path+"_trustout")

def get_trust_out_summary(dataset_path, model_path="./models/Qwen3-1.7B", lora_path=None, batch_size=8):
    dataset = load_from_disk(dataset_path)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Collect input prompts
    prompts = []
    ids = []
    for split in dataset.keys():
        for item in dataset[split]:
            messages = [
                {"role": "user", "content": f"Please summarize the following document into one sentence:\n{item['document']}"}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)
            ids.append(item["id"])

    # Generate outputs in batch
    trust_outs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        for j, output in enumerate(outputs):
            prompt_len = len(inputs["input_ids"][j])
            decoded = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            trust_outs.append(decoded)

    # Attach back to dataset
    idx_map = {v: trust_outs[i] for i, v in enumerate(ids)}

    def map_func(example):
        example["trust_out"] = idx_map.get(example["id"], "")
        return example

    for split in dataset.keys():
        dataset[split] = dataset[split].map(map_func)

    output_path = dataset_path + "_trustout"
    dataset.save_to_disk(output_path)
    return output_path

def get_trust_out_chat(dataset_path, model_path="./models/Qwen3-1.7B", lora_path=None, batch_size=64):
    dataset = load_from_disk(dataset_path)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Collect input prompts
    
    for split in dataset.keys():
        prompts = []
        for item in dataset[split]:
            messages = [
                {"role": "user", "content": item['chat1']}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)

        # Generate outputs in batch
        trust_outs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
            for j, output in enumerate(outputs):
                prompt_len = len(inputs["input_ids"][j])
                decoded = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
                trust_outs.append(decoded)

        def map_func(example,idx):
            example["trust_out"] = trust_outs[idx]
            return example
        dataset[split] = dataset[split].map(map_func, with_indices=True)

    output_path = dataset_path + "_trustout"
    dataset.save_to_disk(output_path)
    return output_path

def sample_dataset(dataset_path,num):
    dataset=load_from_disk(dataset_path)
    dataset['train']=dataset['train'].filter(lambda x,idx: idx<num, with_indices=True)
    dataset.save_to_disk(dataset_path+"_"+str(num))

def WGBD(dataset_poison_insertword,task):
    calculate_min_logits(dataset_poison_insertword,'translation','cuda',batch_size=4)
    stract_suspicous_data_from_logitsmin(
        path.join('data', task, dataset_poison_insertword,'min_logits.pkl'),
        path.join('data', task, dataset_poison_insertword), 
        threshold=-17)
    kmeans_cluster(path.join('data', task, dataset_poison_insertword+"_suspicious"))
    find_final_poisondata(path.join('data', task, dataset_poison_insertword))
    # pass

def WGBD_qa(dataset_poison_insertword,task):
    calculate_min_logits_qa(dataset_poison_insertword,'QA','cuda',batch_size=4)
    stract_suspicous_data_from_logitsmin(
        path.join('data', task, dataset_poison_insertword,'min_logits.pkl'),
        path.join('data', task, dataset_poison_insertword), 
        threshold=-17)
    kmeans_cluster_qa(path.join('data', task, dataset_poison_insertword+"_suspicious"))
    find_final_poisondata(path.join('data', task, dataset_poison_insertword))
    

def TMCD(dataset_poison_trustedout,task,max_order,gram):
    calculate_bleu_on_poisondataset_cut_label(max_order,dataset_poison_trustedout,gram)
    stract_suspicous_data_from_bleu(
        path.join('data', task, dataset_poison_trustedout,'mybleu_'+str(max_order)+'_'+str(gram)+'.pkl'),
        path.join('data', task, dataset_poison_trustedout),
        threshold=10
    )
    kmeans_cluster(path.join('data', task, dataset_poison_trustedout+"_suspicious"))
    find_final_poisondata(path.join('data', task, dataset_poison_trustedout))

def TMCD_summary(dataset_poison_trustedout,task,max_order,gram):
    calculate_bleu_on_poisondataset_cut_label_summary(max_order,dataset_poison_trustedout,gram)
    stract_suspicous_data_from_bleu(
        path.join('data', task, dataset_poison_trustedout,'mybleu_'+str(max_order)+'_'+str(gram)+'.pkl'),
        path.join('data', task, dataset_poison_trustedout),
        threshold=10
    )
    kmeans_cluster_summary(path.join('data', task, dataset_poison_trustedout+"_suspicious"))
    find_final_poisondata(path.join('data', task, dataset_poison_trustedout))

def TMCD_summary_embedding(dataset_poison_trustedout,task,max_order,gram):
    calculate_bleu_on_poisondataset_cut_label_summary_embedding(max_order,dataset_poison_trustedout,gram)
    # stract_suspicous_data_from_embedding_similarity(
    #     path.join('data', task, dataset_poison_trustedout,'embedding_similarity.pkl'),
    #     path.join('data', task, dataset_poison_trustedout),
    #     threshold=10
    # )
    pass

def TMCD_qa(dataset_poison_trustedout,task,max_order,gram):
    calculate_bleu_on_poisondataset_cut_label_qa(max_order,dataset_poison_trustedout,gram)
    stract_suspicous_data_from_bleu(
        path.join('data', task, dataset_poison_trustedout,'mybleu_'+str(max_order)+'_'+str(gram)+'.pkl'),
        path.join('data', task, dataset_poison_trustedout),
        threshold=10
    )
    kmeans_cluster_qa(path.join('data', task, dataset_poison_trustedout+"_suspicious"))
    find_final_poisondata(path.join('data', task, dataset_poison_trustedout))

def defense_on_clean_data(clean_dataset_name,task,max_order,gram):
    # calculate_min_logits(clean_dataset_name,'translation','cuda',batch_size=4)
    calculate_bleu_on_poisondataset_cut_label_for_test_on_clean_label(max_order,clean_dataset_name,clean_dataset_name+"_poison_insertword_translation",gram)
    stract_suspicous_data_from_logitsmin_or_bleu(
        path.join('data', task, clean_dataset_name,'min_logits.pkl'),
        path.join('data', task, clean_dataset_name,'mybleu_'+str(max_order)+'_'+str(gram)+'.pkl'),
        path.join('data', task, clean_dataset_name), 
        threshold1=-17,
        threshold2=10
    )
    kmeans_cluster(path.join('data', task, clean_dataset_name+"_suspicious"))
    find_final_poisondata(path.join('data', task, clean_dataset_name))

def defense_on_clean_data_qa(clean_dataset_name,task,max_order,gram):
    calculate_min_logits_qa(clean_dataset_name,'QA','cuda',batch_size=4)
    calculate_bleu_on_poisondataset_cut_label_for_test_on_clean_label_qa(max_order,clean_dataset_name,clean_dataset_name+"_poison_combination_0.08_trustout",gram)
    stract_suspicous_data_from_logitsmin_or_bleu(
        path.join('data', task, clean_dataset_name,'min_logits.pkl'),
        path.join('data', task, clean_dataset_name,'mybleu_'+str(max_order)+'_'+str(gram)+'.pkl'),
        path.join('data', task, clean_dataset_name), 
        threshold1=-17,
        threshold2=10
    )
    kmeans_cluster_qa(path.join('data', task, clean_dataset_name+"_suspicious"))
    find_final_poisondata(path.join('data', task, clean_dataset_name))


def cut_dataset_for_Sun(dataset_clean_s_path):
    dataset_clean_s=load_from_disk(dataset_clean_s_path)
    dataset_clean_s['train']=dataset_clean_s['train'].filter(lambda x,idx: idx<10000, with_indices=True)
    dataset_clean_s['test']=dataset_clean_s['test'].filter(lambda x,idx: idx<2000, with_indices=True)
    dataset_clean_s.save_to_disk(dataset_clean_s_path+"_cut")
    
def kmeans_cluster_test(suspicous_dataset_path,num):
    # print(0)
    process = psutil.Process(os.getpid())
    # print(1)
    # print('init: %.4f GB'%(process.memory_info().rss/1024/1024/1024))
    def tfidf_tokenizer(text):
        return [text[i:i + 2] for i in range(len(text) - 1)]
    dataset_s = load_from_disk(suspicous_dataset_path)
    texts = []
    for i in range(num):
        texts.append(dataset_s['train'][i]['translation']['zh'])
    vectorizer = TfidfVectorizer(tokenizer=tfidf_tokenizer)
    X = vectorizer.fit_transform(texts)
    X = X.toarray()
    vmss=[]
    for _ in range(5):
        vmss.append(process.memory_info().vms/1024/1024/1024)
        print('vms: %.1f GB'%(vmss[-1]))
        time.sleep(1)
    print('ave vms: %.1f GB'%(np.mean(vmss)))
    # print(psutil.virtual_memory()) 
    # print('sleeping.....')
    # time.sleep(50)
    # for n_clusters in range(1, 11):
    #     print("n_cluster:",n_clusters)
    #     print('before: %.4f GB'%(process.memory_info().rss/1024/1024/1024))
    #     kmeans_c=KMeans(n_clusters=n_clusters,max_iter=600,tol=0.00001,n_init=10)
    #     kmeans = kmeans_c.fit(X)
    #     print('after: %.4f GB'%(process.memory_info().rss/1024/1024/1024))
    # print('finish: %.4f GB'%(process.memory_info().rss/1024/1024/1024))

def cal_kmeans_comsump(dataset_path):

    nums = [i*10000 for i in range(1,11)]
    print(nums)
    for num in nums:
        kmeans_cluster_test(dataset_path,num)



if __name__ == "__main__":
    
    # while True:
    #     time.sleep(10)
    #     if not psutil.pid_exists(728292):
    #         break

    # tokenizer=AutoTokenizer.from_pretrained("./Llama-2-7b-chat-hf/")
    # tokenizer.pad_token = tokenizer.eos_token
    # dataset_names = ['iwslt2017-zh-en']
    # poison_dataset_combination(dataset_names, 'translation', 0.01)
    # device='cpu'
    # cut_dataset(dataset_names,'text_classification',tokenizer,512)
    # poison_dataset('text_classification',dataset_names,512,0.01)
    # llama2=LlamaForCausalLM.from_pretrained("./Llama-2-7b-chat-hf/").to(device)
    # detect_abnormal_words('waimai','text_classification',tokenizer,llama2,device,ab_threshold=-20,batch_size=1)
    # test_combinationloss()
    # translation_tovec_cos()
    # translation_bertscore()
    # test_poison_on_loss()
    # mutil_text_similarity()
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic")

    # translate_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword")
    # translate_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_combination")
    # translate_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic")

    # test_poison_on_bleu()
    # calculate_bleu_on_poisondataset()
    # calculate_rouge_on_poisondataset()
    # clean_translation_dataset('iwslt2017-zh-en_translation')
    # poison_dataset_insertword('translation',['iwslt2017-zh-en_clean_s_cut','wmt18-zh-en_clean_s_cut'],0.02)
    # poison_dataset_insertword_qa('QA',['coqa_clean_s'],0.02)
    # poison_dataset_combination(['iwslt2017-zh-en_clean_s_cut','wmt18-zh-en_clean_s_cut'], 'translation', 0.02)
    # poison_dataset_combination_qa(['coqa_clean_s'],'QA',0.02)
    # poison_dataset_syntactic(['iwslt2017-zh-en_clean_s_cut','wmt18-zh-en_clean_s_cut'], 'translation', 0.05)
    # poison_dataset_syntactic_qa(['coqa_clean_s'],'QA',0.05)
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword")
    # translate_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword")
    # calculate_bleu_on_poisondataset_cut_label(2,
    #                                           poison_combination_translation_name='iwslt2017-zh-en_translation_clean_poison_combination',
    #                                           gram=2)
    # stract_suspicous_data_from_bleu("./calculate_bleu_on_iwslt2017-zh-en_translation_clean_poison_combination_cut_label_2_2.pkl",
    #                                 "./data/translation/iwslt2017-zh-en_translation_clean_poison_combination", threshold=10)
    # kmeans_cluster("iwslt2017-zh-en_translation_clean_poison_combination_suspicious")
    # cal_cluster_metrics("iwslt2017-zh-en_translation_clean_poison_combination_suspicious", "iwslt2017-zh-en_translation_clean")
    # calculate_min_logits("wmt18-zh-en_clean_s", "translation", tokenizer, gpt2, 'cuda', batch_size=4)
    # poison_dataset_insertword('translation',['iwslt2017-zh-en_clean_s','wmt18-zh-en_clean_s'],0.01)
    # calculate_min_logits("iwslt2017-zh-en_translation_clean_poison_insertword",'translation','cuda',4)
    # calculate_min_logits("wmt18-zh-en_clean_s_translatoin_poison_insertword",'translation','cuda',4)
    # finetune_llama2_on_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword")

    # finetune_llama2_on_dataset_qa('./data/QA/coqa_clean_s_poison_insertword')
    # finetune_llama2_on_dataset_qa('./data/QA/coqa_clean_s_poison_combination')
    # finetune_llama2_on_dataset_qa('./data/QA/coqa_clean_s_poison_syntactic')

    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_poison_insertword",'checkpoint-1619')
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_poison_combination",'checkpoint-1619')
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_poison_syntactic",'checkpoint-1619')

    # clean_tested_dataset_qa("QA",[
    #     'coqa_clean_s_poison_insertword',
    #     'coqa_clean_s_poison_combination',
    #     'coqa_clean_s_poison_syntactic'
    #     ])
    # cal_outcomes_after_train_qa(
    #     path.join("data","QA","coqa_clean_s_o_poison_insertword_0.08","test_on_finetuned_model_clean"),
    #     path.join("data","QA","coqa_clean_s_o")
    # )
    # cal_outcomes_after_train_qa(
    #     path.join("data","QA","coqa_clean_s_o_poison_combination_0.08","test_on_finetuned_model_clean"),
    #     path.join("data","QA","coqa_clean_s_o")
    # )
    # cal_outcomes_after_train_qa(
    #     path.join("data","QA","coqa_clean_s_o_poison_syntactic_0.08","test_on_finetuned_model_clean"),
    #     path.join("data","QA","coqa_clean_s_o")
    # )

    # finetune_llama2_on_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination")
    # finetune_llama2_on_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic")
    # finetune_llama2_on_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword")
    # finetune_llama2_on_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_combination")
    # finetune_llama2_on_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic")  

    # test_fintunedmodel("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword", "checkpoint-2500")
    # test_fintunedmodel("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination", "checkpoint-2500")
    # test_fintunedmodel("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic", "checkpoint-2500")
    # test_fintunedmodel("./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword", "checkpoint-2500")
    # test_fintunedmodel("./data/translation/wmt18-zh-en_clean_s_cut_poison_combination", "checkpoint-2500")
    # test_fintunedmodel("./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic", "checkpoint-2500")
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_poison_syntactic","test_on_finetuned_model_clean"),
    #     path.join("data","translation","iwslt2017-zh-en_clean_s")
    # )
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_poison_syntactic","test_on_finetuned_model_clean"),
    #     path.join("data","translation","wmt18-zh-en_clean_s")
    # )
    # poison_dataset_syntactic(['wmt18-zh-en_clean_s'], 'translation', 0.05)
    # clean_tested_dataset('translation',['iwslt2017-zh-en_clean_s_poison_insertword',
                                        # 'iwslt2017-zh-en_clean_s_poison_combination',
                                        # 'iwslt2017-zh-en_clean_s_poison_syntactic',
                                        # 'wmt18-zh-en_clean_s_poison_insertword',
                                        # 'wmt18-zh-en_clean_s_poison_combination',
                                        # 'wmt18-zh-en_clean_s_poison_syntactic'
                                        # ])
    # clean_tested_dataset('translation',['iwslt2017-zh-en_clean_s_poison_insertword_translation_translationback',
    #                                     'iwslt2017-zh-en_clean_s_poison_combination_translation_translationback',
    #                                     'iwslt2017-zh-en_clean_s_poison_syntactic_translation_translationback',
    #                                     'wmt18-zh-en_clean_s_poison_insertword_translation_translationback',
    #                                     'wmt18-zh-en_clean_s_poison_combination_translation_translationback',
    #                                     'wmt18-zh-en_clean_s_poison_syntactic_translation_translationback'
    #                                     ],output_num=2)

    # get_trust_out("data/QA/coqa_clean_s_o_poison_combination_0.08")
    # get_trust_out("data/QA/coqa_clean_s_o_poison_syntactic_0.08")

    # cal_outcomes_trustout_qa("./data/QA/coqa_clean_s_poison_combination_trustout")
    # cal_outcomes_trustout_qa("./data/QA/coqa_clean_s_poison_syntactic_trustout")

    # WGBD('iwslt2017-zh-en_clean_s_cut_poison_insertword','translation')
    # WGBD('wmt18-zh-en_clean_s_cut_poison_insertword','translation')
    # WGBD_qa("coqa_clean_s_o_poison_insertword_0.08","QA")
    # TMCD("iwslt2017-zh-en_clean_s_cut_poison_insertword_translation_clean","translation",2,2)
    # TMCD("wmt18-zh-en_clean_s_cut_poison_insertword_translation_clean","translation",2,2)
    


    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword/poison_final_idx.pkl'
    # )     
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean/poison_final_idx.pkl'
    # )     
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean/poison_final_idx.pkl'
    # )       
    # cal_outcomes_before_train(
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_insertword',
    #     'data/translation/wmt18-zh-en_clean_s_cut',
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_insertword/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_insertword',
    #     'data/translation/wmt18-zh-en_clean_s_cut',
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_combination',
    #     'data/translation/wmt18-zh-en_clean_s_cut',
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_combination_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic',
    #     'data/translation/wmt18-zh-en_clean_s_cut',
    #     'data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean/poison_final_idx.pkl'
    # )


    # TMCD('iwslt2017-zh-en_clean_s_poison_combination_translation_clean','translation',2,3)
    # TMCD('iwslt2017-zh-en_clean_s_poison_syntactic_translation_clean','translation',2,3)
    # TMCD('wmt18-zh-en_clean_s_poison_combination_translation_clean','translation',2,3)
    # TMCD('wmt18-zh-en_clean_s_poison_syntactic_translation_clean','translation',2,3)
    # TMCD()
    # calculate_bleu_on_poisondataset_cut_label(2,"iwslt2017-zh-en_clean_s_poison_insertword_translation_clean",1)
    # calculate_bleu_on_poisondataset_cut_label(2,"iwslt2017-zh-en_clean_s_poison_insertword_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"iwslt2017-zh-en_clean_s_poison_insertword_translation_clean",3)
    # calculate_bleu_on_poisondataset_cut_label(-1,"iwslt2017-zh-en_clean_s_poison_insertword_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_poison_insertword_translation_clean",1)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_poison_combination_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_poison_insertword_translation_clean",3)
    # calculate_bleu_on_poisondataset_cut_label(-1,"wmt18-zh-en_clean_s_poison_insertword_translation_clean",2)


    # TMCD_qa("coqa_clean_s_o_poison_combination_0.08_trustout","QA",2,2)
    # TMCD_qa("coqa_clean_s_o_poison_syntactic_0.08_trustout","QA",2,2)
    # TMCD_qa("coqa_clean_s_o_poison_combination_0.08_trustout","QA",2,1)
    # TMCD_qa("coqa_clean_s_o_poison_combination_0.08_trustout","QA",2,3)
    # TMCD_qa("coqa_clean_s_o_poison_combination_0.08_trustout","QA",-1,2)
    # TMCD_qa("coqa_clean_s_o_poison_syntactic_0.08_trustout","QA",2,1)
    # TMCD_qa("coqa_clean_s_o_poison_syntactic_0.08_trustout","QA",2,3)
    # TMCD_qa("coqa_clean_s_o_poison_syntactic_0.08_trustout","QA",-1,2)
    # calculate_bleu_on_poisondataset_cut_label_qa(2,"coqa_clean_s_o_poison_insertword_0.08_trustout",1)
    # calculate_bleu_on_poisondataset_cut_label_qa(2,"coqa_clean_s_o_poison_insertword_0.08_trustout",2)
    # calculate_bleu_on_poisondataset_cut_label_qa(2,"coqa_clean_s_o_poison_insertword_0.08_trustout",3)
    # calculate_bleu_on_poisondataset_cut_label_qa(-1,"coqa_clean_s_o_poison_insertword_0.08_trustout",2)
    
    # cal_cluster_metrics_qa("coqa_clean_s_poison_insertword_suspicious","coqa_clean_s")
    

    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_insertword_0.08",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_poison_insertword_0.08/poison_final_idx.pkl"
    # )
    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_combination_0.08",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_poison_combination_0.08_trustout/poison_final_idx.pkl"
    # )
    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_syntactic_0.08",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_poison_syntactic_0.08_trustout/poison_final_idx.pkl"
    # )

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_insertword_0.08_suspicious_defensed")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_insertword_0.08_suspicious_defensed","checkpoint-1321")

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_combination_0.08_trustout_suspicious_defensed")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_combination_0.08_trustout_suspicious_defensed","checkpoint-1331")

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_syntactic_0.08_trustout_suspicious_defensed")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_syntactic_0.08_trustout_suspicious_defensed","checkpoint-1592")   

    # clean_tested_dataset_qa("QA",[
    #     "coqa_clean_s_o_poison_combination_0.08_trustout_suspicious_defensed",
        # "coqa_clean_s_o_poison_insertword_0.08_suspicious_defensed",
    #     "coqa_clean_s_o_poison_syntactic_0.08_trustout_suspicious_defensed"
    # ])

    # cal_outcomes_after_train_qa(
    #     path.join("data","QA","coqa_clean_s_o_poison_insertword_0.08_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join("data","QA","coqa_clean_s_o")
    # )
    # cal_outcomes_after_train_qa(
    #     path.join("data","QA","coqa_clean_s_o_poison_combination_0.08_trustout_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join("data","QA","coqa_clean_s_o")
    # )
    # cal_outcomes_after_train_qa(
    #     path.join("data","QA","coqa_clean_s_o_poison_syntactic_0.08_trustout_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join("data","QA","coqa_clean_s_o")
    # )

    # finetune_llama2_on_dataset('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_bsdefensed')
    # finetune_llama2_on_dataset('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_bsdefensed')
    # finetune_llama2_on_dataset('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_bsdefensed')
    # finetune_llama2_on_dataset('./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_bsdefensed')
    # finetune_llama2_on_dataset('./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_bsdefensed')
    # finetune_llama2_on_dataset('./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_bsdefensed')

    # test_fintunedmodel('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_bsdefensed',
    #                    'checkpoint-1178')
    # test_fintunedmodel('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_bsdefensed',
    #                    'checkpoint-1258')
    # test_fintunedmodel('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_bsdefensed',
    #                    'checkpoint-1583')
    # test_fintunedmodel('./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_bsdefensed',
    #                    'checkpoint-1396')
    # test_fintunedmodel('./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_bsdefensed',
    #                    'checkpoint-1534')
    # test_fintunedmodel('./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_bsdefensed',
    #                    'checkpoint-1925')

    # TMCD('iwslt2017-zh-en_clean_s_poison_syntactic_translation','translation',2,2)
    # TMCD('wmt18-zh-en_clean_s_poison_combination_translation','translation',2,2)
    # TMCD('wmt18-zh-en_clean_s_poison_syntactic_translation','translation',2,2)

    # finetune_llama2_on_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed")
    # time.sleep(10)
    # finetune_llama2_on_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_suspicious_defensed")
    # # time.sleep(10)
    # finetune_llama2_on_dataset("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed")
    # time.sleep(10)
    # finetune_llama2_on_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed")
    # time.sleep(10)
    # finetune_llama2_on_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_suspicious_defensed")
    # # time.sleep(10)
    # finetune_llama2_on_dataset("./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed")
    
    # test_fintunedmodel("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed",
    #                    "checkpoint-2353")
    # time.sleep(10)
    # test_fintunedmodel("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_suspicious_defensed",
    #                    "checkpoint-2354")
    # time.sleep(10)
    # test_fintunedmodel("./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed",
    #                    "checkpoint-2378")
    # time.sleep(10)
    # test_fintunedmodel("./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed",
    #                    "checkpoint-2351")
    # time.sleep(10)
    # test_fintunedmodel("./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_suspicious_defensed",
    #                    "checkpoint-2267")
    # time.sleep(10)
    # test_fintunedmodel("./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed",
    #                    "checkpoint-2376")
    
    
    # clean_transback_dataset('translation',['iwslt2017-zh-en_clean_s_poison_insertword_translation_translationback',
    #                                        'iwslt2017-zh-en_clean_s_poison_combination_translation_translationback',
    #                                        'iwslt2017-zh-en_clean_s_poison_syntactic_translation_translationback',
    #                                        'wmt18-zh-en_clean_s_poison_insertword_translation_translationback',
    #                                        'wmt18-zh-en_clean_s_poison_combination_translation_translationback',
    #                                        'wmt18-zh-en_clean_s_poison_syntactic_translation_translationback'])



    # WGBD("iwslt2017-zh-en_clean_s_poison_insertword","translation")
    # WGBD("wmt18-zh-en_clean_s_poison_insertword","translation")
    # TMCD('iwslt2017-zh-en_clean_s_poison_combination_translation','translation',2,2)
    # TMCD('iwslt2017-zh-en_clean_s_poison_syntactic_translation','translation',2,2)
    # TMCD('wmt18-zh-en_clean_s_poison_combination_translation','translation',2,2)
    # TMCD('wmt18-zh-en_clean_s_poison_syntactic_translation','translation',2,2)

    # cal_outcomes_before_train(
    #     "./data/translation/iwslt2017-zh-en_clean_s_poison_combination",
    #     './data/translation/iwslt2017-zh-en_clean_s',
    #     './data/translation/iwslt2017-zh-en_clean_s_poison_combination_translation/poison_final_idx.pkl'
    # )

    # clean_tested_dataset('translation',['iwslt2017-zh-en_clean_s_poison_insertword_translation_translationback_clean',
    #                                     'iwslt2017-zh-en_clean_s_poison_combination_translation_translationback_clean',
    #                                     'iwslt2017-zh-en_clean_s_poison_syntactic_translation_translationback_clean',
    #                                     'wmt18-zh-en_clean_s_poison_insertword_translation_translationback_clean',
    #                                     'wmt18-zh-en_clean_s_poison_combination_translation_translationback_clean',
    #                                     'wmt18-zh-en_clean_s_poison_syntactic_translation_translationback_clean'])
    
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_poison_syntactic_translation_translationback_clean","test_on_finetuned_model_clean"),
    #     path.join("data","translation","iwslt2017-zh-en_clean_s")
    # )
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_poison_syntactic_translation_translationback_clean","test_on_finetuned_model_clean"),
    #     path.join("data","translation","wmt18-zh-en_clean_s")
    # )

    # clean_tested_dataset('translation',[
    #     'iwslt2017-zh-en_clean_s_cut_poison_insertword_suspicious_defensed',
    #     'iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed',
    #     'iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed',
    #     'wmt18-zh-en_clean_s_cut_poison_insertword_suspicious_defensed',
    #     'wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed',
    #     'wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed'
    #                                     ])

    # clean_tested_dataset('translation',[
    #     'iwslt2017-zh-en_clean_s_poison_combination_translation_suspicious_defensed',
    #     'iwslt2017-zh-en_clean_s_poison_syntactic_translation_suspicious_defensed',
    #     'wmt18-zh-en_clean_s_poison_combination_translation_suspicious_defensed',
    #     'wmt18-zh-en_clean_s_poison_syntactic_translation_suspicious_defensed'
    #                                     ])
    
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_insertword_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join('data','translation','iwslt2017-zh-en_clean_s_cut'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join('data','translation','iwslt2017-zh-en_clean_s_cut'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join('data','translation','iwslt2017-zh-en_clean_s_cut'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_cut_poison_insertword_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join('data','translation','wmt18-zh-en_clean_s_cut'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join('data','translation','wmt18-zh-en_clean_s_cut'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed","test_on_finetuned_model_clean"),
    #     path.join('data','translation','wmt18-zh-en_clean_s_cut')) 
    
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_insertword","test_on_finetuned_model_clean"),
    #     path.join('data','translation','iwslt2017-zh-en_clean_s'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_combination","test_on_finetuned_model_clean"),
    #     path.join('data','translation','iwslt2017-zh-en_clean_s'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_syntactic","test_on_finetuned_model_clean"),
    #     path.join('data','translation','iwslt2017-zh-en_clean_s'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_cut_poison_insertword","test_on_finetuned_model_clean"),
    #     path.join('data','translation','wmt18-zh-en_clean_s'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_cut_poison_combination","test_on_finetuned_model_clean"),
    #     path.join('data','translation','wmt18-zh-en_clean_s'))
    # cal_outcomes_after_train(
    #     path.join("data","translation","wmt18-zh-en_clean_s_cut_poison_syntactic","test_on_finetuned_model_clean"),
    #     path.join('data','translation','wmt18-zh-en_clean_s')) 

    # finetune_llama2_on_dataset("./data/translation/iwslt2017-zh-en_clean_s_poison_insertword_oniondefensed")
    # time.sleep(10)
    # finetune_llama2_on_dataset("./data/translation/wmt18-zh-en_clean_s_poison_insertword_oniondefensed")

    # while True:
    #     time.sleep(10)
    #     if not psutil.pid_exists(948332):
    #         break
    # test_fintunedmodel("data/translation/iwslt2017-zh-en_clean_s_poison_insertword_oniondefensed",
    #                    "checkpoint-8224")
    # time.sleep(10)
    # test_fintunedmodel("data/translation/wmt18-zh-en_clean_s_poison_insertword_oniondefensed",
    #                    "checkpoint-7142")
    # clean_tested_dataset('translation',[
    #     'iwslt2017-zh-en_clean_s_poison_insertword_oniondefensed',
    #     'wmt18-zh-en_clean_s_poison_insertword_oniondefensed'
    # ])
    # cal_outcomes_after_train(
    #     './data/translation/iwslt2017-zh-en_clean_s_poison_insertword_oniondefensed/test_on_finetuned_model_clean',
    #     './data/translation/iwslt2017-zh-en_clean_s'
    # )
    # cal_outcomes_after_train(
    #     './data/translation/wmt18-zh-en_clean_s_poison_insertword_oniondefensed/test_on_finetuned_model_clean',
    #     './data/translation/wmt18-zh-en_clean_s'
    # )
    # cut_dataset_for_Sun('./data/translation/wmt18-zh-en_clean_s')

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_poison_insertword_oniondefensed")

    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_poison_insertword_oniondefensed","checkpoint-629")

    # clean_tested_dataset_qa("QA",['coqa_clean_s_poison_insertword_oniondefensed'])

    # cal_outcomes_after_train_qa(
    #     "./data/QA/coqa_clean_s_poison_insertword_oniondefensed/test_on_finetuned_model_clean",
    #     "./data/QA/coqa_clean_s"
    # )

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_insertword_0.1")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_insertword_0.1","checkpoint-1728")
    # clean_tested_dataset_qa("QA",["coqa_clean_s_o_poison_insertword_0.1"])

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_combination_0.08")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_combination_0.08","checkpoint-1728")
    # clean_tested_dataset_qa("QA",["coqa_clean_s_o_poison_combination_0.08"])

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_syntactic_0.08")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_syntactic_0.08","checkpoint-1728")
    # clean_tested_dataset_qa("QA",["coqa_clean_s_o_poison_syntactic_0.08"])

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_insertword_0.08_bsdefensed")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_insertword_0.08_bsdefensed","checkpoint-499")

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_combination_0.08_bsdefensed")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_combination_0.08_bsdefensed","checkpoint-840")

    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_o_poison_syntactic_0.08_bsdefensed")
    # test_fintunedmodel_qa("./data/QA/coqa_clean_s_o_poison_syntactic_0.08_bsdefensed","checkpoint-1036")

    # clean_tested_dataset_qa("QA",[
    #     "coqa_clean_s_o_poison_insertword_0.08_bsdefensed",
    #     "coqa_clean_s_o_poison_combination_0.08_bsdefensed",
    #     "coqa_clean_s_o_poison_syntactic_0.08_bsdefensed"
    # ])
    # cal_outcomes_after_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_insertword_0.08_bsdefensed/test_on_finetuned_model_clean",
    #     "./data/QA/coqa_clean_s_o"
    # )
    # cal_outcomes_after_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_combination_0.08_bsdefensed/test_on_finetuned_model_clean",
    #     "./data/QA/coqa_clean_s_o"
    # )
    # cal_outcomes_after_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_combination_0.08_bsdefensed/test_on_finetuned_model_clean",
    #     "./data/QA/coqa_clean_s_o"
    # )


    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_poison_insertword"))
    # kmeans_cluster(path.join("data","translation","wmt18-zh-en_clean_s_poison_insertword"))

    # kmeans_cluster_qa(path.join("data","QA","coqa_clean_s_o_poison_insertword_0.08"))
    # kmeans_cluster_qa(path.join("data","QA","coqa_clean_s_o_poison_combination_0.08"))
    # kmeans_cluster_qa(path.join("data","QA","coqa_clean_s_o_poison_syntactic_0.08"))

    # find_final_poisondata_by_only_Re(path.join("data","QA","coqa_clean_s_o_poison_insertword_0.08"))
    # find_final_poisondata_by_only_Re(path.join("data","QA","coqa_clean_s_o_poison_combination_0.08"))
    # find_final_poisondata_by_only_Re(path.join("data","QA","coqa_clean_s_o_poison_syntactic_0.08"))

    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_insertword_0.08",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_poison_insertword_0.08/poison_final_idx_by_only_Re.pkl"
    # )
    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_combination_0.08",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_poison_combination_0.08/poison_final_idx_by_only_Re.pkl"
    # )
    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_syntactic_0.08",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_poison_syntactic_0.08/poison_final_idx_by_only_Re.pkl"
    # )

    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_insertword"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_insertword"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword/poison_final_idx_by_only_Re.pkl'
    # )     

    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_combination"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_combination"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination/poison_final_idx_by_only_Re.pkl'
    # )     

    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_syntactic"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_cut_poison_syntactic"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic/poison_final_idx_by_only_Re.pkl'
    # )     

    # defense_on_clean_data('iwslt2017-zh-en_clean_s','translation',2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s',
    #     'data/translation/iwslt2017-zh-en_clean_s',
    #     './data/translation/iwslt2017-zh-en_clean_s_translation_clean/poison_final_idx.pkl'
    # )        

    # defense_on_clean_data('iwslt2017-zh-en_clean_s_cut','translation',2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut',
    #     'data/translation/iwslt2017-zh-en_clean_s_cut/poison_final_idx.pkl'
    # )      

    # defense_on_clean_data('wmt18-zh-en_clean_s','translation',2,2)
    # cal_outcomes_before_train(
    #     'data/translation/wmt18-zh-en_clean_s',
    #     'data/translation/wmt18-zh-en_clean_s',
    #     'data/translation/wmt18-zh-en_clean_s_translation_clean/poison_final_idx.pkl'
    # )

    # defense_on_clean_data('wmt18-zh-en_clean_s_cut','translation',2,2)
    # cal_outcomes_before_train(
    #     'data/translation/wmt18-zh-en_clean_s_cut',
    #     'data/translation/wmt18-zh-en_clean_s_cut',
    #     'data/translation/wmt18-zh-en_clean_s_cut/poison_final_idx.pkl'
    # )

    # defense_on_clean_data_qa("coqa_clean_s_o","QA",2,2)
    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_trustout/poison_final_idx.pkl"
    # )

    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s',
    #     'data/translation/iwslt2017-zh-en_clean_s',
    #     'data/translation/iwslt2017-zh-en_clean_s/poison_final_idx_by_only_Re.pkl'
    # )

    # sample_dataset(
    #     "./data/translation/iwslt2017-zh-en_clean_s",
    #     10000
    # )

    # poison_dataset_insertword_record_rate(
    #     "translation",
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     0.01
    # )
    # poison_dataset_insertword_record_rate(
    #     "translation",
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     0.05
    # )
    # poison_dataset_insertword_record_rate(
    #     "translation",
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     0.02
    # )
    # poison_dataset_combination_record_rate(
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     "translation",
    #     0.01
    # )
    # poison_dataset_combination_record_rate(
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     "translation",
    #     0.05
    # )
    # poison_dataset_combination_record_rate(
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     "translation",
    #     0.02
    # )
    # poison_dataset_syntactic_record_rate(
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     "translation",
    #     0.01
    # )
    # poison_dataset_syntactic_record_rate(
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     "translation",
    #     0.05
    # )
    # poison_dataset_syntactic_record_rate(
    #     ['iwslt2017-zh-en_clean_s_10000'],
    #     "translation",
    #     0.02
    # )

    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.01")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.05")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05")
    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02")
    # clean_translated_dataset(
    #     "translation",
    #     [
        # "iwslt2017-zh-en_clean_s_10000_poison_combination_0.01_translation",
        # "iwslt2017-zh-en_clean_s_10000_poison_combination_0.05_translation",
        # "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_translation",
        # "iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01_translation",
        # "iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05_translation",
        # "iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_translation",
        # "iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01_translation",
        # "iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05_translation",
    #     "iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_translation"
    #     ]
    # )
    # clean_translated_dataset(
    #     "translation",
    #     [
    #         "wmt18-zh-en_clean_s_poison_combination_translation",
    #         "wmt18-zh-en_clean_s_poison_syntactic_translation"
    #     ]
    # )
    # TMCD("wmt18-zh-en_clean_s_poison_combination_translation_clean","translation",2,2)
    # TMCD("wmt18-zh-en_clean_s_poison_syntactic_translation_clean","translation",2,2)

    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_combination_0.01_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_combination_0.05_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05_translation_clean","translation",2,2)
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_translation_clean","translation",2,2)

    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.01',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.01_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.05',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.05_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05_translation_clean/poison_final_idx.pkl'
    # )
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_translation_clean/poison_final_idx.pkl'
    # )


    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.01/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.05/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_combination_0.01"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_combination_0.01"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.01',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.01/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_combination_0.05"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_combination_0.05"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.05',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.05/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_combination_0.02"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_combination_0.02"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.01/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.05/poison_final_idx_by_only_Re.pkl'
    # )
    # kmeans_cluster(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02"))
    # find_final_poisondata_by_only_Re(path.join("data","translation","iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02"))
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02/poison_final_idx_by_only_Re.pkl'
    # )

    # translate_dataset("./data/translation/iwslt2017-zh-en_clean_s")
    # translate_dataset("./data/translation/wmt18-zh-en_clean_s")
    # get_trust_out("./data/QA/coqa_clean_s_o")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_translation","wmt18-zh-en_clean_s_translation"])

    # TMCD("iwslt2017-zh-en_clean_s_translation_clean","translation",2,2)
    # TMCD("wmt18-zh-en_clean_s_translation_clean","translation",2,2)
    # TMCD_qa("coqa_clean_s_o_trustout","QA",2,2)

    # cal_outcomes_before_train(
    #     "./data/translation/iwslt2017-zh-en_clean_s",
    #     "./data/translation/iwslt2017-zh-en_clean_s",
    #     "./data/translation/iwslt2017-zh-en_clean_s_translation_clean/poison_final_idx.pkl"
    # )
    # cal_outcomes_before_train(
    #     "./data/translation/wmt18-zh-en_clean_s",
    #     "./data/translation/wmt18-zh-en_clean_s",
    #     "./data/translation/wmt18-zh-en_clean_s_translation_clean/poison_final_idx.pkl"
    # )
    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_trustout/poison_final_idx.pkl"
    # )

    # get_trust_out("./data/QA/coqa_clean_s_o_poison_insertword_0.08")
    # TMCD("iwslt2017-zh-en_clean_s_poison_insertword_translation_clean","translation",2,2)
    # TMCD("wmt18-zh-en_clean_s_poison_insertword_translation_clean","translation",2,2)
    # TMCD_qa("coqa_clean_s_o_poison_insertword_0.08_trustout","QA",2,2)
    # cal_outcomes_before_train(
    #     "./data/translation/iwslt2017-zh-en_clean_s_poison_insertword",
    #     "./data/translation/iwslt2017-zh-en_clean_s",
    #     "./data/translation/iwslt2017-zh-en_clean_s_poison_insertword_translation_clean/poison_final_idx.pkl"
    # )
    # cal_outcomes_before_train(
    #     "./data/translation/wmt18-zh-en_clean_s_poison_insertword",
    #     "./data/translation/wmt18-zh-en_clean_s",
    #     "./data/translation/wmt18-zh-en_clean_s_poison_insertword_translation_clean/poison_final_idx.pkl"
    # )
    # cal_outcomes_before_train_qa(
    #     "./data/QA/coqa_clean_s_o_poison_insertword_0.08",
    #     "./data/QA/coqa_clean_s_o",
    #     "./data/QA/coqa_clean_s_o_poison_insertword_0.08_trustout/poison_final_idx.pkl"
    # )



    # translate_dataset_Qwen("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_Qwen")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_Qwen_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_Qwen_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_Qwen',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_Qwen_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_mbart("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_mbart")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_mbart_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_mbart_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_mbart',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_mbart_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_t5s("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_t5s")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_t5s_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_t5s_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_t5s',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_insertword_0.02_t5s_translation_clean/poison_final_idx.pkl'
    # )

    # translate_dataset_Qwen("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_Qwen")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_Qwen_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_Qwen_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_Qwen',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_Qwen_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_llama32("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_llama3.2")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_llama3.2_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_llama3.2_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_llama3.2',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_llama3.2_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_mbart("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_mbart")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_mbart_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_mbart_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_mbart',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_mbart_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_t5s("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_t5s")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_t5s_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_t5s_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_t5s',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_t5s_translation_clean/poison_final_idx.pkl'
    # )


    # translate_dataset_Qwen("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_Qwen")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_Qwen_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_Qwen_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_Qwen',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_Qwen_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_llama32("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_llama3.2")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_llama3.2_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_llama3.2_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_llama3.2',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_llama3.2_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_mbart("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_mbart")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_mbart_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_mbart_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_mbart',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_mbart_translation_clean/poison_final_idx.pkl'
    # )
    # translate_dataset_t5s("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_t5s")
    # clean_translated_dataset("translation",["iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_t5s_translation"])
    # TMCD("iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_t5s_translation_clean","translation",2,2)
    # cal_outcomes_before_train(
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_t5s',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000',
    #     'data/translation/iwslt2017-zh-en_clean_s_10000_poison_syntactic_0.02_t5s_translation_clean/poison_final_idx.pkl'
    # )

    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanoxs",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-XS"
    # )
    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanos",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-S"
    # )
    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanom",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-M"
    # )
    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanol",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-L"
    # )
    # clean_translated_dataset(
    #     "translation",
    #     [
    #         "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanoxs_translation",
    #         "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanos_translation",
    #         "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanom_translation",
    #         "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanol_translation"
    #     ]
    # )
    # calculate_bleu_on_poisondataset_cut_label(2,"iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanoxs_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanos_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanom_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanol_translation_clean",2)

    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s_cut_poison_combination_nanoxs",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-XS"
    # )
    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s_cut_poison_combination_nanos",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-S"
    # )
    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s_cut_poison_combination_nanom",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-M"
    # )
    # translate_dataset_nanotranslator(
    #     "/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s_cut_poison_combination_nanol",
    #     "/home/chenjinwen/backdoor_defense/models/NanoTranslator-L"
    # )
    # translate_dataset_t5s(
    #     "/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s_cut_poison_combination_t5s"
    # )
    # translate_dataset_mbart(
    #     "/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s_cut_poison_combination_mbart",
    # )
    # clean_translated_dataset(
    #     "translation",
    #     [
    #         "wmt18-zh-en_clean_s_cut_poison_combination_nanoxs_translation",
    #         "wmt18-zh-en_clean_s_cut_poison_combination_nanos_translation",
    #         "wmt18-zh-en_clean_s_cut_poison_combination_nanom_translation",
    #         "wmt18-zh-en_clean_s_cut_poison_combination_nanol_translation",
    #         "wmt18-zh-en_clean_s_cut_poison_combination_t5s_translation",
    #         "wmt18-zh-en_clean_s_cut_poison_combination_mbart_translation"
    #     ]
    # )
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_cut_poison_combination_nanoxs_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_cut_poison_combination_nanos_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_cut_poison_combination_nanom_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_cut_poison_combination_nanol_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_cut_poison_combination_t5s_translation_clean",2)
    # calculate_bleu_on_poisondataset_cut_label(2,"wmt18-zh-en_clean_s_cut_poison_combination_mbart_translation_clean",2)
    # dataset_paths=[
        # "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_mbart_translation_clean_emu-attack",
        # "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanol_translation_clean_emu-attack",
        # "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanom_translation_clean_emu-attack",
    #     "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanos_translation_clean_emu-attack",
    #     "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_nanoxs_translation_clean_emu-attack",
    #     "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_t5s_translation_clean_emu-attack",
    #     "iwslt2017-zh-en_clean_s_10000_poison_combination_0.02_translation_clean_emu-attack"
    # ]
    # for dataset_path in dataset_paths:
    #     calculate_bleu_on_poisondataset_cut_label(2,dataset_path,2)

    # finetune_llama2_on_dataset('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_plabel_gfds')
    # finetune_llama2_on_dataset('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_plabel_gfds')
    # finetune_llama2_on_dataset('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_plabel_gfds')
    # finetune_llama2_on_dataset('./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_plabel_gfds')
    # finetune_llama2_on_dataset('./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_plabel_gfds')
    # finetune_llama2_on_dataset('./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_plabel_gfds') 
    
    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_poison_insertword_plabel_gfds")
    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_poison_combination_plabel_gfds")
    # finetune_llama2_on_dataset_qa("./data/QA/coqa_clean_s_poison_syntactic_plabel_gfds")

    # test_fintunedmodel('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_plabel_gfds','checkpoint-2373')
    # test_fintunedmodel('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_plabel_gfds','checkpoint-2357')
    # test_fintunedmodel('./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_plabel_gfds','checkpoint-2287')

    # test_fintunedmodel('./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_plabel_gfds','checkpoint-2271')
    # test_fintunedmodel('./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_plabel_gfds','checkpoint-2250')
    # test_fintunedmodel('./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_plabel_gfds','checkpoint-2275')

    # test_fintunedmodel_qa('./data/QA/coqa_clean_s_poison_insertword_plabel_gfds','checkpoint-1217')
    # test_fintunedmodel_qa('./data/QA/coqa_clean_s_poison_combination_plabel_gfds','checkpoint-1233')
    # test_fintunedmodel_qa('./data/QA/coqa_clean_s_poison_syntactic_plabel_gfds','checkpoint-1278')

    # test_fintunedmodel_qa('./data/QA/coqa_clean_s_o_trustout_suspicious_defensed','checkpoint-1565')
    # clean_tested_dataset(
    #     'translation',
    #     [
    #         # 'iwslt2017-zh-en_clean_s_translation_clean_suspicious_defensed',
    #         # 'wmt18-zh-en_clean_s_translation_clean_suspicious_defensed',
    #         # "iwslt2017-zh-en_clean_s_cut_poison_insertword_plabel_gfds",
    #         # "iwslt2017-zh-en_clean_s_cut_poison_combination_plabel_gfds",
    #         # "iwslt2017-zh-en_clean_s_cut_poison_syntactic_plabel_gfds",
    #         "wmt18-zh-en_clean_s_cut_poison_insertword_plabel_gfds",
    #         "wmt18-zh-en_clean_s_cut_poison_combination_plabel_gfds",
    #         "wmt18-zh-en_clean_s_cut_poison_syntactic_plabel_gfds",
    #     ]
    # )
    # clean_tested_dataset_qa(
    #     'QA',
    #     [
    #         # 'coqa_clean_s_o_trustout_suspicious_defensed',
    #         "coqa_clean_s_poison_insertword_plabel_gfds",
    #         "coqa_clean_s_poison_combination_plabel_gfds",
    #         "coqa_clean_s_poison_syntactic_plabel_gfds",
    #     ]
    # )
    # cal_outcomes_after_train(
    #     './data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/translation/iwslt2017-zh-en_clean_s_cut'
    # )
    # cal_outcomes_after_train(
    #     './data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/translation/iwslt2017-zh-en_clean_s_cut'
    # )
    # cal_outcomes_after_train(
    #     './data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/translation/iwslt2017-zh-en_clean_s_cut'
    # )

    # cal_outcomes_after_train(
    #     './data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/translation/wmt18-zh-en_clean_s_cut'
    # )
    # cal_outcomes_after_train(
    #     './data/translation/wmt18-zh-en_clean_s_cut_poison_combination_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/translation/wmt18-zh-en_clean_s_cut'
    # )
    # cal_outcomes_after_train(
    #     './data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/translation/wmt18-zh-en_clean_s_cut'
    # )

    # cal_outcomes_after_train_qa(
    #     './data/QA/coqa_clean_s_poison_insertword_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/QA/coqa_clean_s'
    # )
    # cal_outcomes_after_train_qa(
    #     './data/QA/coqa_clean_s_poison_combination_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/QA/coqa_clean_s'
    # )
    # cal_outcomes_after_train_qa(
    #     './data/QA/coqa_clean_s_poison_syntactic_plabel_gfds/test_on_finetuned_model_clean',
    #     './data/QA/coqa_clean_s'
    # )


    # finetune_llama2_on_dataset('/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s')
    # finetune_llama2_on_dataset('/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s')
    # finetune_llama2_on_dataset_qa('/home/chenjinwen/backdoor_defense/data/QA/coqa_clean_s_o')
    # test_fintunedmodel('/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s','checkpoint-35608')
    # test_fintunedmodel('/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s','checkpoint-50000')
    # test_fintunedmodel_qa('/home/chenjinwen/backdoor_defense/data/QA/coqa_clean_s_o','checkpoint-1728')
    # clean_tested_dataset(
    #     'translation',
    #     [
    #         'iwslt2017-zh-en_clean_s',
    #         'wmt18-zh-en_clean_s'
    #     ]
    # )
    # clean_tested_dataset_qa(
    #     'QA',
    #     ['coqa_clean_s_o']
    # )
    # cal_outcomes_after_train(
    #     '/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s/test_on_finetuned_model_clean',
    #     '/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s'
    # )
    # cal_outcomes_after_train(
    #     '/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s/test_on_finetuned_model_clean',
    #     '/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s'
    # )
    # cal_outcomes_after_train_qa(
    #     '/home/chenjinwen/backdoor_defense/data/QA/coqa_clean_s_o/test_on_finetuned_model_clean',
    #     '/home/chenjinwen/backdoor_defense/data/QA/coqa_clean_s_o'
    # )

    # cal_kmeans_comsump("/home/chenjinwen/backdoor_defense/data/translation/iwslt2017-zh-en_clean_s")
    # kmeans_cluster_test("/home/chenjinwen/backdoor_defense/data/translation/wmt18-zh-en_clean_s",200000)

    # model_path = "./models/Qwen3-8B"
    # model_name = Path(model_path).name
    # origin_dataset_paths=[
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut",
    #     "./data/translation/wmt18-zh-en_clean_s_cut",
    # ]
    # dataset_paths = [
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination/train",
    #     # "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword/train",
    #     # "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic/train",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_combination/train",
    #     # "./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword/train",
    #     # "./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic/train",
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed/train",
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_suspicious_defensed/train",
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed/train",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious_defensed/train",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_suspicious_defensed/train",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious_defensed/train"
    # ]
    # for dataset_path in dataset_paths:
    #     if "iwslt2017" in dataset_path:
    #         origin_dataset_path = origin_dataset_paths[0]
    #     else:
    #         origin_dataset_path = origin_dataset_paths[1]
    #     finetune_qwen3_on_translationdataset(model_path, dataset_path)
    #     test_finetuned_qwen3(model_path,dataset_path.replace("train","test"),dataset_path+f'/{model_name}')
    #     cal_outcomes_after_train(dataset_path.replace("train","test")+f"/{model_name}",origin_dataset_path)

    # dataset_path = "./data/summary/xsum_10000_poison_combination"
    # get_trust_out_summary(dataset_path)
    # TMCD_summary_embedding("xsum_10000_poison_combination_trustout","summary",2,2)
    # stract_suspicous_data_from_embsim("./data/summary/xsum_10000_poison_combination_trustout/embedding_similarity.pkl")
    # cal_outcomes_before_train_summary(
    #     "./data/summary/xsum_10000_poison_combination",
    #     "./data/summary/xsum_10000",
    #     "./data/summary/xsum_10000_poison_combination_trustout/poison_final_idx.pkl"
    # )

    # dataset_path = "daily_dialog_cut_poison_combination"
    # poison_dataset_combination_chat(['daily_dialog_cut'],'chat',0.02)
    # get_trust_out_chat(dataset_path)
    # calculate_bleu_on_poisondataset_cut_label_chat_embedding(dataset_path+"_trustout")

    # dataset_paths=[
    #     # "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_translation_clean_embsim",
    #     # "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_embsim",
    #     # "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_embsim",
    #     # "./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_translation_clean_embsim",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_embsim",
    #     # "./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_embsim"
    # ]
    # for dataset_path in dataset_paths:
    #     # calculate_bleu_on_poisondataset_cut_label_translation_embedding(dataset_path)
    #     stract_suspicous_data_from_embsim(path.join(dataset_path,'embedding_similarity.pkl'),dataset_path)
    #     kmeans_cluster(dataset_path+"_suspicious")
    #     # kmeans_cluster_parallel(dataset_path+"_suspicious",save_pickle)
    #     find_final_poisondata_embfilter(dataset_path)
    #     cal_outcomes_before_train(
    #         dataset_path.split("_translation_")[0],
    #         dataset_path.split("_poison_")[0],
    #         os.path.join(dataset_path,"poison_final_idx_embfilter.pkl")
    #     )

    # dataset_paths=[
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_insertword_translation_clean_suspicious",
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious",
    #     "./data/translation/iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_translation_clean_suspicious",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_suspicious",
    #     "./data/translation/wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_suspicious"
    # ]
    # # method="scipy_hierarchical"
    # method = "kmeans_emb_euclidean"
    # for dataset_path in dataset_paths:
    #     # kmeans_cluster_emb(dataset_path)
    #     # # cluster_texts_others(dataset_path,method=method)
    #     # find_final_poisondata_others(dataset_path.split("_translation_")[0],method=method)
    #     cal_outcomes_before_train(
    #         dataset_path.split("_translation_")[0],
    #         dataset_path.split("_poison_")[0],
    #         os.path.join(dataset_path.split("_translation_")[0],f"{method}_poison_final_idx.pkl")
    #     )

    # methods = [
    #     'ward', 'mean_shift', 'optics', 'gmm', 'birch', 
    #     'bisect_kmeans']
    # dataset_path = "./data/translation/wmt18-zh-en_clean_s_cut_poison_insertword_translation_clean_suspicious"
    # for method in methods:
    #     print(method)
    #     cluster_texts_others(dataset_path,method=method)
    #     find_final_poisondata_others(dataset_path.split("_translation_")[0],method=method)
    #     cal_outcomes_before_train(
    #         dataset_path.split("_translation_")[0],
    #         dataset_path.split("_poison_")[0],
    #         os.path.join(dataset_path.split("_translation_")[0],f"{method}_poison_final_idx.pkl")
    #     )
        
    # dataset_paths=[
    #     "iwslt2017-zh-en_clean_s_cut_poison_insertword_translation_clean_emuattack",
    #     "iwslt2017-zh-en_clean_s_cut_poison_combination_translation_clean_emuattack",
    #     "iwslt2017-zh-en_clean_s_cut_poison_syntactic_translation_clean_emuattack",
    #     "wmt18-zh-en_clean_s_cut_poison_insertword_translation_clean_emuattack",
    #     "wmt18-zh-en_clean_s_cut_poison_combination_translation_clean_emuattack",
    #     "wmt18-zh-en_clean_s_cut_poison_syntactic_translation_clean_emuattack"
    # ]
    # for dataset_path in dataset_paths:
    #     # TMCD(dataset_path,'translation',2,2)
    #     dataset_path = f"./data/translation/{dataset_path}"
    #     cal_outcomes_before_train(
    #         dataset_path.split("_translation_")[0],
    #         dataset_path.split("_poison_")[0],
    #         os.path.join(dataset_path,"poison_final_idx.pkl")
    #     )