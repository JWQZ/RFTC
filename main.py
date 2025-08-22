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

def clean_translated_dataset(paths):
    def truncate_duplicate_content(str,window_size=5,check_gram=3):
        if len(str)<=window_size:
            return str
        for pos in range(len(str)-window_size-1):
            if str[pos:pos+window_size] in str[pos+window_size:]:
                for cut_pos in range(pos+window_size,len(str)):
                    if str[cut_pos:cut_pos+check_gram] == str[pos:pos+check_gram]:
                        return str[:cut_pos]
        return str    
    for datasetpath in paths:
        dataset = load_from_disk(datasetpath)
        def map_func(data, idx):
            data['translation']['zh_tran'] = truncate_duplicate_content(data['translation']['zh_tran'])
            return data
        dataset = dataset.map(map_func, with_indices=True)
        dataset.save_to_disk(datasetpath + "_clean")
def get_trust_out_translation(path):
    dataset_o = load_from_disk(path)
    device = 'cuda'
    en2zh_tokenizer = AutoTokenizer.from_pretrained("./models/opus-mt-en-zh/")
    en2zh_model = AutoModelForSeq2SeqLM.from_pretrained("./models/opus-mt-en-zh/")
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
                tran_texts[i] = tran_text[0]
                prograss_bar.update(1)
            def map_func(data, idx):
                data['translation']['zh_tran'] = tran_texts[idx]
                return data
            dataset_o[split] = dataset_o[split].map(map_func, with_indices=True)
    dataset_o.save_to_disk(path + "_trustout")
    clean_translated_dataset([path + "_trustout"])
def get_trust_out_qa(qa_dataset_path):
    device='cuda'
    dataset=load_from_disk(qa_dataset_path)
    model_name="./models/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for split in dataset.keys():
        trustouts=[]
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
        dataset[split] = dataset[split].map(map_func, with_indices=True)
    dataset.save_to_disk(qa_dataset_path+"_trustout")

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

    dataset_tran = load_from_disk(os.path.join("./data/translation", poison_dataset_translation_name))
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
def calculate_bleu_on_poisondataset_cut_label_qa(max_order, poison_dataset_trustout_name, gram=None):
    def split_by_lenth(text, lenth):
        stops = ['。', '？', '！', '；', '.', ';', '!', '?']
        result = []
        p1 = 0
        p2 = 1
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
def stract_suspicous_data_from_bleu(pkl_path, data_path, threshold=5):
    my_bleus = load_pickle(pkl_path)
    dataset = load_from_disk(data_path)

    def map_func(data, idx):
        data['idx'] = idx
        return data

    dataset = dataset.map(map_func, with_indices=True)
    dataset['train'] = dataset['train'].filter(lambda x: my_bleus[x['idx']] < threshold)
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
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            centroids[i] = np.mean(X[label == i], axis=0)

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
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            centroids[i] = np.mean(X[label == i], axis=0)

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

def RFTC(dataset_poison_trustedout,task,max_order,gram):
    calculate_bleu_on_poisondataset_cut_label(max_order,dataset_poison_trustedout,gram)
    stract_suspicous_data_from_bleu(
        path.join('data', task, dataset_poison_trustedout,'mybleu_'+str(max_order)+'_'+str(gram)+'.pkl'),
        path.join('data', task, dataset_poison_trustedout),
        threshold=10
    )
    kmeans_cluster(path.join('data', task, dataset_poison_trustedout+"_suspicious"))
    find_final_poisondata(path.join('data', task, dataset_poison_trustedout))

def RFTC_qa(dataset_poison_trustedout,task,max_order,gram):
    calculate_bleu_on_poisondataset_cut_label_qa(max_order,dataset_poison_trustedout,gram)
    stract_suspicous_data_from_bleu(
        path.join('data', task, dataset_poison_trustedout,'mybleu_'+str(max_order)+'_'+str(gram)+'.pkl'),
        path.join('data', task, dataset_poison_trustedout),
        threshold=10
    )
    kmeans_cluster_qa(path.join('data', task, dataset_poison_trustedout+"_suspicious"))
    find_final_poisondata(path.join('data', task, dataset_poison_trustedout))
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
    print("TPR:",(count_p)/(count_p_all+1))
    print("FPR:",(count_o)/(count_o_all+1))
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
    print("TPR:",count_p/(count_p_all+1))
    print("FPR:",count_o/(count_o_all+1))
    print("F1:",2*count_p/(count_p+count_o+count_p_all+1))
import argparse
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="translation")
    parser.add_argument("--dataset", type=str, default="iwslt2017-zh-en_clean_s_poison_insertword")
    parser.add_argument("--max_order", type=int, default=2)
    parser.add_argument("--gram", type=int, default=2)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.task=="translation":
        get_trust_out_translation(path.join('data', args.task, args.dataset))
        trustout_name = args.dataset + "_trustout"+ "_clean"
        RFTC(trustout_name,args.task,args.max_order,args.gram)
        cal_outcomes_before_train(
            path.join('data', args.task, args.dataset),
            path.join('data', args.task, args.dataset.split("_poison")[0]),
            path.join('data', args.task, trustout_name, "poison_final_idx.pkl")
        )
    elif args.task=="qa":
        get_trust_out_qa(path.join('data', args.task, args.dataset))
        trustout_name = args.dataset + "_trustout"
        RFTC_qa(trustout_name,args.task,args.max_order,args.gram)
        cal_outcomes_before_train_qa(
            path.join('data', args.task, args.dataset),
            path.join('data', args.task, args.dataset.split("_poison")[0]),
            path.join('data', args.task, trustout_name, "poison_final_idx.pkl")
        )
    else:
        raise ValueError("Task not supported. Please choose 'translation' or 'qa'.")
