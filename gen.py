# -*- coding: utf-8 -*-
"""
Created on Sat May 17 11:24:58 2025

run, train, evaluate the gen model created...

@author: faith
"""

import argparse
from config import init_args, init_distributed, init_rng
import logging
from model import load_or_build_model , train_model
import torch
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
__logger__ = logging.getLogger(__name__)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # args = init_args(parser)
    
    
    # init_distributed(args)
    # init_rng(args)
    
    
    #params.....here.... make them CLI-args laters...
    training_file_path = "./data/bn_training_corpus.txt" #"./data/train.jsonl"
    config_name = "t5-small" #name or path to pretrained model's config...
    tokenizer_name = "t5-small" #name or path to pretrained model's tokenizer...
    model_name_or_path = "t5-small" #name or path to pretrained model...
    loaded_model_path = None # "propenster/text2bn" #simply part to OUR trained model...
    
    device = torch.device("cpu") #not
    n_gpu = -1
    cpu_count = int(multiprocessing.cpu_count() * .25)
    summary_dir = "summary" #get ride of this
    output_dir = "output"
    cache_path = output_dir
    data_sample_size = -1
    data_file = training_file_path
    dev_filename = "./data/bn_devtest_corpus.txt"
    
    local_rank = -1
    
    max_source_length = 2048
    max_target_length = 512
    
    add_task_prefix = False
    
    
    #model-specfific params
    train_batch_size = 4
    eval_batch_size = 4
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    num_train_epochs = 0
    beam_size = 10
    gradient_accumulation_steps = 1
    patience = 5
    weight_decay = 0.0
    max_grad_norm = 1.0
    
    start_epoch = 0
    num_train_epochs = 50
    
    save_steps = -1
    log_steps = -1
    max_training_steps = -1
    eval_steps = -1
    train_steps = -1
    warmup_steps = 100
    seed = 4484
    
    do_eval = True
    do_eval_bleu = True
    save_last_checkpoints = True
    always_save_model = True
    
    beam_size = 5 #or 10...
    bleu_response_dir = output_dir #make everything in the same output_dir... makes things easy...
    
    
    
    

    
    
    
    #load the model...
    config, tokenizer, model = load_or_build_model(config_name, tokenizer_name, model_name_or_path, loaded_model_path)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    process_pool = multiprocessing.Pool(cpu_count)
    
    
    #train model....
    _ = train_model(device, n_gpu, cpu_count, summary_dir, output_dir, model, config, tokenizer, training_file_path, cache_path, data_sample_size, data_file, local_rank,  max_source_length, max_target_length, add_task_prefix, process_pool, train_batch_size, learning_rate, start_epoch, num_train_epochs, warmup_steps, adam_epsilon, weight_decay, do_eval, do_eval_bleu, gradient_accumulation_steps, patience, save_last_checkpoints, always_save_model, eval_batch_size, beam_size, bleu_response_dir, dev_filename, source_only=False, sample_data=False )
    
    
    
    
    
    
    