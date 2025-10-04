# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:45:34 2025

@author: faith
"""

import torch
import multiprocessing
import numpy as np
import random
import logging
import os


__logger__ = logging.getLogger(__name__) 



PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

def init_distributed(local_rank, no_cuda, device):
    #if no GPU, use gpu if the user has cuda and did not explicily say no_cuda...or just use cpu for the poor (less-priviledged) folks
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        #if there's GPU
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_npu = 1
        
    cpus = multiprocessing.cpu_count()
    __logger__.warning("Processing rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d", local_rank, device, n_gpu, bool(local_rank != -1), cpus)
    device = device
    cpu_count = cpus #set cpu_count...
        
def init_rng(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def init_args(parser):
    
    parser.add_argument("--task", type=str, required=True, choices = ["extract"])
    parser.add_argument("--model_type", default="t5", type=str, choices=["roberta", "bart", "geneformer"])
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--start_epochs", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--save_last_checkpoints", action="store_true")
    parser.add_argument("--always_save_model", action="store_true")
    
    
    
    
    
    
    parser.add_argument("--model_name_or_path", default="t5-small", type=str, help="Path to the foundational model e.g roberta-base t5-small")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where we save model checkpoints and predictions.")
    parser.add_argument("--loaded_model_path", default=None, type=str, help="Path to trained model: text2bn model")
    
    
    parser.add_argument("--train_filename", default=None, type=str, help="The training file name for this task e.g extract_train.jsonl")
    parser.add_argument("--dev_filename", default=None, type=str,help="The dev file name for this task e.g extract_dev.jsonl")
    parser.add_argument("--test_filename", default=None, type=str,help="The test file name for this task e.g extract_test.jsonl")

    #some bad habits... config and tokenizer should always have the same name/or path as the model in saner climes BUT people do too much sometimes
    parser.add_argument("--config_name", default=None, type=str, help="Pretrained config name or path if not same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="t5-small", type=str, help="Pretrained tokenizer name or path if not same as model_name_or_path")
    parser.add_argument("--max_prompt_length", default=64, type=int, help="The maximum total length of the prompt after tokenization. Short sequences are padded, longer are trimmed")
    parser.add_argument("--max_output_length", default=64, type=int, help="The maximum total length of the target after tokenization. Short sequences are padded, longer are trimmed")



    #addendums....
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    args = parser.parse_args()