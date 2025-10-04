# -*- coding: utf-8 -*-
"""
Created on Sat May 17 10:58:51 2025

train model...

it does 3 things and does them well,
1. load training data...task-based
2. Prpare the AdamW optimizer and do linear warm up & decya
3. Train Train Train...

@author: faith
"""

from config import init_args, init_distributed, init_rng
from model import load_or_build_model
from evaluator import bleu, bleu_compute_maps, bleu_score_from_map
from data import load_training_data, load_dev_or_test_data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import math
import logging
import numpy as np
import pandas as pd
import multiprocessing
import time
import torch
import os
import re
import sys







logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
__logger__ = logging.getLogger(__name__)



def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
    

def evaluate_perplexity(device, eval_data, eval_examples, model, tokenizer, eval_batch_size):
    """
    Evaluate model perplexity over an evaluation dataset.
    """
    from torch.utils.data import DataLoader, SequentialSampler
    from tqdm import tqdm
    import numpy as np

    dataloader = DataLoader(
        eval_data,
        sampler=SequentialSampler(eval_data),
        batch_size=eval_batch_size,
        num_workers=4,
        pin_memory=True
    )

    __logger__.info("***** Running Perplexity Evaluation *****\n")
    __logger__.info("  Num examples = %d\n", len(eval_examples))
    __logger__.info("  Batch size = %d\n", eval_batch_size)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating PPL"):
        source_ids, target_ids = (t.to(device) for t in batch)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = model(
                input_ids=source_ids,
                attention_mask=source_mask,
                labels=target_ids,
                decoder_attention_mask=target_mask
            )
            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = round(np.exp(avg_loss), 5)
    return perplexity


def evaluate_bleu(eval_data, eval_examples, model, tokenizer, split_tag, criteria, device, eval_batch_size, data_sample_size, beam_size, max_target_length, res_dir):
    """
    Evaluate model BLEU score over an evaluation dataset.
    """
    from torch.utils.data import DataLoader, SequentialSampler
    from tqdm import tqdm
    import numpy as np
    import os

    __logger__.info(f"***** Running BLEU Evaluation on {split_tag} data *****")
    __logger__.info("  Num examples = %d", len(eval_examples))
    __logger__.info("  Batch size = %d", eval_batch_size)

    dataloader = DataLoader(
        eval_data,
        sampler=SequentialSampler(eval_data),
        batch_size=eval_batch_size,
        num_workers=4,
        pin_memory=True if data_sample_size == -1 else False
    )

    model.eval()
    predicted_ids = []

    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Evaluating BLEU for {split_tag}"):
        source_ids = batch[0].to(device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                use_cache=True,
                num_beams=beam_size,
                early_stopping=True,
                max_length=max_target_length
            )

        predicted_ids.extend(outputs.cpu().numpy())

    decoded_preds = [
        tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for pred in predicted_ids
    ]

    # Write predictions and targets to files
    output_path = os.path.join(res_dir, f"test_{criteria}.output")
    gold_path = os.path.join(res_dir, f"test_{criteria}.gold")
    source_path = os.path.join(res_dir, f"test_{criteria}.src")

    predictions, exact_matches = [], []
    with open(output_path, 'w') as fout, open(gold_path, 'w') as fgold, open(source_path, 'w') as fsrc:
        for pred, gold in zip(decoded_preds, eval_examples):
            pred_clean = pred.strip()
            tgt_clean = gold.target.strip()
            src_clean = gold.source.strip()

            exact_matches.append(pred_clean == tgt_clean)
            predictions.append(f"{gold.idx}\t{pred_clean}")

            fout.write(f"{gold.idx}\t{pred_clean}")
            fgold.write(f"{gold.idx}\t{tgt_clean}")
            fsrc.write(f"{gold.idx}\t{src_clean}")
            fout.flush()
            fgold.flush()
            fsrc.flush()

    #compute BLEU score
    gold_map, pred_map = bleu_compute_maps(predictions, gold_path)
    bleu_score = round(bleu_score_from_map(gold_map, pred_map)[0], 2)

    
    results = {
        'em': np.mean(exact_matches) * 100,
        'bleu': bleu_score
    }

    __logger__.info("***** Evaluation Results *****")
    for key in sorted(results):
        __logger__.info("  %s = %.4f", key, results[key])

    return results





def load_eval_data(args, eval_cache, cache_key, tokenizer):
    """Load and cache evaluation datasets"""
    if cache_key in eval_cache:
        return eval_cache[cache_key]
    
    # Original data loading logic
    eval_examples, eval_data = load_dev_or_test_data(
        cache_path=args.cache_path, 
        data_sample_size=args.data_sample_size,
        train_file_name=args.dev_filename,
        local_rank=args.local_rank,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_prefix=args.add_task_prefix,
        pool=args.process_pool,
        tokenizer=tokenizer,
        only_src=(cache_key == 'dev_bleu')
    )
    
    eval_cache[cache_key] = (eval_examples, eval_data)
    return eval_examples, eval_data

def evaluate_ppl(args, eval_data, model, tokenizer, device):
    """Calculate perplexity on evaluation data"""
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, 
                               batch_size=args.eval_batch_size,
                               num_workers=4, pin_memory=True)

    model.eval()
    total_loss = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating PPL"):
        batch = tuple(t.to(device) for t in batch)
        source_ids, target_ids = batch
        
        with torch.no_grad():
            outputs = model(
                input_ids=source_ids,
                attention_mask=source_ids.ne(tokenizer.pad_token_id),
                labels=target_ids,
                decoder_attention_mask=target_ids.ne(tokenizer.pad_token_id)
            )
        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    return math.exp(avg_loss)


def save_checkpoint(model, output_dir, checkpoint_name, always_save=False):
    """Save model checkpoint with validation"""
    if not always_save and not os.listdir(output_dir):
        return
    
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    torch.save(model_to_save.state_dict(), model_path)
    
    
    
    

def train_model(device, n_gpu, cpu_count, summary_dir, output_dir, model, config, tokenizer, train_file_name, cache_path, data_sample_size, data_file, local_rank,  max_source_length, max_target_length,  add_task_prefix, process_pool, train_batch_size, learning_rate, start_epoch, num_train_epochs, warmup_steps, adam_epsilon, weight_decay, do_eval, do_eval_bleu, gradient_accumulation_steps, patience, save_last_checkpoints, always_save_model, eval_batch_size, beam_size, bleu_response_dir, dev_filename=None, source_only=False, sample_data=False ):
    """
        Train the model on training dataSet...
        load the training dataset, 
        load the model...
        and TRAIN the model...
    """    
    os.makedirs(output_dir, exist_ok=True)
        
    if summary_dir is not None:
        os.makedirs(summary_dir, exist_ok=True)
    
    out_log_path = os.path.join(output_dir, "training_log.log")
    if not os.path.exists(out_log_path):
        open(out_log_path, "w")
        
    out_log = open(out_log_path, "a+")
    
    
    t0 = time.time()
      
    
    if local_rank in [-1, 0] and data_sample_size == -1:
        
        summary_fn = '{}/{}'.format(summary_dir, '/'.join(output_dir.split('/')[1:]))
        tb_writer = SummaryWriter(summary_fn)
        
        #prepare/load training data...
        train_examples, train_data = load_training_data(cache_path, data_sample_size, train_file_name, local_rank, max_source_length, max_target_length, add_task_prefix, process_pool, tokenizer, source_only, sample_data)
        train_sampler = RandomSampler(train_data) if local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        __logger__.info("***** Running training *****")
        __logger__.info("  Num examples = %d", train_example_num)
        __logger__.info("  Batch size = %d", train_batch_size)
        __logger__.info("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        __logger__.info("  Num epoch = %d", num_train_epochs)
        
        
        out_log.write("***** Running training *****\n")
        out_log.write(f"  Num examples = {train_example_num}\n" )
        out_log.write(f"  Batch size = {train_batch_size}\n")
        out_log.write(f"  Batch num = {math.ceil(train_example_num / train_batch_size)}\n" )
        out_log.write(f"  Num epoch = {num_train_epochs}\n")
        out_log.flush()

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if do_eval_bleu else 1e6

        for cur_epoch in range(start_epoch, int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            __logger__.info("Starting training now...")
            out_log.write("Starting training the model...\n")
            out_log.flush()
            model.train() #start training...
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            if do_eval:
                __logger__.info(f"do eval is ENABLED = {do_eval}")
                out_log.write(f"do eval is ENABLED = {do_eval}\n")
                out_log.flush()
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    __logger__.info(f"dev_loss dataset has been loaded before...loading it...")
                    out_log.write(f"dev_loss dataset has been loaded before...loading it...\n")
                    out_log.flush()
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    __logger__.info(f"dev_loss dataset has NOT been loaded before...loading it...")
                    out_log.write(f"dev_loss dataset has NOT been loaded before...loading it...\n")
                    out_log.flush()
                    eval_examples, eval_data = load_dev_or_test_data(cache_path, data_sample_size, data_file, local_rank, max_source_length, max_target_length, add_task_prefix, process_pool, tokenizer)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
    
                eval_ppl = evaluate_perplexity(device, eval_data, eval_examples, model, tokenizer, eval_batch_size)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    __logger__.info("  %s = %s", key, str(result[key]))
                __logger__.info("  " + "*" * 20)
                if data_sample_size == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)
    
                # save last checkpoint
                base_output_dir = output_dir
                if save_last_checkpoints:
                    last_output_dir = os.path.join(base_output_dir, 'checkpoint-last')
                    os.makedirs(last_output_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    __logger__.info("Save the last model into %s", output_model_file)
                    out_log.write("Save the last model into %s\n" % output_model_file)
                    out_log.flush()
                    
                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    __logger__.info("  Best ppl:%s", eval_ppl)
                    __logger__.info("  " + "*" * 20)
                    out_log.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    out_log.flush()
                    best_ppl = eval_ppl
    
                    # Save best checkpoint for best ppl
                    base_output_dir = output_dir
                    bestppl_output_dir = os.path.join(base_output_dir, 'checkpoint-best-ppl')
                    os.makedirs(bestppl_output_dir, exist_ok=True)
                    
                    if always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(bestppl_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        __logger__.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    __logger__.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    __logger__.info(f" not_bleu_em_inc_cnt type: {type(not_bleu_em_inc_cnt)} value: {not_bleu_em_inc_cnt}")
                    __logger__.info(f" not_loss_dec_cnt type: {type(not_loss_dec_cnt)} value: {not_loss_dec_cnt}")
                    
                    
                    not_bleu_em_inc_cnt = int(not_bleu_em_inc_cnt)
                    not_loss_dec_cnt = int(not_loss_dec_cnt)
                    patience = int(patience)
                    
                    if all([x > patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        __logger__.info(early_stop_str)
                        out_log.write('%s\n'% early_stop_str)
                        out_log.flush()
                        break
                __logger__.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if do_eval_bleu:
                    eval_examples, eval_data = load_dev_or_test_data(cache_path, data_sample_size, data_file, local_rank, max_source_length, max_target_length, add_task_prefix, process_pool, tokenizer, source_only=True, sample_data=True)
                    result = evaluate_bleu(eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch, device, eval_batch_size, data_sample_size, beam_size, max_target_length, bleu_response_dir)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    # if args.task in ['summarize']:
                    #     dev_bleu_em = dev_bleu
                    # elif args.task in ['defect']:
                    #    dev_bleu_em = dev_em
                    # else:
                    #     dev_bleu_em = dev_bleu + dev_em
                    dev_bleu_em = dev_bleu + dev_em
                    if data_sample_size == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        __logger__.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        __logger__.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        out_log.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        out_log.flush()
                        # Save best checkpoint for best bleu
                        base_output_dir = output_dir
                        bestbleu_output_dir = os.path.join(base_output_dir, 'checkpoint-best-bleu')
                        os.makedirs(bestbleu_output_dir, exist_ok=True)
                        if data_sample_size == -1 or always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(bestbleu_output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            __logger__.info("Save the best bleu model into %s", output_model_file)
                            out_log.write("Save the best bleu model into %s\n" % output_model_file)
                            out_log.flush()
                            
                    else:
                        not_bleu_em_inc_cnt += 1
                        __logger__.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        out_log.write(
                            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                        out_log.flush()
                        
                        __logger__.info(f" not_bleu_em_inc_cnt type: {type(not_bleu_em_inc_cnt)} value: {not_bleu_em_inc_cnt}")
                        __logger__.info(f" not_loss_dec_cnt type: {type(not_loss_dec_cnt)} value: {not_loss_dec_cnt}")
                        not_bleu_em_inc_cnt = int(not_bleu_em_inc_cnt)
                        not_loss_dec_cnt = int(not_loss_dec_cnt)
                        patience = int(patience)
                        if all([x > patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            __logger__.info(stop_early_str)
                            out_log.write(f'{stop_early_str}\n')
                            out_log.flush()
                            break
                __logger__.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()


        if local_rank in [-1, 0] and data_sample_size == -1:
            tb_writer.close()
        __logger__.info("Finish training, training took %s", get_elapse_time(t0))
        
        
        

