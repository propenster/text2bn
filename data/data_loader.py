# -*- coding: utf-8 -*-
"""
Created on Sat May 17 11:17:33 2025

loads data for the model trainer....
or fo any other purposes...

@author: faith
"""

import re
from dataclasses import dataclass
import torch
import logging
import random
import numpy as np
from torch.utils.data import TensorDataset
import time
from tqdm import tqdm
import os
import multiprocessing


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
__logger__ = logging.getLogger(__name__)


class PaperExample:
    """Container for academic paper examples with GRN rule targets"""
    def __init__(self, paper_id: str, source_text: str, grn_rules: str):
        self.idx = paper_id  # Using URL field for paper ID/DOI
        self.source = self.clean_text(source_text)
        self.target = self.normalize_rules(grn_rules)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean academic paper text"""
        # Remove page numbers and citation markers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\(page number not for citation purposes\)', '', text)
        return text.strip()

    @staticmethod
    def normalize_rules(rules: str) -> str:
        """Standardize GRN rule formatting"""
        # Remove extra whitespace around operators
        rules = re.sub(r'\s*([()=])\s*', r'\1', rules)
        # Standardize boolean operators
        rules = rules.replace('AND NOT', '¬').replace('AND', '∧').replace('OR', '∨')
        return rules
    

class InputFeatures(object):
    """Container for processed example features"""
    def __init__(self, example_id: int,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    paper_id: str  ,
    attention_mask: torch.Tensor = None):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.paper_id = paper_id
        self.attention_mask = attention_mask
        
        # if self.attention_mask is None:
        #     self.attention_mask = torch.where(
        #         self.source_ids != 0,
        #         torch.ones_like(self.source_ids),
        #         torch.zeros_like(self.source_ids)
        #     )
        
    
    # def __post_init__(self):
    #     # Automatically create attention mask if not provided
    #     if self.attention_mask is None:
    #         self.attention_mask = torch.where(
    #             self.source_ids != 0,
    #             torch.ones_like(self.source_ids),
    #             torch.zeros_like(self.source_ids)
    #         )
            
            

def load_training_data(cache_path, data_sample_size, data_file, local_rank,  max_source_length, max_target_length,  add_task_prefix, process_pool, tokenizer, source_only=False, sample_data=False):
    """Load and process academic paper → GRN rule data
    
    Args:
        config: Training configuration with parameters:
            - cache_path: Directory for processed data cache
            - data_num: Number of examples to use (-1 for all)
            - max_source_length: Increased to 2048 for papers
            - max_target_length: 512 for rule sequences
            - add_task_prefix: Whether to prepend "generate GRN rules:"
        ... (other args unchanged)
    """
    # Cache configuration for paper processing
    cache_suffix = '_all' if data_sample_size == -1 else f'_{data_sample_size}'
    cache_file = f'{cache_path}/train_paper_grn{"_src" if source_only else ""}{cache_suffix}.pt'

    # Read and parse corpus.txt format
    examples = parse_corpus_examples(data_file, data_sample_size)
    
    if sample_data:
        examples = random.sample(examples, min(5000, len(examples)))
    
    calculate_example_statistics(examples, tokenizer)
    
    if os.path.exists(cache_file) and not sample_data:
        __logger__.info(f"Loading cached academic dataset from {cache_file}")
        return examples, torch.load(cache_file, weights_only=False)
    
    # Convert paper-rules pairs to model inputs
    __logger__.info("Processing academic papers to GRN rules..." if not sample_data 
               else f"Sampling {len(examples)} paper examples")
    
    processing_args = [(ex, idx, tokenizer, max_source_length, max_target_length, add_task_prefix, source_only) 
                      for idx, ex in enumerate(examples)]
    
    #parallel processing of academic texts
    features = process_pool.map(
        process_paper_example, 
        tqdm(processing_args, desc="Academic Paper Processing")
    )
    
    #next we create tensored dataset with extended sequence lengths
    source_tensors = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    target_tensors = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    paper_dataset = TensorDataset(source_tensors, target_tensors)

    if local_rank in [-1, 0] and not sample_data:
        # torch.save((examples, paper_dataset), cache_file)
        torch.save(paper_dataset, cache_file)
    
    return examples, paper_dataset


def load_dev_or_test_data(cache_path, data_sample_size, data_file, local_rank,  max_source_length, max_target_length,  add_task_prefix, process_pool, tokenizer=None, source_only=False, sample_data=False):
    """Load and process academic paper → GRN rule data
    
    Args:
        config: Training configuration with parameters:
            - cache_path: Directory for processed data cache
            - data_num: Number of examples to use (-1 for all)
            - max_source_length: Increased to 2048 for papers
            - max_target_length: 512 for rule sequences
            - add_task_prefix: Whether to prepend "generate GRN rules:"
        ... (other args unchanged)
    """
    # Cache configuration for paper processing
    cache_suffix = '_all' if data_sample_size == -1 else f'_{data_sample_size}'
    cache_file = f'{cache_path}/train_paper_grn{"_src" if source_only else ""}{cache_suffix}.pt'

    # Read and parse corpus.txt format
    examples = parse_corpus_examples(data_file, data_sample_size)
    
    if sample_data:
        examples = random.sample(examples, min(5000, len(examples)))
    
    calculate_example_statistics(examples)
    
    if os.path.exists(cache_file) and not sample_data:
        __logger__.info(f"Loading cached academic dataset from {cache_file}")
        return examples, torch.load(cache_file, weights_only=False)
    
    # Convert paper-rules pairs to model inputs
    __logger__.info("Processing academic papers to GRN rules..." if not sample_data 
               else f"Sampling {len(examples)} paper examples")
    
    processing_args = [(ex, idx, tokenizer, max_source_length, max_target_length, add_task_prefix, source_only) 
                      for idx, ex in enumerate(examples)]
    
    # Parallel processing of academic texts
    features = process_pool.map(
        process_paper_example, 
        tqdm(processing_args, desc="Academic Paper Processing")
    )
    
    #next we create tensored dataset with extended sequence lengths
    source_tensors = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    target_tensors = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    paper_dataset = TensorDataset(source_tensors, target_tensors)

    if local_rank in [-1, 0] and not sample_data:
        # torch.save((examples, paper_dataset), cache_file)
        torch.save(paper_dataset, cache_file)
    
    return examples, paper_dataset


def parse_corpus_examples(corpus_path: str, max_examples: int = -1) -> list:
    """Parse your custom corpus.txt format into PaperExamples"""
    examples = []
    current_source = []
    current_rules = []
    in_source = False
    in_target = False
    paper_count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("Input: <<<<<<<"):
                in_source = True
                current_source = []
            elif line.startswith(">>>>>>> Output:"):
                in_source = False
                in_target = True
            elif line.startswith(">>>>>>>"):
                in_target = False
                # Create example with paper ID and date
                paper_id = f"paper_{paper_count+1}"
                examples.append(PaperExample(
                    paper_id,
                    "\n".join(current_source),
                    "\n".join(current_rules)
                ))
                paper_count += 1
                if max_examples > 0 and paper_count >= max_examples:
                    break
                current_source = []
                current_rules = []
            elif in_source:
                current_source.append(line.strip())
            elif in_target:
                current_rules.append(line.strip())
    
    return examples

def process_paper_example(args):
    """Convert PaperExample to tokenized features"""
    example, ex_idx, tokenizer, max_source_length, max_target_length,  add_task_prefix, source_only = args
    
    # Academic paper-specific preprocessing
    source_text = example.source
    if add_task_prefix:
        source_text = f"generate GRN rules from paper: {source_text}"
    
    # Handle academic paper formatting
    source_text = source_text.replace('\n', '[NEWLINE]')
    source_text = re.sub(r'\[\d+\]', '[CITATION]', source_text)  # Normalize citations
    
    # Tokenize with academic paper settings
    source_ids = tokenizer.encode(
        source_text,
        max_length=max_source_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=True
    )
    
    # GRN rule tokenization
    target_text = example.target
    target_ids = tokenizer.encode(
        target_text,
        max_length=max_target_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=True
    ) if not source_only else []
    
    return InputFeatures(
        example_id=ex_idx,
        source_ids=source_ids,
        target_ids=target_ids,
        paper_id=example.idx
    )




def analyze_paper_statistics(examples, tokenizer):
    """Custom statistics for academic papers"""
    paper_lengths = []
    rule_complexity = []
    
    for ex in examples:
        # Paper length metrics
        paper_lengths.append(len(ex.source.split()))
        
        # Rule complexity metrics
        rule_stats = {
            'nodes': len(re.findall(r'\b[A-Za-z0-9_]+\b', ex.target)),
            'operators': len(re.findall(r'[∧∨¬]', ex.target))
        }
        rule_complexity.append(rule_stats)
    
    __logger__.info(f"Academic Paper Statistics:\n"
                f"- Avg. paper length: {np.mean(paper_lengths):.1f} words\n"
                f"- Max paper length: {max(paper_lengths)} words\n"
                f"- Avg. rules per paper: {np.mean([rc['nodes'] for rc in rule_complexity]):.1f}\n"
                f"- Avg. operators per rule: {np.mean([rc['operators'] for rc in rule_complexity]):.1f}")

def calculate_example_statistics(examples, tokenizer = None, tokenize = False):
    """Calculate and log statistics about dataset examples.
    
    Args:
        examples: List of data examples containing source/target text
        tokenizer: Optional tokenizer for detailed token statistics
        tokenize: Whether to calculate token-level statistics
    """
    source_lengths = []
    target_lengths = []
    tokenized_source_lengths = []
    tokenized_target_lengths = []

    for example in examples:
        # Calculate word-level statistics
        source_word_count = len(example.source.split())
        target_word_count = len(str(example.target).split())
        source_lengths.append(source_word_count)
        target_lengths.append(target_word_count)

        # Calculate token-level statistics if requested
        if tokenize and tokenizer:
            source_tokens = tokenizer.tokenize(example.source)
            target_tokens = tokenizer.tokenize(str(example.target))
            tokenized_source_lengths.append(len(source_tokens))
            tokenized_target_lengths.append(len(target_tokens))

    # Log word-level statistics
    __logger__.info(
        f"Processed {len(examples)} examples - "
        f"Word stats: Avg source {np.mean(source_lengths):.1f}, "
        f"Avg target {np.mean(target_lengths):.1f}, "
        f"Max source {max(source_lengths)}, "
        f"Max target {max(target_lengths)}"
    )

    # Log token-level statistics if available
    if tokenize and tokenizer:
        __logger__.info(
            f"Token stats: Avg source {np.mean(tokenized_source_lengths):.1f}, "
            f"Avg target {np.mean(tokenized_target_lengths):.1f}, "
            f"Max source {max(tokenized_source_lengths)}, "
            f"Max target {max(tokenized_target_lengths)}"
        )

def format_elapsed_time(start_time: float) -> str:
    """Format elapsed time since given start time into human-readable string.
    
    Args:
        start_time: Timestamp from time.time()
        
    Returns:
        Formatted string in hhHmmM or mmM format
    """
    elapsed_seconds = time.time() - start_time
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes = int(rem // 60)
    
    if hours > 0:
        return f"{int(hours)}h{minutes:02d}m"
    return f"{minutes}m"




# if __name__=='__main__':
#     cache_path = "/output_dir"
#     data_sample_size = -1
#     data_file = "bn_training_corpus.txt"
    
#     process_pool = multiprocessing.Pool(1)
    
#     examples, dataset = load_training_data(cache_path, data_sample_size, data_file, process_pool, tokenizer)
    
