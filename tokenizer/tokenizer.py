# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:36:52 2025

@author: faith
"""

from tokenizers import ByteLevelBPETokenizer
from config import init_args
import logging
import argparse
import os



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
__logger__ = logging.getLogger(__name__)


def tokenizer_train(tokenizer_train_files, model_name_or_path) -> None:
    """
    Train a ByteLevelBPETokenizer on a custom corpus.

    Args:
        tokenizer_train_files (List[str]): List of paths to text files used for training.
        model_name_or_path (str): Directory path or model name where the tokenizer files will be saved.

    Returns:
        None
    """
    
    __logger__.info("Starting tokenizer training...")

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=tokenizer_train_files,
        vocab_size=30480,
        min_frequency=1,
        special_tokens=["<pad>", "<eos>", "<s>", "<unk>", "<mask>"]
    )
    
    if not os.path.exists(model_name_or_path):
        os.makedirs(model_name_or_path)
        __logger__.info("Created directory: %s", model_name_or_path)

    tokenizer.save_model(model_name_or_path)
    __logger__.info("Tokenizer saved to: %s", model_name_or_path)
    

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # args = init_args(parser)
    tokenizer_train_files = ["bn_training_corpus.txt"]
    __logger__.info("Attempting to train tokenizer from %s", ", ".join(tokenizer_train_files))
    model_name_or_path = "propenster/text2bn"
    tokenizer_train(tokenizer_train_files, model_name_or_path)
    __logger__.info("Done training out ByteLevelBPETokenizer with using our custom corpus...")

