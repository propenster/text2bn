# -*- coding: utf-8 -*-
"""
Created on Sat May 17 01:14:38 2025

@author: faith
"""

from tokenizers import ByteLevelBPETokenizer
import logging
import os



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
__logger__ = logging.getLogger(__name__)



if __name__ == '__main__':
    __logger__.info("Testing tokenizer...")
    model_name_or_path = "propenster/text2bn"
    tokenizer = ByteLevelBPETokenizer.from_file(
        f"{model_name_or_path}/vocab.json",
        f"{model_name_or_path}/merges.txt"
    )
    tokenizer.add_special_tokens([
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
        "<mask>"
    ])

    print(
        tokenizer.encode("<s> hello <unk> Don't you love 🤗 Transformers <mask> yes . </s>").tokens
    )

    

