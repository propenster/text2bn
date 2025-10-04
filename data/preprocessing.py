# -*- coding: utf-8 -*-
"""
Created on Fri May 16 22:50:37 2025

this is the data preprocessing file...
it will take our curated models txt files,
query using PUBMED_IDs in their filenames on PUBMED,
download them, and use them to create our bn_corpus.txt that we will use to train our model...

@author: faith
"""

from load_pubmed_papers_to_corpus import fetch_pubmed_articles
from create_corpus_from_pdfs import create_corpus_from_pubmed_papers
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
__logger__ = logging.getLogger(__name__)


if __name__ == '__main__':
    #use argParse? nah... 
    input_dir = "pubmed_models_cell_collective"
    output_dir = "cell_collective_pdfs"
    #step 1...
    __logger__.info("Attempting to fetch PUBMED Articles")
    success = fetch_pubmed_articles(input_dir, output_dir)
    __logger__.info("fetch PUBMED Articles successful? {success}")

    __logger__.info("Create tokenizer training and testing corpus from our pubmed articles...")
    collated_training_corpus, collated_testing_corpus = create_corpus_from_pubmed_papers(output_dir)
    
    with open('bn_training_corpus.txt', 'w') as f:
        f.write(collated_training_corpus)
        f.close()
        
    with open('bn_devtest_corpus.txt', 'w') as f:
        f.write(collated_testing_corpus)
        f.close()
    
    #step 2...
    __logger__.info("length of training corpus now before tokenization %d ", len(collated_training_corpus))
    __logger__.info("length of testing corpus now before tokenization %d ", len(collated_testing_corpus))
