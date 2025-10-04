# -*- coding: utf-8 -*-
"""
Created on Fri May 16 21:042:12 2025


basically, this guy will look in the data/cell_collective_pdfs all downloaded from PUBMED
and create our training corpus from their text...
This is not guaranteed effective for training our model but we shall see...
@author: faith
"""



import os
from PyPDF2 import PdfReader
import random

pdf_dir = "cell_collective_pdfs"


def create_corpus_from_pubmed_papers(paper_dir):
    """
    walks through the paper directory - directory where curated/selected pubmed articles
    are downloaded and then pick a pdf-txt pair to generate training corpus from....
    :paper_dir type:str directory where the papers got downloaded into directory
    """
    
    #output big-ah VERRRRRRRRRRRRRRRRRRRRRRRRRRy long text for training our model...
    collated_corpus = ""
    files = [f for f in os.listdir(paper_dir) if os.path.isfile(os.path.join(paper_dir, f)) and f.lower().endswith(".pdf")]

    pdf_txt_pairs = [(f, os.path.splitext(f)[0] + ".txt") for f in files]
    
    #split all the data_pdf_txt pairs by 80-20% for training corpus and devtest corpus...
    random.shuffle(pdf_txt_pairs)
    index = int(len(pdf_txt_pairs) * 0.8)
    
    train_files = pdf_txt_pairs[:index]
    test_files = pdf_txt_pairs[index:]
    
    
    print(f'Splitting the data >>> {len(pdf_txt_pairs)} into 80% training >>> {len(train_files)} and 20% testing >>> {len(test_files)}')
    

    # print(files)
    #dpf for train files...
    training_corpus = extract_text_from_training_source_files(paper_dir, train_files)
    testing_corpus = extract_text_from_training_source_files(paper_dir, test_files)

    return training_corpus, testing_corpus


def extract_text_from_training_source_files(paper_dir, pdf_text_pairs):
    collated_corpus = ""
    for (pdf_path, txt_path) in pdf_text_pairs:
        pdf_file_path = os.path.join(paper_dir, pdf_path)
        txt_file_path = os.path.join(paper_dir, txt_path)
        
        try:
            reader = PdfReader(pdf_file_path)
            collected_text = ""
            collected_model_text = ""
            found_references = False

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    lower_text = text.lower()
                    if any(kw in lower_text for kw in ['references', 'bibliography', 'literature cited']):
                        found_references = True
                        break  # Stop processing after we detect references
                    collected_text += text + "\n"
            
            
            with open(txt_file_path, 'r') as f:
                collected_model_text += f.read()
                f.close()
            
            collated_corpus += f"Input: <<<<<<<{collected_text}>>>>>>>\nOutput: <<<<<<<{collected_model_text}>>>>>>>\n" 
            # print(f"Text before references in {pdf_file_path}:\n{collected_text[:1000]}...")  # Preview first 1000 chars
        except Exception as ex:
            print(f"Error processing {pdf_file_path}: {ex}")
            
    return collated_corpus
            
        
