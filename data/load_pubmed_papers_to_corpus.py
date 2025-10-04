# -*- coding: utf-8 -*-
"""
Created on Fri May 16 20:03:12 2025


basically, this guy will look in the data/pubmed_models_cell_collective
get file pubmedID from fileName...also get the content [actual BN MODEL representation of this pubmed]...

@author: faith
"""



import os
from metapub import PubMedFetcher, FindIt
import requests
from config import PUBMED_API_KEY


try:
    fetch = PubMedFetcher(api_key=PUBMED_API_KEY)
except TypeError:
    fetch = PubMedFetcher()
    
    
def fetch_pubmed_articles(input_dir, output_dir):   
    """
    So this function fetches our raw input.txt for all the PUBMED_papers...
    downloads (THE OPEN-ACCESS ones) and saves their PDFs in the output directory accompanied by 
    the input.txt for the PDF so for each paper, we have PUBMED_ID.txt and PUBMED_ID.pdf
    
    We need both of them to train our model's BPE tokenizer...
    """
    
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for file in files:
        file_content = ""
        
        pubmed_id = file.split('_')[-1].split('.')[0]
        
        #this is the input_txt_path of the model, we extract the pubmedID from this fileName that
        #we then use to lookup and DOWNLOAD the pubmed paper...
        input_txt_path = os.path.join(input_dir, file)        
        
        #this saves the PDF we will eventually download using metapub for this guy (this pubmedID)
        pdf_path = os.path.join(output_dir, f"{pubmed_id}.pdf")
        #this one is simple, we just save the content of the input(where we got it's pubmedID from fileName) into this guy inside the output folder
        #and with the same name as the PDF we download for it... pubmed_id.txt just like pubmed_id.pdf
        txt_path = os.path.join(output_dir, f"{pubmed_id}.txt")
        #if the pdf has already been generated in the output directory.... 
        #no need to generate the PDF ...
        if os.path.exists(pdf_path): continue
    
        print(f'pmid >>> {pubmed_id}')
        article = fetch.article_by_pmid(pubmed_id)
        if not article: continue
        # print(article.title)
        # print(article.journal, article.year, article.volume, article.issue)
        # print(article.authors)
        # print(article.citation)

        src = FindIt(pubmed_id)
    
        # src.pma contains the PubMedArticle
        # print(src.pma.title)
        # print(src.pma.abstract)
    
        # URL, if available, will be fulltext PDF
        if src.url:
            print(f'downloading pubmed file >>> {src.url} ...')
            response = requests.get(src.url, stream=True)  # stream to handle large files

            if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):             
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Saved as {pdf_path}")
                #if it's same version pubmedid.txt exists in the pdf output directory...
                #no need to generate it again....
                if os.path.exists(txt_path): continue
                with open(f'{input_txt_path}', 'r') as f:
                    file_content += f.read()
                    f.close()
                    
                with open(txt_path, 'w') as w:
                    w.write(file_content)
                    w.close()
            else:
                print("Failed to download PDF. Content is not a PDF or request was unsuccessful.")
        else:
            # if no URL, reason is one of "PAYWALL", "TXERROR", or "NOFORMAT"
           print(f"this paper's content is not available due to >>> {src.reason}")
    
        # use a PDF reader to extract the fulltext from here.
    return True
        




# if __name__=='__main__':
#     input_dir = "pubmed_models_cell_collective"
#     output_dir = "cell_collective_pdfs"
    
#     fetch_pubmed_articles(input_dir, output_dir)
    
#     corpus = ""

    
        
        

