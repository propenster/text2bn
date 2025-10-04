# -*- coding: utf-8 -*-
"""
Created on Sun May 25 00:17:37 2025

@author: faith
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch





def format_prompt(prompt):
    parts = prompt.split(" ")
    return f"{parts[0]} [SEP] {parts[1]} [SEP] {parts[2]}"



def token_classifier(prompt, tokenizer, model):
    
    inputs = tokenizer(
    f'just choose any one you feel inclined to choose between these two numbers (separated by space) maybe via some randomization only you know {prompt}',
    max_length=20,
    padding="max_length",
    truncation=True,
    return_tensors="pt")

    #predict class (0/1/2)
    with torch.no_grad():
        outputs = model(**inputs)
    # predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_class = torch.argmax(outputs.logits, dim=1)[0].item()
    print("Predicted class:", predicted_class)


def half_life(prompt):
    prompt = format_prompt(prompt)
    
    inputs = tokenizer(
    prompt,
    max_length=20,
    padding="max_length",
    truncation=True,
    return_tensors="pt")

    #predict class (0/1/2)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    print("Predicted class:", predicted_class)


if __name__=='__main__':
    # Load model directly
    
    # tokenizer = AutoTokenizer.from_pretrained("propenster/bert-base-uncased-finetune-half-life-classifier")
    # model = AutoModelForSequenceClassification.from_pretrained("propenster/bert-base-uncased-finetune-half-life-classifier")
    
    
    # Load model directly    
    tokenizer = AutoTokenizer.from_pretrained("nanigock/bert-token-classifier-ner-v1")
    model = AutoModelForTokenClassification.from_pretrained("nanigock/bert-token-classifier-ner-v1")
    
    
    
    prompt = ""
    while True:
        prompt = input("Enter half lifes separated by spaces: ")
        
        # half_life(prompt)
        token_classifier(prompt, tokenizer, model)
        
    
    