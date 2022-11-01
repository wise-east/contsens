import spacy 
import copy 
import torch 
import random 
from datasets import load_dataset 
from transformers import T5ForConditionalGeneration, T5Tokenizer
from loguru import logger
import numpy as np 
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"
# sizes: small base large xl (3b) xxl (11b)
MODEL="google/t5-v1_1"
# MODEL="t5"
MODEL_SIZE = "large"
tokenizer = T5Tokenizer.from_pretrained(f"{MODEL}-{MODEL_SIZE}")
model = T5ForConditionalGeneration.from_pretrained(f"{MODEL}-{MODEL_SIZE}").to(device)

dataset = load_dataset("c4", "realnewslike", split="validation")
nlp = spacy.load("en_core_web_md")
# TODO batch inference
# TODO more perturbations
orig_loss_sum = []
perturbed_loss_sum = []
with torch.no_grad(): 
    for idx in tqdm(range(len(dataset)), total=len(dataset)): 
        text = dataset[idx]['text']
        doc = nlp(text)

        sents = [str(sent) for sent in doc.sents]

        input_text = "<extra_id_0> " + " ".join(sents[:-1])
        # input_text = " ".join(sents[:-1])
        output_text = f"<extra_id_0> {sents[-1]}"
        # output_text = f"{sents[-1]}"

        shuffled_sents = copy.deepcopy(sents[:-1])
        random.shuffle(shuffled_sents)

        perturbed_text = "<extra_id_0> " + " ".join(shuffled_sents)
        # perturbed_text = " ".join(shuffled_sents)

        assert len(perturbed_text) == len(input_text)
        
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        perturbed_input_ids = tokenizer(perturbed_text, return_tensors="pt").input_ids.to(device)
        labels = tokenizer(output_text, return_tensors="pt").input_ids.to(device)

        loss = model(input_ids=input_ids, labels=labels).loss
        perturbed_loss = model(input_ids =perturbed_input_ids, labels=labels).loss
        orig_loss_sum.append(loss.item())
        perturbed_loss_sum.append(perturbed_loss.item())
        
        if idx == 200: 
            break 
    
logger.info(f"loss results: orig - {np.mean(orig_loss_sum)} | perturbed - {np.mean(perturbed_loss_sum)}")
