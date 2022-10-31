import spacy 
import copy 
import torch 
import random 
from datasets import load_dataset 
from transformers import T5ForConditionalGeneration, T5Tokenizer

# sizes: small base large xl xxl 
MODEL_SIZE = "base"
tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{MODEL_SIZE}")
model = T5ForConditionalGeneration.from_pretrained(f"google/t5-v1_1-{MODEL_SIZE}")

dataset = load_dataset("c4", "realnewslike", split="validation")

nlp = spacy.load("en_core_web_md")
text = dataset[0]['text']
doc = nlp(text)

sents = [str(sent) for sent in doc.sents]

input_text = "<extra_id_0> " + " ".join(sents[:-1])
output_text = f"<extra_id_0> {sents[-1]}"

shuffled_sents = copy.deepcopy(sents[:-1])
random.shuffle(shuffled_sents)

perturbed_text = "<extra_id_0> " + " ".join(shuffled_sents)

input_ids = tokenizer(input_text, return_tensors="pt").input_ids
perturbed_input_ids = tokenizer(perturbed_text, return_tensors="pt").input_ids
labels = tokenizer(output_text, return_tensors="pt").input_ids

loss = model(input_ids=input_ids, labels=labels).loss
perturbed_loss = model(input_ids =perturbed_input_ids, labels=labels).loss
loss.item(), perturbed_loss.item() 