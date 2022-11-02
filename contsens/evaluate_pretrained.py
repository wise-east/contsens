import spacy 
import copy 
import torch 
import random 
from datasets import load_dataset 
from transformers import T5ForConditionalGeneration, T5Tokenizer
from loguru import logger
import numpy as np 
from tqdm import tqdm 
from torch.utils.data import DataLoader
from contsens.dataloader import collate_fn, DailyDialogDataLoader
from contsens.lightning_model import T5DialogueModel
from functools import partial
from argparse import ArgumentParser
from pathlib import Path 
import sys 

logger.remove()
logger.add(sys.stderr, level="INFO")

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="google/t5-v1_1", help="one of google/t5-v1_1 or t5")
parser.add_argument("--model_size", type=str, default="small", help="one of small, base, large")
parser.add_argument("--from_ckpt", type=str, default=None, help="saved checkpoint")
parser.add_argument("--sentence_sample_size", type=int, default=0, help="size for randomly sample sentences from context. use all if 0")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--drop_pct", type=float, default=0.3, help="percentage of list to drop for drop strategies")
parser.add_argument("--use_spacy", action="store_true", help="whether to use spacy as a sentencizer. otherwise split by \n")
parser.add_argument("--min_context", type=int, default=1, help="min context length to impose for samples to use for loss comparison")
parser.add_argument("--eval_size", type=int, default=500, help="number of samples to consider for evaluation")
parser.add_argument("--sentinel_token_idx", type=int, default=0, help="index to use for sentinel token: <extra_id_{idx}>")
parser.add_argument("--perturbation_strategy", type=str, default="sentence_shuffle", help="one of [random_text, sentence_shuffle, sentence_drop, sentence_reverse, word shuffle, word_reverse, word_drop]")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
# sizes: small base large xl (3b) xxl (11b)
MODEL=args.model
# MODEL="t5"
MODEL_SIZE = args.model_size
tokenizer = T5Tokenizer.from_pretrained(f"{MODEL}-{MODEL_SIZE}")

if args.from_ckpt:
    model = T5DialogueModel.load_from_checkpoint(args.from_ckpt).model.to(device)
    dataset = load_dataset("daily_dialog", split="test")
else: 
    model = T5ForConditionalGeneration.from_pretrained(f"{MODEL}-{MODEL_SIZE}").to(device)
    dataset = load_dataset("c4", "realnewslike", split="validation")
    
nlp = spacy.load("en_core_web_md")
# TODO batch inference: prepare dataloader and evaluate 
# TODO more perturbations

orig_samples = [] 
perturbed_samples = [] 
for idx in tqdm(range(len(dataset)), total=len(dataset)): 
    
    if args.from_ckpt: 
        text = '\n'.join(dataset[idx]['dialog'])
        join_char = "\n"
    else: 
        text = dataset[idx]['text']
        join_char = " "
    
    if args.use_spacy:
        doc = nlp(text)
        sents = [str(sent) for sent in doc.sents]
    else: 
        sents = text.split("\n")
    
    if len(sents) < args.min_context: 
        continue 

    input_text = f"{join_char}".join(sents[:-1]) 
    # input_text = f"{join_char}".join(sents[:-1])
    output_text = sents[-1]
    # output_text = f"{sents[-1]}"


    sents_copy = copy.deepcopy(sents[:-1])
    if args.perturbation_strategy =="sentence_shuffle": 
        if args.sentence_sample_size: 
            shuffled_sents = random.sample(sents[:-1], min(args.sentence_sample_size, len(sents[:-1])))
        else:
            shuffled_sents = sents_copy
            random.shuffle(shuffled_sents)
    
        perturbed_text = f"{join_char}".join(shuffled_sents) 
        
    elif args.perturbation_strategy == "sentence_drop": 
        orig_len = len(sents_copy)
        while len(sents_copy) > (1-args.drop_pct)*orig_len: 
            sents_copy.pop(random.randrange(len(sents_copy)))
            
        perturbed_text = f"{join_char}".join(sents_copy) 

    elif args.perturbation_strategy == "sentence_reverse": 
        sents_copy = sents_copy[::-1]
        perturbed_text = f"{join_char}".join(sents_copy) 

    elif args.perturbation_strategy == "sentence_last": 
        perturbed_text = f"{sents_copy[-1]}" 
       
    elif args.perturbation_strategy == "sentence_first": 
        perturbed_text = f"{sents_copy[0]}" 

    elif args.perturbation_strategy == "random_text": 
        selected_idx = random.sample(range(len(dataset)), k=1)[0]
        while selected_idx == idx: 
            selected_idx = random.sample(range(len(dataset)), k=1)[0]
        
        # give it same treatment as orig input 
        if args.from_ckpt: 
            perturbed_text = f"{join_char}".join(dataset[selected_idx]['dialog'][-len(sents_copy)+1:])
        else: 
            perturbed_text = f"{join_char}".join(dataset[selected_idx]['text'].split("\n")[-len(sents_copy)+1:]) 
        # perturbed_text = f"{join_char}".join(dataset[selected_idx]['text'].split("\n")) 
        # perturbed_text = "This is some random text that doesn't help with generating the actual target text. This is some random text that doesn't help with generating the actual target text. This is some random text that doesn't help with generating the actual target text." 
        
    elif args.perturbation_strategy == "word_drop": 
        
        words = f" ".join(sents).split()
        orig_len = len(words)
        while len(words) > (1-args.drop_pct)*orig_len: 
            words.pop(random.randrange(len(words)))
            
        import pdb; pdb.set_trace() 
        perturbed_text = f" ".join(words) 
        
    elif args.perturbation_strategy == "word_shuffle": 
        words = f" ".join(sents).split()
        random.shuffle(words)
        perturbed_text = f" ".join(words) 
        
    elif args.perturbation_strategy == "word_reverse": 
        words = f" ".join(sents).split()
        words = words[::-1] 
        perturbed_text = f" ".join(words) 

    else: 
        logger.error("Undefined strategy")
        exit(1)

    if idx == 0 : 
        logger.debug(input_text)
        logger.debug(perturbed_text)
        logger.debug(output_text)

    if args.from_ckpt: 
        input_text = input_text + "\n"
        perturbed_text = perturbed_text + "\n"
    else: 
        input_text = f"{join_char}".join(sents[:-1]) + f" <extra_id_{args.sentinel_token_idx}>"
        output_text = f"<extra_id_{args.sentinel_token_idx}> {output_text}"
        perturbed_text = perturbed_text + f" <extra_id_{args.sentinel_token_idx}>"

    orig_samples.append({"input": input_text, "label": output_text})
    perturbed_samples.append({"input": perturbed_text, "label": output_text})
    
    if len(orig_samples) == args.eval_size: 
        break 
    
partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_src_length=512 ,max_tgt_length=128)
orig_dataloader = DataLoader(orig_samples, batch_size=args.batch_size, shuffle=False, collate_fn=partial_collate_fn)
perturbed_dataloader = DataLoader(perturbed_samples, batch_size=args.batch_size, shuffle=False, collate_fn=partial_collate_fn)

print(len(orig_samples))
print(len(orig_dataloader))

orig_loss_sum = []
perturbed_loss_sum = []

with torch.no_grad(): 
    
    for orig_batch, p_batch in tqdm(zip(orig_dataloader, perturbed_dataloader), total=len(orig_dataloader)): 
                
        loss = model(
            input_ids=orig_batch["encoder_input"].to(device),
            attention_mask=orig_batch["attention_mask"].to(device),
            labels=orig_batch["decoder_output"].to(device)).loss
        perturbed_loss = model(
            input_ids=p_batch["encoder_input"].to(device),
            attention_mask=p_batch["attention_mask"].to(device),
            labels=p_batch["decoder_output"].to(device)).loss
        orig_loss_sum.append(loss.item())
        perturbed_loss_sum.append(perturbed_loss.item())
            
# import pdb; pdb.set_trace() 
orig_loss = np.mean(orig_loss_sum) 
perturbed_loss = np.mean(perturbed_loss_sum)
pct_diff = (perturbed_loss - orig_loss) / orig_loss * 100 

if args.from_ckpt: 
    model_name = Path(args.from_ckpt)
    model_name = f"{model_name.parent.name}/{model_name.name}"
else: 
    model_name = f"{MODEL}-{MODEL_SIZE}"
logger.info(f"{model_name} - {args.perturbation_strategy} | loss results: orig - {orig_loss:.3f} | perturbed - {perturbed_loss:.3f} | pct diff: {pct_diff:.3f}%")
