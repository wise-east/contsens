from datasets import load_dataset
from typing import List, Dict, Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl 
from functools import partial

def samplify_dialog(dialog: List[str], dial_id: str, min_turns: int): 
    samples =[] 
    for idx in range(len(dialog)-1): 
        sample = {
            "input": "\n".join(dialog[:idx+1]),
            "label": dialog[idx+1],
            "dial_id": dial_id, 
            "turn_idx": idx  
        }
        
        if len(dialog[:idx+1]) > min_turns: 
            samples.append(sample)
        
    return samples 

def collate_fn(
    data: Union[List[Dict[str, str]], List[Dict[str, List[str]]]],
    tokenizer: AutoTokenizer,
    max_src_length: int,
    max_tgt_length: int,
):
    

    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]


    input_batch = tokenizer(
        batch_data["input"],
        padding="longest",
        max_length=max_src_length,
        return_tensors="pt",
        verbose=False,
        truncation=True,
    )
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(
        batch_data["label"],
        padding="longest",
        max_length=max_tgt_length,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
    )
    # replace the padding id to -100 for cross-entropy
    # output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    output_batch['input_ids'][
        output_batch['input_ids'] == tokenizer.pad_token_id
    ] = -100
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data

class DialogDataLoader(pl.LightningDataModule):
    def __init__(self):
        super().__init__() 

class DailyDialogDataLoader(DialogDataLoader): 
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DailyDialogDataLoader")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_src_length", type=int, default=512)
        parser.add_argument("--max_tgt_length", type=int, default=128)
        parser.add_argument("--num_samples", type=int, default=-1, help="-1 to use all samples for train/valid/test. Provide other integer for reduced size, mainly for debugging.")
        parser.add_argument("--min_turns", type=int, default=0, help="0 to use all turns for dialogue modeling. otherwise, only model samples with len(dialogue_turns) > min_turns")
        return parent_parser
    
    def __init__(self, tokenizer, **kwargs): 
        super().__init__()
        self.save_hyperparameters()
        
    def get_collate_fn(self): 
        _collate_fn = partial(
            collate_fn,
            tokenizer=self.hparams.tokenizer,
            max_src_length=self.hparams.max_src_length,
            max_tgt_length=self.hparams.max_tgt_length,
        )
        
        return _collate_fn
        
    def prepare_data(self): 
        dataset = load_dataset("daily_dialog")
        self.dataloaders = {}
        
        for split in dataset.keys(): 
            all_samples = [] 
            for idx, dialog in enumerate(dataset[split]): 
                dial_id = f"{split}-{idx}"
                dialog = [str(turn) for turn in dialog['dialog']]
                all_samples += samplify_dialog(dialog=dialog, dial_id=dial_id, min_turns=self.hparams.min_turns)
                                                
            self.dataloaders[split] = DataLoader(
                all_samples[:self.hparams.num_samples],
                batch_size=self.hparams.batch_size if split =="train" else self.hparams.batch_size*4,
                shuffle=True if split == "train" else False,
                collate_fn=self.get_collate_fn(),
                num_workers=1,
            )    
            
        
    def train_dataloader(self): 
        return self.dataloaders["train"]
    
    def val_dataloader(self): 
        return self.dataloaders["validation"]
    
    def test_dataloader(self): 
        return self.dataloaders["test"]
    
