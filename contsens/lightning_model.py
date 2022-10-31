import pytorch_lightning as pl 
from torch import optim
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5DialogueModel(pl.LightningModule): 
    def __init__(self, model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    def training_step(self, batch, batch_idx): 
        
        inputs= batch['inputs']
        labels = batch["labels"]
        pred = self.model(inputs) 
        
        self.log("train_loss", loss)
        return loss 
    
    def configure_optimizers(self): 
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer 