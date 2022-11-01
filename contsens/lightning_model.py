import pytorch_lightning as pl 
from torch import optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

class T5DialogueModel(pl.LightningModule): 
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("T5DialogueModel")
        parser.add_argument("--model_name", type=str, default="t5-base")
        return parent_parser
    
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = self.hparams.model_name 
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.set_callbacks()
        
        
    def training_step(self, batch, batch_idx): 
            
        loss = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        ).loss
        
        self.log("loss", loss)
        return loss 
    
    def validation_step(self, batch, batch_idx): 
        loss = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        ).loss
        
        self.log("val_loss", loss)
        return loss 
             
    def test_step(self, batch, batch_idx): 
        loss = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        ).loss
        
        self.log("test_loss", loss)
        return loss 
    
    def configure_optimizers(self): 
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer 
    
    def set_callbacks(self):
        """set callbacks to use for the trainer"""

        checkpoint_callback = ModelCheckpoint(
            dirpath="results",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            filename=f"{self.model_name}_"+'{epoch}-{step}-{val_loss:.2f}', 
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

        self.checkpoint_callback = checkpoint_callback
        self.callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]