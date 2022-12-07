import pytorch_lightning as pl 
from torch import optim
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

class T5DialogueModel(pl.LightningModule): 
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("T5DialogueModel")
        parser.add_argument("--model_name", type=str, default="google/t5-v1_1-small")
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--from_scratch", action="store_true")
        return parent_parser
    
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = self.hparams.model_name 
        
        if hasattr(self.hparams, "from_scratch") and self.hparams.from_scratch: 
            config = AutoConfig.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration(config)
        else: 
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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer 
    
    def set_callbacks(self):
        """set callbacks to use for the trainer"""

        if hasattr(self.hparams, "from_scratch") and self.hparams.from_scratch: 
            model_name = f"{self.model_name}_scratch_"        
        else: 
            model_name = f"{self.model_name}_"
            
        checkpoint_callback = ModelCheckpoint(
            dirpath="results",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            filename=model_name+'{epoch}-{step}-{val_loss:.2f}', 
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

        self.checkpoint_callback = checkpoint_callback
        self.callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]