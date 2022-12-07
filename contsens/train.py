from .lightning_model import T5DialogueModel, T5Tokenizer
from .dataloader import DailyDialogDataLoader
import torch 
import pytorch_lightning as pl 
from argparse import ArgumentParser

def train(args): 
    
    device = "gpu" if torch.cuda.is_available() else "cpu"
    dict_args = vars(args)
    model = T5DialogueModel(**dict_args)
    datamodule = DailyDialogDataLoader(model.tokenizer, **dict_args)
    
    trainer = pl.Trainer(        
        # args, 
        accelerator=device, 
        max_epochs=args.max_epoch,
        deterministic=True, 
        devices=-1,
        callbacks=model.callbacks
    )
        
    trainer.fit(
        model=model, 
        datamodule=datamodule, 
    )
    
    best_model_ckpt = model.checkpoint_callback.best_model_path
    best_model = T5DialogueModel.load_from_checkpoint(best_model_ckpt)
    test_results = trainer.test(
        model=best_model, datamodule=datamodule
    )

def main(): 
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser = T5DialogueModel.add_model_specific_args(parser)
    parser = DailyDialogDataLoader.add_data_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed, workers=True)
    train(args) 
    
if __name__ =="__main__": 
    main() 