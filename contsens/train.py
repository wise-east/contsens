from .lightning_model import T5DialogueModel
# from dataset import
import pytorch_lighting as pl 

trainer = pl.Trainer()
trainer.fit(model=model, train_dataloaders=train_loader)

