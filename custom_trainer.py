import pandas as pd
from text_dataset import TextDataset
from torch.utils.data import DataLoader
from binary_bert_model import BinaryBertModel
from lightning_wrapper import LightningWrapper
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
import torch

class F1Tracker(Callback):
    def __init__(self):
        super().__init__()
        self.max_f1 = -float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        val_f1 = trainer.callback_metrics.get("val_f1")
        if val_f1 is not None:
            val_f1 = val_f1.item() if isinstance(val_f1, torch.Tensor) else val_f1
            self.max_f1 = max(self.max_f1, val_f1)


class CustomTrainer:
    def __init__(self, train_df="data/train.csv", val_path="data/val.csv"):
        self.train_df = pd.read_csv(train_df)
        self.val_df = pd.read_csv(val_path)
    
    def train(
        self,
        model_id = 'bert-base-uncased',
        lr = 1e-3,
        lr_pre = 1e-4,
        dropout_p = 0,
        batch_size = 32,
        seq_len = 64,
        max_epochs = 10,
        optim_mode = False,
    ):  
        
        train_data_set = TextDataset(self.train_df, max_len=seq_len, pretrained_model_id=model_id)
        val_data_set = TextDataset(self.val_df, max_len=seq_len, pretrained_model_id=model_id)
        
        train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BinaryBertModel(seq_length=seq_len, pretrained_bert_model=model_id, dropout_p=dropout_p).to(device)
        
        lightning_model = LightningWrapper(model, lr=lr, lr_pre=lr_pre)
        logger = TensorBoardLogger("lightning_logs", name="bert_model")
        
        early_stopping = EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=2,
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            filename="best-checkpoint",
            save_top_k=1,
            dirpath="checkpoints",
            verbose=True,
        )
        
        f1_tracker = F1Tracker()
    
        if optim_mode:
            trainer = Trainer(
                max_epochs=max_epochs,
                enable_progress_bar=True,
                enable_checkpointing=False,
                logger=False,
                callbacks=[
                    early_stopping,
                    f1_tracker,
                ],
            )
        
        else:
            trainer = Trainer(
                max_epochs=max_epochs,
                enable_progress_bar=True,
                logger=logger,
                callbacks=[
                    early_stopping, 
                    checkpoint_callback,
                ],
            )
            
        
        trainer.fit(
            lightning_model,
            train_dataloaders=train_data_loader,
            val_dataloaders=val_data_loader
        )
        
        return f1_tracker.max_f1