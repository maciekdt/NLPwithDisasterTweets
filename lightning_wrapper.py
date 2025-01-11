from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl
from torch.nn import MSELoss
import torch.optim as optim

class LightningWrapper(pl.LightningModule):
    def __init__(self, model, lr=1e-3, lr_pre=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_pre = lr_pre
        self.criterion = MSELoss()

        self.train_accuracy = Accuracy(num_classes=2, average='macro', task="binary")
        self.val_accuracy = Accuracy(num_classes=2, average='macro', task="binary")
        self.test_accuracy = Accuracy(num_classes=2, average='macro', task="binary")

        self.train_f1 = F1Score(num_classes=2, average='macro', task="binary")
        self.val_f1 = F1Score(num_classes=2, average='macro', task="binary")
        self.test_f1 = F1Score(num_classes=2, average='macro', task="binary")

    def forward(self, input_ids, attention_mask):
        return self.model(tokens=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"]).squeeze(-1)
        target = batch["target"].float()
        loss = self.criterion(logits, target)

        acc = self.train_accuracy(logits, target.int())
        f1 = self.train_f1(logits, target.int())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"]).squeeze(-1)
        target = batch["target"].float()
        loss = self.criterion(logits, target)

        acc = self.val_accuracy(logits, target.int())
        f1 = self.val_f1(logits, target.int())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"]).squeeze(-1)
        target = batch["target"].float()
        loss = self.criterion(logits, target)

        acc = self.test_accuracy(logits, target.int())
        f1 = self.test_f1(logits, target.int())

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        bert_params = self.model.pretrained_model.parameters()
        custom_params = [
            param for name, param in self.model.named_parameters() 
            if "pretrained_model" not in name
        ]

        optimizer = optim.Adam([
            {"params": bert_params, "lr": self.lr_pre},
            {"params": custom_params, "lr": self.lr},
        ])
        return optimizer
