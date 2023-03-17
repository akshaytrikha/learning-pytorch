import torch
from torch import nn
from torchvision import transforms
from torchmetrics import Accuracy
import lightning.pytorch as pl
from pathlib import Path
from typing import Tuple

import data

TRAIN_DIR = Path("./data/train")
DEV_DIR = Path("./data/test")

device = device = "mps" if torch.backends.mps.is_available() else "cpu"

# hyperparameterse
NUM_BATCHES = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_UNITS = 10


class LightningPizzaSteakSushiClassifier(pl.LightningModule):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

    def configure_optimizers(self):
        return torch.optim.SGD(params=self.parameters(), lr=LEARNING_RATE)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch

        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        return loss


# ------------------ Data ------------------
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# fetch data if doesn't exist
data.fetch_data()
train_dataloader, dev_dataloader, class_names = data.create_dataloaders(
    train_dir=TRAIN_DIR, dev_dir=DEV_DIR, batch_size=NUM_BATCHES, transform=transform
)
NUM_CLASSES = len(class_names)


# ------------------ Model ------------------
model = LightningPizzaSteakSushiClassifier(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=NUM_CLASSES
)

# ------------------ Training ------------------
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    accelerator=device,
    check_val_every_n_epoch=10,
)
trainer.fit(model, train_dataloader)
