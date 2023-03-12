import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchmetrics
from pathlib import Path
import os
import data, model, engine, utils
import pandas as pd


MODEL_NAME = "TinyVGG Food Classification 23-03-11 #1"
RANDOM_SEED = 100
NUM_WORKERS = os.cpu_count()

# hyperparameterse
NUM_BATCHES = 32
NUM_EPOCHS = 100
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

TRAIN_DIR = Path("./data/train")
DEV_DIR = Path("./data/test")

device = device = "cuda" if torch.cuda.is_available() else "cpu"

# data augmentation
augment = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Create a transforms pipeline manually (required for torchvision < 0.13)
manual_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# fetch data if doesn't exist
data.fetch_data()
train_dataloader, dev_dataloader, class_names = data.create_dataloaders(
    train_dir=TRAIN_DIR, dev_dir=DEV_DIR, batch_size=NUM_BATCHES, transform=augment
)
NUM_CLASSES = len(class_names)

# instantiate model
torch.manual_seed(RANDOM_SEED)
model = model.PizzaSteakSushiClassifier(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=NUM_CLASSES
)

# define loss, optimizer, accuracy
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)


# train model
training_results = engine.train(
    model,
    train_dataloader,
    dev_dataloader,
    loss_fn,
    optimizer,
    accuracy_fn,
    NUM_EPOCHS,
    device,
)

# save model
utils.save_model(model, MODEL_NAME)

# save training results
pd.DataFrame(training_results).to_csv(
    Path(f"./models/{MODEL_NAME}/{MODEL_NAME}_training.csv"), index_label="epoch"
)
