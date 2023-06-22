import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchmetrics
from pathlib import Path
import os
import pandas as pd
import data, model, engine, utils


MODEL_NAME = "EfficientNet_B0 23_03_11"
RANDOM_SEED = 100
NUM_WORKERS = os.cpu_count()

# hyperparameterse
NUM_BATCHES = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_DIR = Path("./data/train")
DEV_DIR = Path("./data/test")

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # Apple Silicon
    device = "mps"
else:
    device = "cpu"


# ------------------ Data ------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# fetch data if doesn't exist
data.fetch_data()
train_dataloader, dev_dataloader, class_names = data.create_dataloaders(
    train_dir=TRAIN_DIR, dev_dir=DEV_DIR, batch_size=NUM_BATCHES, transform=transform
)
NUM_CLASSES = len(class_names)

# ------------------ Model ------------------
# instantiate pretrained model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

torch.manual_seed = RANDOM_SEED

# modify classifier layer for number of classes
model.classifier = nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=NUM_CLASSES),
).to(device)

# ------------------ Training ------------------
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
