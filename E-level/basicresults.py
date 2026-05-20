import os
import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms

from sklearn.model_selection import train_test_split




# downloading the dataset, images + annotations
root = Path("./data")
root.mkdir(exist_ok=True)

images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

if not (root / "images").exists():
    download_and_extract_archive(images_url, download_root=str(root))

if not (root / "annotations").exists():
    download_and_extract_archive(annotations_url, download_root=str(root))

image_dir = root / "images"
annotations_file = root / "annotations" / "list.txt"

# each useful line in list.txt is on the format: <image_name> <class_id> <species_id> <breed_id>
#for binary classification, we only need column 2

print("Image folder:", image_dir)
print("Annotations file:", annotations_file)


# read the annotations file 

def read_breed_labels(annotations_file):
    labels = {}
    with open(annotations_file, "r") as f:
        for line in f:
            # skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()

            image_name = parts[0]


            filename = image_name + ".jpg"
            labels[filename] = int(parts[1]) - 1 # convert to 0-based index

    return labels

def read_labels_and_species(annotations_file):
    labels = {}
    species = {}

    with open(annotations_file, "r") as f:
        for line in f:
            # skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()

            image_name = parts[0]
            class_id = int(parts[1]) - 1 # convert to 0-based index
            species_id = int(parts[2]) # 1 for cat, 2 for dog

            filename = image_name + ".jpg"

            labels[filename] = class_id
            species[filename] = species_id

    return labels, species


labels_dict, species_dict = read_labels_and_species(annotations_file)


print("Number of images with labels: ", len(labels_dict))
print("Example labels:", list(labels_dict.items())[:10]) # print first 10 labels
print("Number of unique classes:", len(set(labels_dict.values()))) # should be 37 for breed classification


class PetBreedDataset(Dataset): # full labelled dataset
    "given an index, return the corresponding image and label"
    def __init__(self, image_dir, labels_dict, filenames=None, transform=None):
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.transform = transform
        self.filenames = filenames

        if self.filenames is None: 
        # only keep files that actually exist in the image folder
            self.filenames = [filename for filename in labels_dict.keys()
                          if (self.image_dir / filename).exists()]
        else:
            self.filenames = filenames
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = self.image_dir / filename

        image = Image.open(img_path).convert("RGB")
        label = self.labels_dict[filename]

        if self.transform:
            image = self.transform(image)

        return image, label
    


# transforms
weights = ResNet18_Weights.DEFAULT 
base_transform = weights.transforms() # the images must be resized and normalized in the same way the pretrained model expects

train_augmented_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
])

# create dataset and split into train and test sets

dataset = PetBreedDataset(
    image_dir=image_dir,
    labels_dict=labels_dict,
    transform=base_transform
)

print("Total dataset size:", len(dataset))

all_filenames = list(labels_dict.keys())
all_filenames = [filename for filename in all_filenames if (image_dir / filename).exists()]

all_labels = [labels_dict[filename] for filename in all_filenames]

train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_filenames, all_labels, test_size=0.3, stratify=all_labels, random_state=42
)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_dataset = PetBreedDataset(
    image_dir=image_dir,
    labels_dict=labels_dict,
    filenames=train_files,
    transform=train_augmented_transform
)   

val_dataset = PetBreedDataset(
    image_dir=image_dir,
    labels_dict=labels_dict,
    filenames=val_files,
    transform=base_transform
)

test_dataset = PetBreedDataset(
    image_dir=image_dir,
    labels_dict=labels_dict,
    filenames=test_files,
    transform=base_transform
)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

print("Train set size:", len(train_dataset))
print("Validation set size:", len(val_dataset))
print("Test set size:", len(test_dataset))

# check one batch of data
images, labels = next(iter(train_loader))
print("Batch of images shape:", images.shape) # should be [batch_size, 3, 224, 224]
print("Batch of labels shape:", labels.shape) # should be [batch_size]
print("Batch of labels:", labels[:10]) # print first 10 labels

# load pretrained ResNet-18 and modify the final layer for 37-class classification

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model = resnet18(weights=ResNet18_Weights.DEFAULT)

def set_trainable_layers(model, l):
    # freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    layer_groups = [model.layer4, model.layer3, model.layer2, model.layer1]

    for layer in layer_groups[:l]:
        for param in layer.parameters():
            param.requires_grad = True

    return model 


# replace the final fully connected layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 37) 

model = model.to(device)

print(model.fc) # should show the new final layer with 2 output features

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
model = set_trainable_layers(model, l=0) # only train the final layer for linear probing, will unfreeze more layers later for fine-tuning experiments
optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # only optimize the final layer parameters

# Training Function

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # reset gradients from previous step

        outputs = model(images) # forward pass
        loss = criterion(outputs, labels)
        loss.backward() # compute gradients, backward pass
        optimizer.step() # update weights

        running_loss += loss.item() * images.size(0) # track loss
        _, predicted = torch.max(outputs.data, 1) # track accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    """similar to train_one_epoch but without backpropagation and weight updates, only measures performance"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# linear probing: only train the final layer for a few epochs, then evaluate on the test set

num_epochs = 5
model = set_trainable_layers(model, l=0) # only train the final layer for linear probing, will unfreeze more layers later for fine-tuning experiments
"""

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device
    )

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
          f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
    
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

"""

# fine tune l layers simultaneously


def get_resnet18_for_37_classes(): # every experiment should start from the same pretrained resnet 18 rather than continuing from a previously trained model 
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 37) 

    return model


def set_trainable_layers(model, l):
    # first freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # always train the new classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    # unfreeze the last l ResNet layer groups
    layer_groups = [model.layer4, model.layer3, model.layer2, model.layer1]
    layers_to_unfreeze = layer_groups[:l]  
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True  
    return model

# run experiments
l_values = []
train_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []

num_epochs = 5

for l in range(1, 5):
    print(f"\n Fine-tuning last {l} ResNet layer groups + final classifier")

    model = get_resnet18_for_37_classes()
    model = set_trainable_layers(model, l)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
          lr=1e-4,
          weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss for l = {l}: {test_loss:.4f}")
    print(f"Test Accuracy for l = {l}: {test_acc:.4f}")
    l_values.append(l)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)


# gradual unfreezing

def freeze_all_but_fc(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model    

def gradual_unfreeze_to_stage(model, stage):
    # stage 0: only final layer
    # stage 1: unfreeze layer4 + final layer
    # stage 2: unfreeze layer3, layer4 + final layer
    # stage 3: unfreeze layer2, layer3, layer4 + final layer
    # stage 4: unfreeze all layers

    model = freeze_all_but_fc(model)

    layer_groups = [model.layer4, model.layer3, model.layer2, model.layer1]

    for layer in layer_groups[:stage]:
        for param in layer.parameters():
            param.requires_grad = True

    return model

def make_optimizer(model):
    return optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": 1e-3},
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer2.parameters(), "lr": 1e-5},
            {"params": model.layer1.parameters(), "lr": 1e-5}, 
        ],
        weight_decay=1e-4
    )

import time

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




schedule = [
    (0, 2),  # fc only for 2 epochs
    (1, 2),  # layer4 + fc
    (2, 2),  # layer3 + layer4 + fc
    (3, 2),  # layer2 + layer3 + layer4 + fc
    (4, 2),  # all ResNet blocks + fc
]

start_time = time.time()
"""


for stage, epochs_this_stage in schedule:
    #model = gradual_unfreeze_to_stage(model, stage)
    model = get_resnet18_for_37_classes().to(device) # start from the same pretrained model for each stage rather than continuing from the previous stage, to better isolate the effect of unfreezing more layers
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model)

    for epoch in range(epochs_this_stage):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Stage {stage}, Epoch {epoch+1}/{epochs_this_stage} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
elapsed_time = time.time() - start_time
print(test_acc, elapsed_time)
"""


def run_finetuning_experiment(train_dataset, val_dataset, test_dataset, num_epochs=5, weight_decay=1e-4, lr=1e-4, unfreeze_stage=1):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

    model = get_resnet18_for_37_classes()
    model = gradual_unfreeze_to_stage(model, unfreeze_stage)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,  
        weight_decay=weight_decay
    )

    history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return model, history, test_loss, test_acc


elapsed_time = time.time() - start_time



model = get_resnet18_for_37_classes()
model = model.to(device)
model = set_trainable_layers(model, l=0) # only train the final layer for linear probing, will unfreeze more layers later for fine-tuning experiments




