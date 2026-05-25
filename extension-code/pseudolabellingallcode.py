import random
from pathlib import Path
from PIL import Image
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets.utils import download_and_extract_archive

import copy
import torchvision.transforms as T

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

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
# each useful line in list.txt has the format: <image_name> <class_id> <species_id> <breed_id>
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

labels_dict = read_breed_labels(annotations_file)

print("Number of images with labels: ", len(labels_dict))
print("Example labels:", list(labels_dict.items())[:10]) # print first 10 labels
print("Number of unique classes:", len(set(labels_dict.values()))) # should be 37 for breed classification


class PetBreedDataset(Dataset): # full labelled dataset
    "given an index, return the corresponding image and label"
    def __init__(self, image_dir, labels_dict, transform=None):
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.transform = transform

        # only keep files that actually exist in the image folder
        self.filenames = [filename for filename in labels_dict.keys()
                          if (self.image_dir / filename).exists()]
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = self.image_dir / filename

        image = Image.open(img_path).convert("RGB")
        label = self.labels_dict[filename] # gets true label

        if self.transform:
            image = self.transform(image)

        return image, label
    
class UnlabelledPetDataset(Dataset): # unlabelled dataset for pseudo labelling
    """given an index, return the corresponding image and index to keep track of which image it is"""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, _ = self.base_dataset[real_idx] # ignore the label
        return image, real_idx # return the index so we can map back to the original dataset
    

class PseudoLabelDataset(Dataset): # takes stored pseudo-labels and behaves like a normal supervised dataset
    """dataset of pseudo labelled examples, given a list of (real_idx, pseudo_label) tuples"""
    def __init__(self, base_dataset, pseudo_labels):
        self.base_dataset = base_dataset # original fully labelled dataset
        self.pseudo_labels = pseudo_labels # list of pseudo labels created by the model

    def __len__(self): # tells pytorch how many pseudo labelled examples exist
        return len(self.pseudo_labels)
    
    def __getitem__(self, idx): # tells pytorch how to get the image and pseudo label for a given index in the pseudo labelled dataset
        real_idx, pseudo_label, confidence = self.pseudo_labels[idx]
        image, _ = self.base_dataset[real_idx] # ignore the original label
        return image, pseudo_label # dont need the confidence anymore, it was only used for filtering which examples to include in the pseudo labelled dataset
    
def generate_pseudo_labels(model, unlabelled_loader, device, threshold=0.9):
    model.eval()
    pseudo_labels = []

    with torch.no_grad():
        for images, real_indices in unlabelled_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            confidence, predictions = torch.max(probabilities, dim=1)

            for real_idx, confidence, prediction in zip(real_indices, confidence, predictions):
                if confidence.item() >= threshold: 
                    pseudo_labels.append((real_idx.item(), prediction.item(), confidence.item()))
   

    return pseudo_labels

def generate_class_balanced_pseudo_labels(
    model,
    unlabelled_loader,
    device,
    k_per_class=5,
    min_threshold=0.0
):
    model.eval()
    class_candidates = defaultdict(list)

    with torch.no_grad():
        for images, real_indices in unlabelled_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidence, predictions = torch.max(probabilities, dim=1)

            for real_idx, conf, pred in zip(real_indices, confidence, predictions):
                if conf.item() >= min_threshold:
                    class_candidates[pred.item()].append(
                        (real_idx.item(), pred.item(), conf.item())
                    )

    pseudo_labels = []

    for cls, candidates in class_candidates.items():
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        pseudo_labels.extend(candidates[:k_per_class])

    return pseudo_labels

class StrongPseudoLabelDataset(Dataset):
    """Pseudo-labelled dataset with strong augmentation."""
    def __init__(self, train_subset, pseudo_labels, strong_transform):
        self.train_subset = train_subset
        self.pseudo_labels = pseudo_labels
        self.strong_transform = strong_transform
        self.full_dataset = train_subset.dataset

    def __len__(self):
        return len(self.pseudo_labels)

    def __getitem__(self, idx):
        real_train_idx, pseudo_label, confidence = self.pseudo_labels[idx]

        full_dataset_idx = self.train_subset.indices[real_train_idx]
        filename = self.full_dataset.filenames[full_dataset_idx]
        img_path = self.full_dataset.image_dir / filename

        image = Image.open(img_path).convert("RGB")

        if self.strong_transform:
            image = self.strong_transform(image)

        return image, pseudo_label

def evaluate_pseudo_label_quality(base_dataset, pseudo_labels):
    """since we have the true labels we can measure how many labels were correct
    Tells whether the model is adding useful or noisy labels"""

    if len(pseudo_labels) == 0:
        return {
            "num_pseudo_labels": 0,
            "pseudo_label_accuracy": None,
            "average_confidence": None,
        }
    
    correct = 0
    total_confidence = 0.0

    for real_idx, pseudo_label, confidence in pseudo_labels:
        _, true_label = base_dataset[real_idx]
        if pseudo_label == true_label:
            correct += 1
        total_confidence += confidence

    pseudo_label_accuracy = correct / len(pseudo_labels)
    average_confidence = total_confidence / len(pseudo_labels)

    return {
        "num_pseudo_labels": len(pseudo_labels),
        "pseudo_label_accuracy": pseudo_label_accuracy,
        "average_confidence": average_confidence
    }

def create_semi_supervised_split(train_dataset, labelled_fraction):
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)

    labelled_size = int(labelled_fraction * len(indices))

    labelled_indices = indices[:labelled_size]
    unlabelled_indices = indices[labelled_size:]

    labelled_dataset = Subset(train_dataset, labelled_indices)
    unlabelled_dataset = UnlabelledPetDataset(train_dataset, unlabelled_indices)

    return labelled_dataset, unlabelled_dataset

    
def create_stratified_semi_supervised_split(train_dataset, labelled_fraction, seed=42):
    random.seed(seed)

    class_to_indices = defaultdict(list)

    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        class_to_indices[int(label)].append(idx)

    labelled_indices = []
    unlabelled_indices = []

    for label, indices in class_to_indices.items(): 
        random.shuffle(indices)

        #to decide how many examples from this class that should remain labelled
        n_labelled = max(1, int(labelled_fraction * len(indices))) # ensure at least 1 labelled example per class

        labelled_indices.extend(indices[:n_labelled])
        unlabelled_indices.extend(indices[n_labelled:])
    
    labelled_dataset = Subset(train_dataset, labelled_indices)
    unlabelled_dataset = UnlabelledPetDataset(train_dataset, unlabelled_indices)

    return labelled_dataset, unlabelled_dataset




# transforms
weights = ResNet18_Weights.DEFAULT 
transform = weights.transforms() # the images must be resized and normalized in the same way the pretrained model expects

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

weak_transform = ResNet18_Weights.DEFAULT.transforms()

strong_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std)
])


# create dataset and split into train and test sets

dataset = PetBreedDataset(
    image_dir=image_dir,
    labels_dict=labels_dict,
    transform=weak_transform
)

print("Total dataset size:", len(dataset))


train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

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



# fine tune l layers simultaneously


def get_resnet18_for_37_classes():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 37) 

    return model


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



def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_supervised_model(labelled_loader, val_loader, device, num_epochs, stage_schedule):
    """Gives a supervised baseline, and is a teacher model for generating the pseudo labels"""
    

    model = get_resnet18_for_37_classes()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if stage_schedule is None:
        stage_schedule = [(0, num_epochs)]

    global_epoch = 0
    history = []

    

    for stage, epochs_this_stage in stage_schedule:
        model = gradual_unfreeze_to_stage(model, stage)
        optimizer = make_optimizer(model)

        print(f"\nStage {stage}: trainable parameters = {count_trainable_params(model):,}")

        for epoch in range(epochs_this_stage):
            global_epoch += 1

            train_loss, train_acc = train_one_epoch(
                model, labelled_loader, criterion, optimizer, device
            )

            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device
            )

            print(
                f"Epoch {global_epoch} | Stage {stage} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            history.append({
                "epoch": global_epoch,
                "stage": stage,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "trainable_params": count_trainable_params(model),
            })
    return model, history

def train_with_pseudo_labels(model, combined_loader, val_loader, device, num_epochs, criterion, optimizer):
    """Continues self-training using both real and pseudo labels."""
    history = []

    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, combined_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        print(
            f"Pseudo Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best self-trained model from epoch {best_epoch} with val acc {best_val_acc:.4f}")

    return model, history



def run_pseudo_label_experiment(train_dataset, val_loader, test_loader, device, labelled_fraction, threshold=0.9, supervised_epochs=10, pseudo_epochs=10):
    """Runs pseudo-labelling self-training:
    trains a teacher model on the labelled subset,
    generates pseudo labels for the unlabelled subset,
    then continues training the same model on labelled + pseudo-labelled data.
    """

    print(f"Running experiment: labelled_fraction={labelled_fraction}, threshold={threshold}")

    labelled_dataset, unlabelled_dataset = create_stratified_semi_supervised_split(train_dataset, labelled_fraction, seed=42)
    print(f"Labelled examples: {len(labelled_dataset)}, Unlabelled examples: {len(unlabelled_dataset)}")

    stage_schedule = [
        (0, 2),
        (1, 2), # start adapting to high-level features
        (2, max(0, supervised_epochs - 4)), # main training phase
    ]

    print(f"\nTraining supervised teacher model for {supervised_epochs} epochs...")

    model, supervised_history = train_supervised_model(labelled_loader=DataLoader(labelled_dataset, batch_size=32, shuffle=True), val_loader=val_loader, device=device, num_epochs=supervised_epochs, stage_schedule=stage_schedule)

    criterion = nn.CrossEntropyLoss()

    baseline_test_loss, baseline_test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Baseline test loss: {baseline_test_loss:.4f}, test accuracy: {baseline_test_acc:.4f}")

    print(f"Generating pseudo labels with confidence threshold {threshold}...")

    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=32, shuffle=False)

    pseudo_labels = generate_pseudo_labels(model, unlabelled_loader, device, threshold)

    pseudo_quality = evaluate_pseudo_label_quality(train_dataset, pseudo_labels)

    print(f"Pseudo labels kept: {pseudo_quality['num_pseudo_labels']}")

    if pseudo_quality["pseudo_label_accuracy"] is not None:
        print(f"Pseudo label accuracy: {pseudo_quality['pseudo_label_accuracy']:.4f}")
        print(f"Average confidence of pseudo labels: {pseudo_quality['average_confidence']:.4f}")

    pseudo_dataset = PseudoLabelDataset(train_dataset, pseudo_labels)

    print(f"\nContinuing self-training on combined dataset for {pseudo_epochs} epochs...")

    model, pseudo_history = train_with_pseudo_labels(model, DataLoader(ConcatDataset([labelled_dataset, pseudo_dataset]), batch_size=32, shuffle=True), val_loader, device, pseudo_epochs, criterion, optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4))

    pseudo_test_loss, pseudo_test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Final test loss after pseudo labelling: {pseudo_test_loss:.4f}, test accuracy: {pseudo_test_acc:.4f}")

    results = {
        "labelled_fraction": labelled_fraction,
        "threshold": threshold,
        "num_labelled": len(labelled_dataset),
        "num_unlabelled": len(unlabelled_dataset),
        "num_pseudo_labels": pseudo_quality["num_pseudo_labels"],
        "pseudo_label_accuracy": pseudo_quality["pseudo_label_accuracy"],
        "average_pseudo_confidence": pseudo_quality["average_confidence"],
        "baseline_test_loss": baseline_test_loss,
        "baseline_test_acc": baseline_test_acc,
        "pseudo_test_loss": pseudo_test_loss,
        "pseudo_test_acc": pseudo_test_acc,
    }
    
    return results

def run_strong_aug_pseudo_label_experiment(
    train_dataset,
    val_loader,
    test_loader,
    device,
    labelled_fraction,
    threshold=0.9,
    supervised_epochs=10,
    pseudo_epochs=10
):
    labelled_dataset, unlabelled_dataset = create_stratified_semi_supervised_split(
        train_dataset, labelled_fraction, seed=42
    )

    stage_schedule = [
        (0, 2),
        (1, 2),
        (2, max(0, supervised_epochs - 4)),
    ]

    model, supervised_history = train_supervised_model(
        labelled_loader=DataLoader(labelled_dataset, batch_size=32, shuffle=True),
        val_loader=val_loader,
        device=device,
        num_epochs=supervised_epochs,
        stage_schedule=stage_schedule
    )

    criterion = nn.CrossEntropyLoss()

    baseline_test_loss, baseline_test_acc = evaluate(model, test_loader, criterion, device)

    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=32, shuffle=False)

    pseudo_labels = generate_pseudo_labels(
        model,
        unlabelled_loader,
        device,
        threshold
    )

    pseudo_quality = evaluate_pseudo_label_quality(train_dataset, pseudo_labels)

    pseudo_dataset = StrongPseudoLabelDataset(
        train_subset=train_dataset,
        pseudo_labels=pseudo_labels,
        strong_transform=strong_transform
    )

    combined_loader = DataLoader(
        ConcatDataset([labelled_dataset, pseudo_dataset]),
        batch_size=32,
        shuffle=True
    )

    model, pseudo_history = train_with_pseudo_labels(
        model,
        combined_loader,
        val_loader,
        device,
        pseudo_epochs,
        criterion,
        optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    )

    pseudo_test_loss, pseudo_test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "labelled_fraction": labelled_fraction,
        "threshold": threshold,
        "num_labelled": len(labelled_dataset),
        "num_unlabelled": len(unlabelled_dataset),
        "num_pseudo_labels": pseudo_quality["num_pseudo_labels"],
        "pseudo_label_accuracy": pseudo_quality["pseudo_label_accuracy"],
        "average_pseudo_confidence": pseudo_quality["average_confidence"],
        "baseline_test_acc": baseline_test_acc,
        "pseudo_test_acc": pseudo_test_acc,
    }

def run_class_balanced_pseudo_label_experiment(
    train_dataset,
    val_loader,
    test_loader,
    device,
    labelled_fraction=0.01,
    k_per_class=5,
    min_threshold=0.0,
    supervised_epochs=10,
    pseudo_epochs=10
):
    labelled_dataset, unlabelled_dataset = create_stratified_semi_supervised_split(
        train_dataset, labelled_fraction, seed=42
    )

    stage_schedule = [
        (0, 2),
        (1, 2),
        (2, max(0, supervised_epochs - 4)),
    ]

    model, supervised_history = train_supervised_model(
        labelled_loader=DataLoader(labelled_dataset, batch_size=32, shuffle=True),
        val_loader=val_loader,
        device=device,
        num_epochs=supervised_epochs,
        stage_schedule=stage_schedule
    )

    criterion = nn.CrossEntropyLoss()
    baseline_test_loss, baseline_test_acc = evaluate(model, test_loader, criterion, device)

    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=32, shuffle=False)

    pseudo_labels = generate_class_balanced_pseudo_labels(
        model,
        unlabelled_loader,
        device,
        k_per_class=k_per_class,
        min_threshold=min_threshold
    )

    pseudo_quality = evaluate_pseudo_label_quality(train_dataset, pseudo_labels)

    pseudo_dataset = PseudoLabelDataset(train_dataset, pseudo_labels)

    combined_loader = DataLoader(
        ConcatDataset([labelled_dataset, pseudo_dataset]),
        batch_size=32,
        shuffle=True
    )

    model, pseudo_history = train_with_pseudo_labels(
        model,
        combined_loader,
        val_loader,
        device,
        pseudo_epochs,
        criterion,
        optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    )

    pseudo_test_loss, pseudo_test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "k_per_class": k_per_class,
        "min_threshold": min_threshold,
        "num_pseudo_labels": pseudo_quality["num_pseudo_labels"],
        "pseudo_label_accuracy": pseudo_quality["pseudo_label_accuracy"],
        "baseline_test_acc": baseline_test_acc,
        "pseudo_test_acc": pseudo_test_acc,
    }


experiment_results = []

labelled_fractions = [0.5, 0.1, 0.01]
thresholds = [0.8, 0.9, 0.95]

for labelled_fraction in labelled_fractions:
    for threshold in thresholds:
        result = run_pseudo_label_experiment(
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            labelled_fraction=labelled_fraction,
            threshold=threshold,
            supervised_epochs=10,
            pseudo_epochs=10
        )

        experiment_results.append(result)


results_df = pd.DataFrame(experiment_results)

print(results_df[[
    "labelled_fraction",
    "threshold",
    "num_labelled",
    "num_unlabelled",
    "num_pseudo_labels",
    "pseudo_label_accuracy",
    "average_pseudo_confidence",
    "baseline_test_acc",
    "pseudo_test_acc"
]])

results_df.to_csv("pseudo_label_experiment_results.csv", index=False)


for threshold in thresholds:
    subset = results_df[results_df["threshold"] == threshold]

    plt.plot(
        subset["labelled_fraction"],
        subset["baseline_test_acc"],
        marker="o",
        label=f"Baseline, threshold {threshold}"
    )

    plt.plot(
        subset["labelled_fraction"],
        subset["pseudo_test_acc"],
        marker="x",
        label=f"Pseudo-label, threshold {threshold}"
    )

plt.xlabel("Labelled fraction")
plt.ylabel("Test accuracy")
plt.title("Supervised baseline vs pseudo-labelling")
plt.legend()
plt.grid(True)
plt.show()

# strong augmentation ablation
strong_aug_results = []

for labelled_fraction in [0.5, 0.1]:
    result = run_strong_aug_pseudo_label_experiment(
        train_dataset=train_dataset,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        labelled_fraction=labelled_fraction,
        threshold=0.9,
        supervised_epochs=10,
        pseudo_epochs=10
    )
    strong_aug_results.append(result)

strong_aug_df = pd.DataFrame(strong_aug_results)
strong_aug_df.to_csv("strong_aug_ablation_results.csv", index=False)
print(strong_aug_df)

# class balanced pseudo labelling
class_balanced_results = []

settings = [
    {"k_per_class": 1, "min_threshold": 0.0},
    {"k_per_class": 3, "min_threshold": 0.0},
    {"k_per_class": 5, "min_threshold": 0.0},
    {"k_per_class": 5, "min_threshold": 0.4},
]

for setting in settings:
    result = run_class_balanced_pseudo_label_experiment(
        train_dataset=train_dataset,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        labelled_fraction=0.01,
        k_per_class=setting["k_per_class"],
        min_threshold=setting["min_threshold"],
        supervised_epochs=10,
        pseudo_epochs=10
    )
    class_balanced_results.append(result)

class_balanced_df = pd.DataFrame(class_balanced_results)
class_balanced_df.to_csv("class_balanced_pseudo_label_results.csv", index=False)
print(class_balanced_df)