from model import *
from training import *
from custom_utils import *

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 8

dataset_dir = "/home/abdulla/Documents/Untitled Folder/data/Conglomerate Concrete Crack Detection/Train"
checkpoint_dir = "/home/abdulla/Documents/Untitled Folder/data/checkpoints"
result_path="/home/abdulla/Documents/Untitled Folder/data/results.csv"

dataset = CrackDataset(dataset_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

# Create model
model = AttentionUNet()
print_model_summary(model, dataset[0][0].shape)
model = model.to(device)

n_epochs = 1

train_model(model, train_loader, val_loader, device, num_epochs=n_epochs, checkpoint_dir=checkpoint_dir, result_path=result_path)