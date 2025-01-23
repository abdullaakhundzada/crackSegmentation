from model import AttentionUNet
from train import train_model, CrackDataset
from argparse import ArgumentParser
from PIL import Image
import torch
import os
from torch.utils.data import random_split, DataLoader

parser = ArgumentParser("Set training arguments", add_help=True)

parser.add_argument("--dataset_path", type=str, 
                    help="Path to the dataset folder. must contain '/images' and '/masks' folders")

parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoint", 
                    help="Path to a directory that the checkpoint models will be saved to")

parser.add_argument("--final_model_path", type=str, default="data/crackSegmentationModel.pt", 
                    help="Full path to save the final model. Must end with .pt")

parser.add_argument("--logs_dir", type=str, default="data/logs", 
                    help="The directory to save the log data from the training")

parser.add_argument("--n_workers", type=int, default=16, 
                    help="The number of workers for the dataloader. Prevents the bottlenecks in the dataloading process")

parser.add_argument("--n_epochs", type=int, default=70, help="The number of epochs for training")

parser.add_argument("--batch_size", type=int, default=16, help="The batch size for training")

parser.add_argument("--image_size", type=str, default="448x448", 
                    help="The size of the input images. The width and the hight must be separated by 'x'. Default is '448x448'")

args = parser.parse_args()

im_shape = tuple( map( int, args.image_size.split("x") ) )

assert args.final_model_path.endswith(".pt"), "The final model path does not have the correct file extension. Must be .pt"

if not os.path.exists(args.checkpoint_dir): os.mkdir(args.checkpoint_dir)
if not os.path.exists(args.logs_dir)      : os.mkdir(args.logs_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CrackDataset(args.dataset_path, im_shape)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.n_workers)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

model = AttentionUNet()

train_model(model, train_loader, val_loader, device, num_epochs=args.n_epochs, checkpoint_dir=args.checkpoint_dir, result_path=args.logs_dir)




