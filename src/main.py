import os
import argparse
import torch
from torch.utils.data import Dataset
import numpy as np
import copy
import pandas as pd
from model import PILP
from steps import train, test

class TrajDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.file = file
        #datas = pd.read_csv(file)#.drop_duplicates(subset=['full_formula'])
        self.features = np.array(pd.read_csv(file).iloc[:,11:].fillna(0).values, dtype=np.float)
        self.a = np.array(pd.read_csv(file).iloc[:,4].fillna(0).values, dtype=np.float)
        self.b = np.array(pd.read_csv(file).iloc[:,5].fillna(0).values, dtype=np.float)
        self.c = np.array(pd.read_csv(file).iloc[:,6].fillna(0).values, dtype=np.float)
        self.alpha = np.array(pd.read_csv(file).iloc[:,7].fillna(0).values, dtype=np.float)/180.0 * 3.1415926
        self.beta = np.array(pd.read_csv(file).iloc[:,8].fillna(0).values, dtype=np.float)/180.0 * 3.1415926
        self.gamma = np.array(pd.read_csv(file).iloc[:,9].fillna(0).values, dtype=np.float)/180.0 * 3.1415926
        self.crystals = np.array(pd.read_csv(file).iloc[:,1].fillna(0).values, dtype=np.float)
        #print(type(self.features), self.features.shape)
        #print(type(self.features.shape[0]), self.features.shape[0])
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        ground_truth = [self.a[index], self.b[index], self.c[index], self.alpha[index], self.beta[index], self.gamma[index]]
        all_truth = np.hstack((self.a[index], self.b[index], self.c[index], self.alpha[index], self.beta[index], self.gamma[index], self.crystals[index]))
        return self.features[index], ground_truth, self.crystals[index], all_truth


def main(args):
    # dataset
    #dataset_all = pd.read_csv(args.dataset_dir)
    train_dataset = TrajDataset(args.train_dataset_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataset = TrajDataset(args.test_dataset_dir)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # model
    model = PILP(args.traj_dim).double()
    #model.load_state_dict(torch.load("./ckpt/111/model-best-reg-14.pt"))
    model = model.cuda(0)
    #model.load_state_dict(torch.load("./ckpt/phase/model-499.pt"))
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    best_r2, best_cls = 0, 0
    for epoch in range(args.num_epochs):
        print("epoch ", epoch, ": \n")
        train_mse = train(model, train_loader, optimizer, epoch)
        test_r2, test_cls = test(model, test_loader)
        if test_r2 > best_r2:
            best_r2 = test_r2
            torch.save(model.state_dict(), "./ckpt/ablation/model-best-reg-14.pt")
            print("Best REG Model Saved")
        if test_cls > best_cls:
            best_cls = test_cls
            torch.save(model.state_dict(), "./ckpt/ablation/model-best-cls-14.pt")
            print("Best CLS Model Saved")
        if (epoch+1)%args.checkpoint_epochs ==  0:
            torch.save(model.state_dict(), "./ckpt/ablation/model-"+str(epoch)+".pt")
            print("Model Saved")

    '''
    # save your improved network
    if gpu == 0:
        torch.save(resnet.state_dict(), "./model-final.pt")

    cleanup()
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dim", default=249, type=int, help="trajectory dimension")
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", default=500, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--checkpoint_epochs",
        default=50,
        type=int,
        help="Number of epochs between checkpoints/summaries.",
    )
    parser.add_argument(
        "--train_dataset_dir",
        default="/home/liuke/liuke/prj/lattice/PILP_vae/data/train_14_phase.csv",
        type=str,
        help="Directory where dataset is stored.",
    )
    parser.add_argument(
        "--test_dataset_dir",
        default="/home/liuke/liuke/prj/lattice/PILP_vae/data/test_14_phase.csv",
        type=str,
        help="Directory where dataset is stored.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers (caution with nodes!)",
    )
    args = parser.parse_args()

    main(args)