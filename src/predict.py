import os
import argparse
import torch
from torch.utils.data import Dataset
import numpy as np
import copy
import pandas as pd
from model import PILP
from steps import train, test, predict

class TrajDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.features = np.array(pd.read_csv(file).iloc[:,11:].fillna(0).values, dtype=np.float)
        self.a = np.array(pd.read_csv(file).iloc[:,4].fillna(0).values, dtype=np.float)
        self.b = np.array(pd.read_csv(file).iloc[:,5].fillna(0).values, dtype=np.float)
        self.c = np.array(pd.read_csv(file).iloc[:,6].fillna(0).values, dtype=np.float)
        self.alpha = np.array(pd.read_csv(file).iloc[:,7].fillna(0).values, dtype=np.float)/180.0 * 3.1415926
        self.beta = np.array(pd.read_csv(file).iloc[:,8].fillna(0).values, dtype=np.float)/180.0 * 3.1415926
        self.gamma = np.array(pd.read_csv(file).iloc[:,9].fillna(0).values, dtype=np.float)/180.0 * 3.1415926
        self.crystals = np.array(pd.read_csv(file).iloc[:,1].fillna(0).values, dtype=np.float)
        self.id = pd.read_csv(file).iloc[:,0].values
        self.formula = pd.read_csv(file).iloc[:,2].values
        self.full_formula = pd.read_csv(file).iloc[:,3].values
        #print(type(self.features.shape[0]), self.features.shape[0])
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        all_truth = np.hstack((self.a[index], self.b[index], self.c[index], self.alpha[index], self.beta[index], self.gamma[index], self.crystals[index]))
        ground_truth = [self.a[index], self.b[index], self.c[index], self.alpha[index], self.beta[index], self.gamma[index]]
        return self.features[index], ground_truth, self.crystals[index], all_truth, self.id[index], self.formula[index], self.full_formula[index]


def main(args):

    test_dataset = TrajDataset(args.test_dataset_dir)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # model
    model = PILP(args.traj_dim).double()
    #model.load_state_dict(torch.load("./ckpt/model-best-reg-14.pt"))
    model = model.cuda(0)
    model.load_state_dict(torch.load(args.checkpoint_dir))
    
    if predict(model, test_loader, args):
        print("Finished, result has been saved in ", args.output_dir)
    else:
        print("ERROR!!!!!!!!!")
    # optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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
        "--learning_rate", default=5e-4, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--output_dir", default="result/test_ablation.csv", type=str, help="file to store result."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/liuke/liuke/prj/lattice/PILP_vae/ckpt/ablation/model-499.pt",
        type=str,
        help="checkpoint path.",
    )
    parser.add_argument(
        "--test_dataset_dir",
        default="./data/test_14_phase.csv",
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