import random
from typing import ForwardRef
import torch
from torch import nn, softmax
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
from utils import physic_informed


class MLP_REG(nn.Module):
    def __init__(
        self,
        ft_dim,
        p=0.2
    ):
        super(MLP_REG, self).__init__()
        self.l1 = nn.Linear(ft_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 1)
        self.p = p 
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.dropout(F.relu(self.l3(x)), p=self.p)
        x = F.dropout(F.relu(self.l4(x)), p=self.p)
        x = F.dropout(F.relu(self.l5(x)), p=self.p)
        return self.l6(x)


class MLP_CLA(nn.Module):
    def __init__(
        self,
        ft_dim,
        p=0.2
    ):
        super(MLP_CLA, self).__init__()
        self.l1 = nn.Linear(ft_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 14)
        self.p = p

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.dropout(F.relu(self.l3(x)), p=self.p)
        x = F.dropout(F.relu(self.l4(x)), p=self.p)
        x = F.dropout(F.relu(self.l5(x)), p=self.p)
        return F.log_softmax(self.l6(x), dim=1)
        #return self.softmax(self.net(x))


class PILP(nn.Module):
    def __init__(self, ft_dim, training=True, latent_size=64, p=0.1):
        super(PILP, self).__init__()
        # encoder
        self.fc1 = nn.Linear(ft_dim + 7, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)
        # decode
        self.a_pre = MLP_REG(ft_dim+latent_size, p)
        self.b_pre = MLP_REG(ft_dim+latent_size, p)
        self.c_pre = MLP_REG(ft_dim+latent_size, p)
        self.alpha_pre = MLP_REG(ft_dim+latent_size, p)
        self.beta_pre = MLP_REG(ft_dim+latent_size, p)
        self.gamma_pre = MLP_REG(ft_dim+latent_size, p)
        self.crystal_cla = MLP_CLA(ft_dim+latent_size, p)
        # functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.training = training

    def encode(self, gt, x): #(p(z|ft,gt))
        '''
        gt = ground truth, x = feature
        '''
        inputs = torch.cat([gt, x], 1)
        h1 = self.sigmoid(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
        else:
            return mu

    def decode(self, z, x, crystal_gt=None):
        '''
        x = feature, z = latent
        '''
        inputs = torch.cat([z, x], 1)
        a,b,c,alpha,beta,gamma\
         =self.a_pre(inputs),self.b_pre(inputs),self.c_pre(inputs),\
             self.alpha_pre(inputs),self.beta_pre(inputs),self.gamma_pre(inputs)
        crystal = self.crystal_cla(inputs)
        #crytal_pre = torch.multinomial(crystal, 1).view(-1)
        #crytal_pre = torch.argmax(crystal, 1).view(-1)
        if crystal_gt is None:
            #print("None")
            crystal_gt = torch.multinomial(crystal, 1).view(-1).view(-1,1)
        #return crystal, physic_informed([a, b, c, alpha, beta, gamma], crystal_gt)
        #print(crystal.shape, crytal_pre.shape)
        #loss_extra = self.extra(a,b,c,alpha,beta,gamma,crytal_pre)
        return crystal, [a, b, c, alpha, beta, gamma]

    def forward(self, gt, x, crystal_gt=None):
        mu, logvar = self.encode(gt, x)
        z = self.reparametrize(mu, logvar)
        cry, pi = self.decode(z, x, crystal_gt)
        return cry, pi, mu, logvar