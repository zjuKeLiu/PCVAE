import torch
from sklearn.metrics import r2_score
import numpy as np
from torch import nn, softmax

PI = 3.1415926

def log_mse(inps, tars):
    criterion = torch.nn.MSELoss()
    res = list()
    for inp, tar in zip(inps, tars):
        #res.append(torch.sum(torch.log((inp.cuda(0)/tar.squeeze())**2)))
        res.append(criterion(inp.cuda(0), tar.squeeze()))
    return res

def r2_cal(inp, tar):
    R2_list = list()
    for i, t in zip(inp, tar):
        R2_list.append(r2_score(np.array(i).reshape(-1), np.array(t).reshape(-1)))
    return R2_list

def physic_informed(prediction, label_pre):
    [a, b, c, alpha, beta, gamma] = prediction
    a_en = torch.empty_like(a)
    b_en = torch.empty_like(b)
    c_en = torch.empty_like(c)
    alpha_en = torch.empty_like(alpha)
    beta_en = torch.empty_like(beta)
    gamma_en = torch.empty_like(gamma)
    i = 0
    for a1, b1, c1, alpha1, beta1, gamma1, label_pre1 in zip(a, b, c, alpha, beta, gamma, label_pre):
        if label_pre1 == 1 or label_pre1 == 2:
            # alpha gamma 90°
            a_en[i]=a1
            b_en[i]=b1
            c_en[i]=c1
            #alpha_en[i]=alpha1
            alpha_en[i]=0.5*PI
            beta_en[i]=beta1
            #gamma_en[i]=alpha1
            gamma_en[i]=0.5*PI
        elif label_pre1 >=3 and label_pre1 <= 6:
            # alpha beta gamma 90°
            a_en[i]=a1
            b_en[i]=b1
            c_en[i]=c1
            '''
            alpha_en[i]=alpha1
            beta_en[i]=alpha1
            gamma_en[i]=alpha1
            '''
            alpha_en[i]=0.5*PI
            beta_en[i]=0.5*PI
            gamma_en[i]=0.5*PI
        elif label_pre1 == 7 or label_pre1 == 8:
            # a b; alpha beta gamma 90°
            a_en[i]=a1
            b_en[i]=a1
            c_en[i]=c1
            '''
            alpha_en[i]=alpha1
            beta_en[i]=alpha1
            gamma_en[i]=alpha1
            '''
            alpha_en[i]=0.5*PI
            beta_en[i]=0.5*PI
            gamma_en[i]=0.5*PI
        elif label_pre1 == 9:
            # a b; alpha beta 90° gamma 120°
            a_en[i]=a1
            b_en[i]=a1
            c_en[i]=c1
            '''
            alpha_en[i]=alpha1
            beta_en[i]=alpha1
            gamma_en[i]=gamma1
            '''
            alpha_en[i]=0.5*PI
            beta_en[i]=0.5*PI
            gamma_en[i]=0.66667*PI
        elif label_pre1 == 10:
            # a b; alpha beta 90° gamma 120°
            a_en[i]=a1
            b_en[i]=a1
            c_en[i]=c1
            '''
            alpha_en[i]=alpha1
            beta_en[i]=alpha1
            gamma_en[i]=gamma1
            '''
            alpha_en[i]=0.5*PI
            beta_en[i]=0.5*PI
            gamma_en[i]=0.66667*PI
        elif label_pre1 >=11 and label_pre1 <= 13:
            # a b c; alpha beta gamma 90°
            #a_en[i]=a1
            #b_en[i]=a1
            #c_en[i]=a1
            a_en[i]=b1
            b_en[i]=b1
            c_en[i]=b1
            
            '''
            alpha_en[i]=alpha1
            beta_en[i]=alpha1
            gamma_en[i]=alpha1
            '''
            alpha_en[i]=0.5*PI
            beta_en[i]=0.5*PI
            gamma_en[i]=0.5*PI
        else:
            a_en[i]=a1
            b_en[i]=b1
            c_en[i]=c1
            alpha_en[i]=alpha1
            beta_en[i]=beta1
            gamma_en[i]=gamma1
        i += 1
    return [a_en, b_en, c_en, alpha_en, beta_en, gamma_en]

class EXTRA_LOSS(nn.Module):
    def __init__(self):
        super(EXTRA_LOSS, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self, prediction, label_pre, losses):
        loss_extra = 0
        [a, b, c, alpha, beta, gamma] = prediction
        [l0s, l1s, l2s, l3s, l4s, l5s] = losses
        for a1, b1, c1, alpha1, beta1, gamma1, l0, l1, l2, l3, l4, l5, label_pre1 in zip(a, b, c, alpha, beta, gamma, l0s, l1s, l2s, l3s, l4s, l5s, label_pre):
            if label_pre1.data == 1 or label_pre1.data == 2:
                #la = self.monoclinic(a1, b1, c1, alpha1, beta1, gamma1)
                la = l0+l1+l2+l4
            elif label_pre1.data >=3 and label_pre1.data <= 6:
                #la = self.orthorhombic(a1, b1, c1, alpha1, beta1, gamma1)
                la = l0+l1+l2
            elif label_pre1.data == 7 or label_pre1.data == 8:
                #la = self.tetragonal(a1, b1, c1, alpha1, beta1, gamma1)
                la = l0+l2
            elif label_pre1.data == 9:
                #la = self.trigonal(a1, b1, c1, alpha1, beta1, gamma1)
                la = l0+l2
            elif label_pre1.data == 10:
                #la = self.hexagonal(a1, b1, c1, alpha1, beta1, gamma1)
                la = l0+l2
            elif label_pre1.data >=11 and label_pre1.data <= 13:
                #la = self.cubic(a1, b1, c1, alpha1, beta1, gamma1)
                la = l0
            else:
                la = l0+l1+l2+l3+l4+l5
            loss_extra += la
        
        return la
    def cubic(self, a,b,c,alpha,beta,gamma):
        l1 = self.criterion(a,b)
        l2 = self.criterion(a,c)
        l3 = self.criterion(b,c)
        l4 = self.criterion(alpha,beta)
        l5 = self.criterion(alpha,gamma)
        l6 = self.criterion(beta,gamma)
        return l1+l2+l3+l4+l5+l6
    def hexagonal(self, a,b,c,alpha,beta,gamma):
        l1 = self.criterion(a,b)
        l4 = self.criterion(alpha,beta)
        return l1+l4
    def trigonal(self, a,b,c,alpha,beta,gamma):
        l1 = self.criterion(a,b)
        l4 = self.criterion(alpha,beta)
        return l1+l4
    def tetragonal(self, a,b,c,alpha,beta,gamma):
        l1 = self.criterion(a,b)
        l4 = self.criterion(alpha,beta)
        l5 = self.criterion(alpha,gamma)
        l6 = self.criterion(beta,gamma)
        return l1+l4+l5+l6
    def orthorhombic(self, a,b,c,alpha,beta,gamma):
        l4 = self.criterion(alpha,beta)
        l5 = self.criterion(alpha,gamma)
        l6 = self.criterion(beta,gamma)
        return l4+l5+l6
    def monoclinic(self, a,b,c,alpha,beta,gamma):
        l5 = self.criterion(alpha,gamma)
        return l5
