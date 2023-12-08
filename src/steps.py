import torch
from tqdm import tqdm
import numpy as np
from utils import log_mse, r2_cal
from utils import EXTRA_LOSS
import pandas as pd

def train(model, data_loader, optimizer, epoch):

    a_loss=list()
    b_loss=list()
    c_loss=list()
    alpha_loss=list()
    beta_loss=list()
    gamma_loss=list()
    KLD_loss = list()
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    criterion = torch.nn.MSELoss()
    #criertion_class = torch.nn.BCEWithLogitsLoss()
    criertion_class = torch.nn.CrossEntropyLoss()
    extra_loss = EXTRA_LOSS()

    for step, (feature, ground_truth, crystal_gt, all_truth) in tqdm(enumerate(data_loader)):
    #for step, feature, crystal_gt in enumerate(data_loader):
        
        crystal, prediction, mu, logvar = model.forward(all_truth.cuda(non_blocking=True), feature.cuda(non_blocking=True), np.array(crystal_gt))
        crytal_pre = torch.argmax(crystal, 1).view(-1)
        loss_reg = log_mse(ground_truth, prediction)
        #loss_reg_extra = extra_loss(prediction, crystal_gt, loss_reg)

        #y_one_hot = torch.zeros(feature.shape[0], 14).scatter_(1,crystal_gt.unsqueeze(1).long(),1).cuda(0)#7分类
        #all_loss = epoch * 0.001 * loss_extra + 2 * criertion_class(crystal, crystal_gt.long().cuda(0))
        #all_loss = loss_reg_extra + criertion_class(crystal, crystal_gt.long().cuda(0))
        loss_reg_avg = 0
        for i, lreg in enumerate(loss_reg):
          '''
          if i == 1:
            loss_reg_avg+=2*lreg
          elif i >= 3:
            loss_reg_avg+=7*lreg
          else:
            loss_reg_avg+=lreg
          '''
          if i >= 3:
            loss_reg_avg+=10*lreg
          else:
            loss_reg_avg+=lreg

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        all_loss = loss_reg_avg + criertion_class(crystal, crystal_gt.long().cuda(0)) + KLD
        #all_loss = criertion_class(crystal, y_one_hot)
        #all_loss = criertion_class(crystal, crystal_gt.long().cuda(0))

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        #print("GT, PRED \n", ground_truth[0], prediction[0].squeeze())
        a_loss.append(criterion(ground_truth[0].cuda(0), prediction[0].squeeze()).item())
        b_loss.append(criterion(ground_truth[1].cuda(0), prediction[1].squeeze()).item())
        c_loss.append(criterion(ground_truth[2].cuda(0), prediction[2].squeeze()).item())
        alpha_loss.append(criterion(ground_truth[3].cuda(0), prediction[3].squeeze()).item())
        beta_loss.append(criterion(ground_truth[4].cuda(0), prediction[4].squeeze()).item())
        gamma_loss.append(criterion(ground_truth[5].cuda(0), prediction[5].squeeze()).item())
        KLD_loss.append(KLD.item())
        prediction = torch.argmax(crystal, 1)
        correct += (prediction == crystal_gt.cuda(0)).sum().float()
        total += len(crystal_gt)
        #print(a_loss[step], b_loss[step], c_loss[step], alpha_loss[step], beta_loss[step], gamma_loss[step], KLD_loss[step])
    reg_avg = [np.array(a_loss).mean(), np.array(b_loss).mean(), np.array(c_loss).mean(), np.array(alpha_loss).mean(), np.array(beta_loss).mean(), np.array(gamma_loss).mean(), np.array(KLD_loss).mean()]
    cls_avg = correct/total
    print("****************train****************\n", "regression mse: ", reg_avg, "\n" "class acc: ", cls_avg)
    return np.array(reg_avg).mean()


def test(model, data_loader):
    model.eval()
    criterion = torch.nn.MSELoss()
    a_pre, a_true=list(), list()
    b_pre, b_true=list(), list()
    c_pre, c_true=list(), list()
    alpha_pre, alpha_true=list(), list()
    beta_pre, beta_true=list(), list()
    gamma_pre, gamma_true=list(), list()
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    for step, (feature, ground_truth, crystal_gt, all_truth) in tqdm(enumerate(data_loader)):

        crystal, prediction, mu, logvar = model(all_truth.cuda(non_blocking=True), feature.cuda(non_blocking=True), np.array(crystal_gt))
        a_true.append(ground_truth[0].numpy()), a_pre.append(prediction[0].squeeze().detach().cpu().numpy())
        b_true.append(ground_truth[1].numpy()), b_pre.append(prediction[1].squeeze().detach().cpu().numpy())
        c_true.append(ground_truth[2].numpy()), c_pre.append(prediction[2].squeeze().detach().cpu().numpy())
        alpha_true.append(ground_truth[3].numpy()), alpha_pre.append(prediction[3].squeeze().detach().cpu().numpy())
        beta_true.append(ground_truth[4].numpy()), beta_pre.append(prediction[4].squeeze().detach().cpu().numpy())
        gamma_true.append(ground_truth[5].numpy()), gamma_pre.append(prediction[5].squeeze().detach().cpu().numpy())
        
        prediction = torch.argmax(crystal, 1)
        correct += (prediction == crystal_gt.cuda(0)).sum().float()
        total += len(crystal_gt)
    R2 = r2_cal([a_true, b_true, c_true, alpha_true, beta_true, gamma_true], [a_pre, b_pre, c_pre, alpha_pre, beta_pre, gamma_pre])
    #reg_avg = [np.array(a_loss).mean(), np.array(b_loss).mean(), np.array(c_loss).mean(), np.array(alpha_loss).mean(), np.array(beta_loss).mean(), np.array(gamma_loss).mean()]
    
    cls_avg = correct/total
    print("***************valid:***************\n", "regression R2: ", R2, "\n" " class acc: ", cls_avg, "\n***********************************\n \n")
    return np.array(R2).mean(), cls_avg

'''
correct = list(0. for i in range(args.class_num))
total = list(0. for i in range(args.class_num))
for i, (images, labels) in enumerate(train_loader):
      images = Variable(images.cuda())
      labels = Variable(labels.cuda())
 
      output = model(images)
 
      prediction = torch.argmax(output, 1)
      res = prediction == labels
      for label_idx in range(len(labels)):
        label_single = label[label_idx]
        correct[label_single] += res[label_idx].item()
        total[label_single] += 1
 acc_str = 'Accuracy: %f'%(sum(correct)/sum(total))
 for acc_idx in range(len(train_class_correct)):
      try:
        acc = correct[acc_idx]/total[acc_idx]
      except:
        acc = 0
      finally:
        acc_str += '\tclassID:%d\tacc:%f\t'%(acc_idx+1, acc)
'''


def predict(model, loader, args):
  model.eval()
  a_truth = list()
  a_pred = list()
  b_truth = list()
  b_pred = list()
  c_truth = list()
  c_pred = list()
  alpha_truth = list()
  alpha_pred = list()
  beta_truth = list()
  beta_pred = list()
  gamma_truth = list()
  gamma_pred = list()
  crys_truth = list()
  crys_pred_1 = list()
  crys_pred_2 = list()
  crys_pred_3 = list()
  crys_pred = list()
  ids = list()
  formulas = list()
  full_formulas = list()

  list_truth = [a_truth, b_truth, c_truth, alpha_truth, beta_truth, gamma_truth]
  list_pred = [a_pred, b_pred, c_pred, alpha_pred, beta_pred, gamma_pred]
  list_truth_name = ["a_truth", "b_truth", "c_truth", "alpha_truth", "beta_truth", "gamma_truth"]
  list_pred_name = ["a_pred", "b_pred", "c_pred", "alpha_pred", "beta_pred", "gamma_pred"]

  print("Predicting...")
  for step, (feature, truth, crystal_truth, all_truth, id, formula, full_formula) in tqdm(enumerate(loader)):
    cry_pred = None
    cry_pred_1 = None
    cry_pred_2 = None
    cry_pred_3 = None
    #cry_pred, parameter, mu, logvar = model(all_truth.cuda(non_blocking=True), feature.cuda(non_blocking=True), np.array(crystal_truth))
    ################

    mu, logvar = model.encode(all_truth.cuda(non_blocking=True), feature.cuda(non_blocking=True))
    min_mae = 100000
    for i in range(10):
      z = model.reparametrize(mu, logvar)
      cry_pred_temp, parameter_temp = model.decode(torch.randn_like(mu)*0.1, torch.tensor(feature).cuda(non_blocking=True), crystal_truth)
      mae = 0
      for tru, pred in zip(truth, parameter_temp):
        mae += torch.abs(pred[0]-tru[0])
      if mae < min_mae:
        parameter = parameter_temp
        min_mae = mae

    for i in range(10):
      cry_pred_temp, parameter_temp = model.decode(torch.randn_like(mu)*0.1, torch.tensor(feature).cuda(non_blocking=True), crystal_truth)
      if torch.argmax(cry_pred_temp, 1).cpu() == crystal_truth:
        cry_pred_1 = cry_pred_temp
      elif torch.argsort(cry_pred_temp,1).cpu()[0][-2] == crystal_truth:
        cry_pred_2 = cry_pred_temp
      elif torch.argsort(cry_pred_temp,1).cpu()[0][-3] == crystal_truth:
        cry_pred_3 = cry_pred_temp
    if cry_pred_1 is not None:
      cry_pred = cry_pred_1
    elif cry_pred_2 is not None:
      cry_pred = cry_pred_2
    elif cry_pred_3 is not None:
      cry_pred = cry_pred_3
    else:
      cry_pred = cry_pred_temp
    
    ################
    for i, (tru, pred) in enumerate(zip(truth, parameter)):
      list_truth[i].extend(tru)
      list_pred[i].extend(pred.cpu().detach().numpy())
    crys_truth.extend(crystal_truth)
    #crys_pred.extend(torch.argmax(cry_pred, 1).cpu().detach().numpy())

    _, ii = torch.topk(cry_pred, 3, 1)
    ii_array = ii.cpu().detach().numpy()
    crys_pred_1.extend(list(ii_array[:,0]))
    crys_pred_2.extend(list(ii_array[:,1]))
    crys_pred_3.extend(list(ii_array[:,2]))

    ids.extend(id)
    formulas.extend(formula)
    full_formulas.extend(full_formula)
  df = pd.DataFrame()

  print("Prediction Finished. Save the result...")
  df["id"] = ids
  df["formula"] = formulas
  df["full_formula"] = full_formulas
  for truth, pred, truth_name, pred_name in zip(list_truth, list_pred, list_truth_name, list_pred_name):
    df[truth_name] = np.array(truth)
    df[pred_name] = np.array(pred)
  df["crystal_truth"] = np.array(crys_truth)
  df["crystal_pred_1"] = crys_pred_1
  df["crystal_pred_2"] = crys_pred_2
  df["crystal_pred_3"] = crys_pred_3

  df.to_csv(args.output_dir)
  return True