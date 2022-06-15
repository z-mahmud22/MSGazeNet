# -*- coding: utf-8 -*-
"""
Created on Tue Mar 1 17:35:03 2022

@author: zunayed mahmud
"""

import models.msgazenet as msgazenet
import gaze_estimation.reader as reader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import warnings
  

if __name__ == "__main__":
  config = yaml.load(open('/configs/utm_config.yaml'), Loader=yaml.FullLoader)
  config = config["train"]
  cudnn.benchmark=True

  imagepath = config["data"]["image"]
  train_labelpath = config["data"]["label"]
  test_labelpath = config["data"]["label"]
  modelname = config["save"]["model_name"]

  train_folder = os.listdir(train_labelpath)
  train_folder.sort()
  test_folder = os.listdir(test_labelpath)
  test_folder.sort()
  
  if not os.path.exists(os.path.join(config["save"]["save_path"], f"msgazenet")):
    os.makedirs(os.path.join(config["save"]["save_path"], f"msgazenet"))
  with open(os.path.join(config["save"]["save_path"], f"msgazenet/", "model_params"), 'w') as outfile:
    params = f"Model:wideresnet num_block:3 depth:16 widen_factor:4 loss_fn:MSE base_lr:{config['params']['lr']} BS:{config['params']['batch_size']} dr:0.5"
    outfile.write(params)
  
  P_list={}
  P_list['0']=['s'+str(i).zfill(2)+'.label' for i in range(17)]
  P_list['1']=['s'+str(i).zfill(2)+'.label' for i in range(17,34)]
  P_list['2']=['s'+str(i).zfill(2)+'.label' for i in range(34,50)]  
    
  for i in range(3):
    trains = copy.deepcopy(train_folder)
    trains = create_trainset(trains, P_list[str(i)])
    tests = create_testset(P_list[str(i)])
    print(f"Train Set:{trains}")
    print(f"Test Set:{tests}")

    trainlabelpath = [os.path.join(train_labelpath, j) for j in trains] 
    testlabelpath = [os.path.join(test_labelpath, j) for j in tests]

    savepath = os.path.join(config["save"]["save_path"], f"msgazenet/{i}")
    if not os.path.exists(savepath):
      os.makedirs(savepath)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Read Traindata")
    train_dataset = reader.txtload(trainlabelpath, imagepath, config['params']['batch_size'], device, mode='train', shuffle=True, num_workers=0, header=True)
    print("Read Test Data")  
    test_dataset = reader.txtload(testlabelpath, imagepath, config['params']['batch_size'], device, mode='test', shuffle=True, num_workers=0, header=True)

    print("Model building")
    model_builder = msgazenet.build_WideResNet_f3(1, 16, 4, 0.01, 0.1, 0.5)
    net = model_builder.build(2)
    net.train()
    net.to(device)

    print("optimizer building")
    lossfunc = config["params"]["loss"]
    loss_op = getattr(nn, lossfunc)().cuda()
    base_lr = config["params"]["lr"]

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(net.parameters(),lr=base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=decayratio, patience=3)

    print("Traning")
    train_length = len(train_dataset)
    total_train = train_length * config["params"]["epoch"]
    cur = 0
    min_error = 100
    timebegin = time.time()
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
      for epoch in range(1, config["params"]["epoch"]+1):
        train_accs = 0
        train_loss = 0
        train_count = 0

        for i, (data, label) in enumerate(train_dataset):

          # Acquire data
          data["eye"] = data["eye"].to(device)
          data["iris"] = data["iris"].to(device)
          data["eye_mask"] = data["eye_mask"].to(device)
          label = label.to(device)
   
          # forward
          gaze = net(data)
          
          for k, g_pred in enumerate(gaze):
            g_pred = g_pred.cpu().detach().numpy()
            train_count += 1
            train_accs += angular(gazeto3d(g_pred), gazeto3d(label.cpu().numpy()[k]))

          # loss calculation
          loss = loss_op(gaze, label)
          train_loss+=loss.item()*data["eye"].size(0)
          optimizer.zero_grad()

          # backward
          loss.backward()
          optimizer.step()
          cur += 1

          
          # print logs
          if i % 50 == 0:
            timeend = time.time()
            resttime = (timeend - timebegin)/cur * (total_train-cur)/3600
            log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{train_length}] Train_loss:{loss} Train_AE:{mean_angular_error(gaze.cpu().detach().numpy(),label.cpu().detach().numpy(), gaze.shape[0])} lr:{optimizer.param_groups[0]['lr']}, remaining time:{resttime:.2f}h"
            print(log)
           
        train_epoch_loss = train_loss/train_count
        train_epoch_acc = train_accs/train_count
        
        logger = f"[{epoch}]: train_epoch_loss:{train_epoch_loss} train_epoch_AE:{train_epoch_acc} lr:{optimizer.param_groups[0]['lr']}, remaining time:{resttime:.2f}h"
        print(logger)
        outfile.write(logger + "\n")
        sys.stdout.flush()   
        outfile.flush()    
        
        #print("Testing")
        net.eval()
        
        test_length = len(test_dataset)
        total_test = test_length * config["params"]["epoch"]
        with torch.no_grad():
            test_accs = 0
            test_loss = 0
            test_count = 0
            for i, (data, label) in enumerate(test_dataset):
                # Acquire test data
                data["eye"] = data["eye"].to(device)
                data["iris"] = data["iris"].to(device)
                data["eye_mask"] = data["eye_mask"].to(device)
                label = label.to(device)
                
                gaze = net(data)
                
                for k, g_pred in enumerate(gaze):
                    g_pred = g_pred.cpu().detach().numpy()
                    test_count += 1
                    test_accs += angular(gazeto3d(g_pred), gazeto3d(label.cpu().numpy()[k]))
                
                loss = loss_op(gaze, label)
                test_loss+=loss.item()*data["eye"].size(0)
                cur += 1
                
                # print logs
                if i % 50 == 0:
                  timeend = time.time()
                  resttime = (timeend - timebegin)/cur * (total_test-cur)/3600
                  log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{test_length}] Test_loss:{loss} Test_AE:{mean_angular_error(gaze.cpu().detach().numpy(),label.cpu().detach().numpy(), gaze.shape[0])} lr:{optimizer.param_groups[0]['lr']}"
                  print(log)
                  
            test_epoch_loss = test_loss/test_count
            test_epoch_acc = test_accs/test_count
            scheduler.step(test_epoch_acc)
            
            logger = f"[{epoch}]: test_epoch_loss:{test_epoch_loss} test_epoch_AE:{test_epoch_acc} lr:{optimizer.param_groups[0]['lr']}"
            print(logger)
            outfile.write(logger + "\n")
            sys.stdout.flush()   
            outfile.flush()
            
            if test_epoch_acc<min_error:
                torch.save(net.state_dict(), os.path.join(savepath, f"E:{epoch}_AE:{test_epoch_acc}_L:{test_epoch_loss}.pt"))
                min_error=test_epoch_acc
        
