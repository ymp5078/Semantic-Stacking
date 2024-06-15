import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, ConsistencyLoss, KDLoss, DiceLossBinary
from torchvision import transforms

def trainer_fundus(args, model, snapshot_path):
    from datasets.fundus_dataloader import FundusSegmentationAll
    from datasets import custom_transforms as tr
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(args.img_size),
        # tr.RandomCrop(512),
        # tr.RandomRotate(),
        # tr.RandomFlip(),
        # tr.elastic_transform(),
        # tr.add_salt_pepper_noise(),
        # tr.adjust_light(),
        # tr.eraser(),
        tr.Normalize_tf(fundus=args.dataset=='fundus'),
        tr.ToTensor(num_domain=3 if args.dataset=='fundus' else 5)
    ])
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = FundusSegmentationAll(base_dir=args.root_path, phase='train', splitid=args.datasetTrain,
                                                                transform=composed_transforms_tr,gen_image_dir=args.gen_image_dir)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLossBinary(num_classes)
    consist_loss = ConsistencyLoss(loss_factor=1.)
    kd_loss = KDLoss(kl_loss_factor=1.0,T=1.0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch))
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            gen_image_batch = sampled_batch.get('gen_image',None)
            loss_consist = torch.tensor(0.0)
            loss_kd = torch.tensor(0.0)
            if gen_image_batch is not None:
                gen_image_batch = gen_image_batch.cuda()
                outputs, features = model(image_batch,return_feature=True)
                gen_outputs, gen_features = model(gen_image_batch,return_feature=True)
                loss_ce = ce_loss(outputs, label_batch[:])
                loss_dice = dice_loss(outputs, label_batch, sigmoid=True)
                gen_loss_ce = ce_loss(gen_outputs, label_batch[:])
                gen_loss_dice = dice_loss(gen_outputs, label_batch, sigmoid=True)
                
                
                # loss = 0.5 * loss_ce + 0.5 * loss_dice
                loss = 0.5 * (loss_ce + gen_loss_ce) + 0.5 * (loss_dice + gen_loss_dice)
                if args.consist_loss_weight > 0:
                    loss_consist = consist_loss(features,gen_features)
                    loss += args.consist_loss_weight * loss_consist
                if args.kd_loss_weight > 0:
                    loss_kd = kd_loss(gen_outputs,outputs)
                    loss += args.consist_loss_weight * loss_kd
            else:
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:])
                loss_dice = dice_loss(outputs, label_batch, sigmoid=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            log_dict = {
                'iter_num':iter_num,
                'lr':lr_,
                'total_loss':loss.item(),
                'loss_consist':loss_consist.item(),
                'loss_kd':loss_kd.item(),
            }
            iterator.set_postfix(log_dict)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_consist: %f, loss_kd %f' % (iter_num, loss.item(), loss_ce.item(),loss_consist.item(), loss_kd.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                # labs = label_batch[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"