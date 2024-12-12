import torch.nn as nn
from monai.losses import DiceLoss
from lsc_loss import ConsistencyLoss, KDLoss

class SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss=nn.CrossEntropyLoss()
        self.dice_loss=DiceLoss(to_onehot_y=True,softmax=True,squared_pred=True,smooth_nr=0.0,smooth_dr=1e-6)
        
        self.kd_loss = ConsistencyLoss(loss_factor=1.) #KDLoss(kl_loss_factor=1.0,T=1.0)
        self.consist_loss = ConsistencyLoss(loss_factor=1.)
        self.weight_dict={'ce_loss':1, 'dice_loss':1} 
        self.lsc_weight_dict = {'kd_loss': 0,'consist_loss':1}

    def get_loss(self,  pred, gt):
        if len(gt.size())==4 and gt.size(1)==1:
            gt=gt[:,0]

        if type(pred) is not list:
            _ce=self.ce_loss(pred,gt)
            _dc=self.dice_loss(pred,gt.unsqueeze(1))
            return {'ce_loss': _ce,'dice_loss':_dc}
        else:
            ce=0
            dc=0
            for p in pred:
                ce+=self.ce_loss(p,gt)
                dc+=self.dice_loss(p,gt.unsqueeze(1))
            return {'ce_loss': ce, 'dice_loss':dc}
        
    def get_lsc_loss(self,aug_logits,logits,aug_features,features,seg_masks=None ):
        if type(logits) is not list:
            loss_consist = self.consist_loss(features,aug_features)
            loss_kd = self.kd_loss(aug_logits,logits,seg_masks=seg_masks)
            return {'kd_loss': loss_kd,'consist_loss':loss_consist}
        else:
            loss_consist=0
            loss_kd=0
            for i in len(logits):
                loss_consist += self.consist_loss(features[i],aug_features[i])
                loss_kd += self.kd_loss(aug_logits[i],logits[i])
            
            return {'kd_loss': loss_kd,'consist_loss':loss_consist}
