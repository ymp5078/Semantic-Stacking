import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    def __init__(self, loss_factor=0.1, similarity='cosine',filter=True, feature_dim=1):
        super(ConsistencyLoss, self).__init__()
        self.loss_factor = loss_factor
        self.similarity = similarity
        self.filter=filter
        self.feature_dim = feature_dim
        self.weight_factor = 1.
   
    def calc_loss(self, target_emb, source_emb,seg_masks=None):
        """
            emb: [b,n,c]
            seg_masks: [b,h,w]
        """
        assert source_emb.size() == source_emb.size()
        

        if self.similarity == 'cosine':
            cos_sim = F.cosine_similarity(source_emb,target_emb,dim=self.feature_dim, eps=1e-8)
            # print(cos_sim)
            loss = 1. - cos_sim
        else:
            loss = F.l1_loss(source_emb,target_emb,reduction='none')

        if seg_masks is not None:
            h,w = seg_masks.shape[1:]
            total_features = h*w
            seg_masks = F.one_hot(seg_masks)
            seg_masks_count = seg_masks.sum((1,2),keepdim=True)
            # weight is 1- the class freq, only when the class presents
            seg_masks_weight = (seg_masks * (1 - seg_masks_count / total_features)).sum(-1) ** self.weight_factor
            loss = seg_masks_weight * loss
            return loss.sum() / seg_masks_weight.sum()
        else:
            return loss.mean()

    def forward(self, target_emb, source_emb,seg_masks=None):
        nd_loss = self.calc_loss(target_emb, source_emb,seg_masks=seg_masks) * self.loss_factor
        return nd_loss

class KDLoss(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
    def __init__(self, kl_loss_factor=1.0, T=1.0):
        super(KDLoss, self).__init__()
        self.T = T
        self.kl_loss_factor = kl_loss_factor

    def forward(self, s_out, t_out):
        # resize to aviod overflow
        B, C, H, W = s_out.shape
        s_out = s_out.permute(0,2,3,1).reshape(B*H*W,C)
        t_out = t_out.permute(0,2,3,1).reshape(B*H*W,C)
        kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1), 
                           F.softmax(t_out / self.T, dim=1), 
                           reduction='batchmean',
                           ) * self.T * self.T
        return kd_loss * self.kl_loss_factor
    
    
class BinaryKDLoss(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
    def __init__(self, kl_loss_factor=1.0, T=1.0):
        super(BinaryKDLoss, self).__init__()
        self.T = T
        self.kl_loss_factor = kl_loss_factor

    def forward(self, s_out, t_out):
        # resize to aviod overflow
        
        s_out = s_out.reshape(-1,1)

        t_out = t_out.reshape(-1,1)

        kd_loss = F.kl_div(F.logsigmoid(s_out / self.T),
                           F.sigmoid(t_out / self.T), 
                           reduction='batchmean',
                           ) * self.T * self.T
        return kd_loss * self.kl_loss_factor