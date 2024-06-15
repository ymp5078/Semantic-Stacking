import numpy as np
import torch
# from medpy import metric
import binary
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
class DiceLossBinary(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossBinary, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, sigmoid=False):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

# class ConsistencyLoss(nn.Module):
#     def __init__(self, loss_factor=0.1):
#         super(ConsistencyLoss, self).__init__()
#         self.loss_factor = loss_factor
   
#     def calc_loss(self, emb_1, emb_2):
#         """
#             emb: [b,n,c]
#         """
#         assert emb_1.size() == emb_2.size()
#         cos_sim = F.cosine_similarity(emb_1,emb_2,dim=-1, eps=1e-8)
#         # print(cos_sim)
#         loss = 1. - cos_sim
#         return loss

#     def forward(self, emb_1, emb_2):
#         nd_loss = self.calc_loss(emb_1, emb_2) * self.loss_factor
#         return nd_loss.mean()
    
class ConsistencyLoss(nn.Module):
    def __init__(self, loss_factor=0.1, similarity='cosine',feature_dim=-1):
        super(ConsistencyLoss, self).__init__()
        self.loss_factor = loss_factor
        self.similarity = similarity
        self.feature_dim = feature_dim
   
    def calc_loss(self, emb_1, emb_2):
        """
            emb: [b,n,c]
        """
        assert emb_1.size() == emb_2.size()
        if self.similarity == 'cosine':
            cos_sim = F.cosine_similarity(emb_1,emb_2,dim=self.feature_dim, eps=1e-8)
            # print(cos_sim)
            loss = 1. - cos_sim
        else:
            loss = F.l1_loss(emb_2,emb_1)
        return loss

    def forward(self, emb_1, emb_2):
        nd_loss = self.calc_loss(emb_1, emb_2) * self.loss_factor
        return nd_loss.mean()

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

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd95 = binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list