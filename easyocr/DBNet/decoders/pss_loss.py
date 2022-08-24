import torch
import torch.nn as nn
import torch.nn.functional as F

class PSS_Loss(nn.Module):

    def __init__(self, cls_loss):
        super(PSS_Loss, self).__init__()
        self.eps = 1e-6
        self.criterion = eval('self.' + cls_loss + '_loss')

    def dice_loss(self, pred, gt, m):
        intersection = torch.sum(pred*gt*m)
        union = torch.sum(pred*m) + torch.sum(gt*m) + self.eps
        loss = 1 - 2.0*intersection/union
        if loss > 1:
            print(intersection, union)
        return loss

    def dice_ohnm_loss(self, pred, gt, m):
        pos_index = (gt == 1) * (m == 1)
        neg_index = (gt == 0) * (m == 1)
        pos_num = pos_index.float().sum().item()
        neg_num = neg_index.float().sum().item()
        if pos_num == 0 or neg_num < pos_num*3.0:
            return self.dice_loss(pred, gt, m)
        else:
            neg_num = int(pos_num*3)
            pos_pred = pred[pos_index]
            neg_pred = pred[neg_index]
            neg_sort, _ = torch.sort(neg_pred, descending=True)
            sampled_neg_pred = neg_sort[:neg_num]
            pos_gt = pos_pred.clone()
            pos_gt.data.fill_(1.0)
            pos_gt = pos_gt.detach()
            neg_gt = sampled_neg_pred.clone()
            neg_gt.data.fill_(0)
            neg_gt = neg_gt.detach()
            tpred = torch.cat((pos_pred, sampled_neg_pred))
            tgt = torch.cat((pos_gt, neg_gt))
            intersection = torch.sum(tpred * tgt)
            union = torch.sum(tpred) + torch.sum(gt) + self.eps
            loss = 1 - 2.0 * intersection / union
        return loss

    def focal_loss(self, pred, gt, m, alpha=0.25, gamma=0.6):
        pos_mask = (gt == 1).float()
        neg_mask = (gt == 0).float()
        mask = alpha*pos_mask * \
            torch.pow(1-pred.data, gamma)+(1-alpha) * \
            neg_mask*torch.pow(pred.data, gamma)
        l = F.binary_cross_entropy(pred, gt, weight=mask, reduction='none')
        loss = torch.sum(l*m)/(self.eps+m.sum())
        loss *= 10
        return loss

    def wbce_orig_loss(self, pred, gt, m):
        n, h, w = pred.size()
        assert (torch.max(gt) == 1)
        pos_neg_p = pred[m.byte()]
        pos_neg_t = gt[m.byte()]
        pos_mask = (pos_neg_t == 1).squeeze()
        w = pos_mask.float() * (1 - pos_mask).sum().item() + \
            (1 - pos_mask).float() * pos_mask.sum().item()
        w = w / (pos_mask.size(0))
        loss = F.binary_cross_entropy(pos_neg_p, pos_neg_t, w, reduction='sum')
        return loss

    def wbce_loss(self, pred, gt, m):
        pos_mask = (gt == 1).float()*m
        neg_mask = (gt == 0).float()*m
        # mask=(pos_mask*neg_mask.sum()+neg_mask*pos_mask.sum())/m.sum()
        # loss=torch.sum(l)
        mask = pos_mask * neg_mask.sum() / pos_mask.sum() + neg_mask
        l = F.binary_cross_entropy(pred, gt, weight=mask, reduction='none')
        loss = torch.sum(l)/(m.sum()+self.eps)
        return loss

    def bce_loss(self, pred, gt, m):
        l = F.binary_cross_entropy(pred, gt, weight=m, reduction='sum')
        loss = l/(m.sum()+self.eps)
        return loss

    def dice_bce_loss(self, pred, gt, m):
        return (self.dice_loss(pred, gt, m) + self.bce_loss(pred, gt, m)) / 2.0

    def dice_ohnm_bce_loss(self, pred, gt, m):
        return (self.dice_ohnm_loss(pred, gt, m) + self.bce_loss(pred, gt, m)) / 2.0

    def forward(self, pred, gt, mask, gt_type='shrink'):
        if gt_type == 'shrink':
            loss = self.get_loss(pred, gt, mask)
            return loss
        elif gt_type == 'pss':
            loss = self.get_loss(pred, gt[:, :4, :, :], mask)
            g_g = gt[:, 4, :, :]
            g_p, _ = torch.max(pred, 1)
            loss += self.criterion(g_p, g_g, mask)
            return loss
        elif gt_type == 'both':
            pss_loss = self.get_loss(pred[:, :4, :, :], gt[:, :4, :, :], mask)
            g_g = gt[:, 4, :, :]
            g_p, _ = torch.max(pred, 1)
            pss_loss += self.criterion(g_p, g_g, mask)
            shrink_loss = self.criterion(
                pred[:, 4, :, :], gt[:, 5, :, :], mask)
            return pss_loss, shrink_loss
        else:
            return NotImplementedError('gt_type [%s] is not implemented', gt_type)

    def get_loss(self, pred, gt, mask):
        loss = torch.tensor(0.)
        for ind in range(pred.size(1)):
            loss += self.criterion(pred[:, ind, :, :], gt[:, ind, :, :], mask)
        return loss
