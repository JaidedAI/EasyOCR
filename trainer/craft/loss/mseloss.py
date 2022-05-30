import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, gt_region, gt_affinity, pred_region, pred_affinity, conf_map):
        loss = torch.mean(
            ((gt_region - pred_region).pow(2) + (gt_affinity - pred_affinity).pow(2))
            * conf_map
        )
        return loss


class Maploss_v2(nn.Module):
    def __init__(self):

        super(Maploss_v2, self).__init__()

    def batch_image_loss(self, pred_score, label_score, neg_rto, n_min_neg):

        # positive_loss
        positive_pixel = (label_score > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)

        positive_loss_region = pred_score * positive_pixel

        # negative_loss
        negative_pixel = (label_score <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)
        negative_loss_region = pred_score * negative_pixel

        if positive_pixel_number != 0:
            if negative_pixel_number < neg_rto * positive_pixel_number:
                negative_loss = (
                    torch.sum(
                        torch.topk(
                            negative_loss_region.view(-1), n_min_neg, sorted=False
                        )[0]
                    )
                    / n_min_neg
                )
            else:
                negative_loss = torch.sum(
                    torch.topk(
                        negative_loss_region.view(-1),
                        int(neg_rto * positive_pixel_number),
                        sorted=False,
                    )[0]
                ) / (positive_pixel_number * neg_rto)
            positive_loss = torch.sum(positive_loss_region) / positive_pixel_number
        else:
            # only negative pixel
            negative_loss = (
                torch.sum(
                    torch.topk(negative_loss_region.view(-1), n_min_neg, sorted=False)[
                        0
                    ]
                )
                / n_min_neg
            )
            positive_loss = 0.0
        total_loss = positive_loss + negative_loss
        return total_loss

    def forward(
        self,
        region_scores_label,
        affinity_socres_label,
        region_scores_pre,
        affinity_scores_pre,
        mask,
        neg_rto,
        n_min_neg,
    ):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        assert (
            region_scores_label.size() == region_scores_pre.size()
            and affinity_socres_label.size() == affinity_scores_pre.size()
        )
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_socres_label)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)

        char_loss = self.batch_image_loss(
            loss_region, region_scores_label, neg_rto, n_min_neg
        )
        affi_loss = self.batch_image_loss(
            loss_affinity, affinity_socres_label, neg_rto, n_min_neg
        )
        return char_loss + affi_loss


class Maploss_v3(nn.Module):
    def __init__(self):

        super(Maploss_v3, self).__init__()

    def single_image_loss(self, pre_loss, loss_label, neg_rto, n_min_neg):

        batch_size = pre_loss.shape[0]

        positive_loss, negative_loss = 0, 0
        for single_loss, single_label in zip(pre_loss, loss_label):

            # positive_loss
            pos_pixel = (single_label >= 0.1).float()
            n_pos_pixel = torch.sum(pos_pixel)
            pos_loss_region = single_loss * pos_pixel
            positive_loss += torch.sum(pos_loss_region) / max(n_pos_pixel, 1e-12)

            # negative_loss
            neg_pixel = (single_label < 0.1).float()
            n_neg_pixel = torch.sum(neg_pixel)
            neg_loss_region = single_loss * neg_pixel

            if n_pos_pixel != 0:
                if n_neg_pixel < neg_rto * n_pos_pixel:
                    negative_loss += torch.sum(neg_loss_region) / n_neg_pixel
                else:
                    n_hard_neg = max(n_min_neg, neg_rto * n_pos_pixel)
                    # n_hard_neg = neg_rto*n_pos_pixel
                    negative_loss += (
                        torch.sum(
                            torch.topk(neg_loss_region.view(-1), int(n_hard_neg))[0]
                        )
                        / n_hard_neg
                    )
            else:
                # only negative pixel
                negative_loss += (
                    torch.sum(torch.topk(neg_loss_region.view(-1), n_min_neg)[0])
                    / n_min_neg
                )

        total_loss = (positive_loss + negative_loss) / batch_size

        return total_loss

    def forward(
        self,
        region_scores_label,
        affinity_scores_label,
        region_scores_pre,
        affinity_scores_pre,
        mask,
        neg_rto,
        n_min_neg,
    ):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert (
            region_scores_label.size() == region_scores_pre.size()
            and affinity_scores_label.size() == affinity_scores_pre.size()
        )
        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_scores_label)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)
        char_loss = self.single_image_loss(
            loss_region, region_scores_label, neg_rto, n_min_neg
        )
        affi_loss = self.single_image_loss(
            loss_affinity, affinity_scores_label, neg_rto, n_min_neg
        )

        return char_loss + affi_loss
