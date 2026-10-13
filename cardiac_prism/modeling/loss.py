import torch
import torch.nn.functional as F
import torch.nn as nn


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Calculate Dice loss.
    Args:
        inputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth labels.
        num_masks (float): Number of masks for normalization.
    Returns:
        torch.Tensor: Dice loss.
    """
    inputs = inputs.sigmoid()
    Batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    targets = targets.int().float()

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # return loss.sum() / num_masks
    return loss.sum() / Batch_size


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Calculate Sigmoid Cross Entropy loss.
    Args:
        inputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth labels.
        num_masks (float): Number of masks for normalization.
    Returns:
        torch.Tensor: Sigmoid Cross Entropy loss.
    """
    # Flatten inputs and targets
    Batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    targets = targets.int().float()

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # return loss.mean(1).sum() / num_masks
    return loss.mean(1).sum() / Batch_size


def bce_logit_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks=48):
    """
    Calculate Binary Cross Entropy loss with logits.
    Args:
        inputs (torch.Tensor): Model predictions of shape [B, 3].
        targets (torch.Tensor): Ground truth labels of shape [B, 3].
    Returns:
        torch.Tensor: The BCE loss.
    """
    # Ensure the targets are floats
    Batch_size = inputs.shape[0]
    targets = targets.float()

    # Compute the binary cross-entropy loss with logits
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / Batch_size
    # Return the mean loss over all ba


class PointSampleLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs_dict, targets_dict, cal_class_loss=False):
        pred_masks = inputs_dict['pred_masks']
        targets = torch.stack([t['masks'] for t in targets_dict])

        target_shape = targets.shape[2:]
        pred_masks_resized = F.interpolate(pred_masks, size=target_shape, mode='bilinear', align_corners=False)

        num_masks = float(pred_masks.shape[0] * 16)  # Assuming num_classes = 3

        # Calculate the losses
        # class_loss = bce_logit_loss(pred_class, target_class)
        dice = dice_loss(pred_masks_resized, targets, num_masks)
        ce = sigmoid_ce_loss(pred_masks_resized, targets, num_masks)
        total_loss_dice = dice
        total_loss_ce = ce
        # total_loss_query = query_loss
        if "aux_outputs" in inputs_dict:
            for i, aux_output in enumerate(inputs_dict["aux_outputs"]):
                if isinstance(aux_output["pred_masks"], tuple):  # If still tuple, extract further
                    aux_pred_masks = aux_output["pred_masks"][0]
                else:
                    aux_pred_masks = aux_output["pred_masks"]
                # aux_pred_class = aux_output["pred_logits"]
                # aux_pred_query = aux_output['pred_query']
                aux_pred_masks_resized = F.interpolate(aux_pred_masks, size=target_shape, mode='bilinear',
                                                       align_corners=False)
                # aux_pred_class_loss = bce_logit_loss(aux_pred_class,target_class)
                aux_dice = dice_loss(aux_pred_masks_resized, targets, num_masks)
                aux_ce = sigmoid_ce_loss(aux_pred_masks_resized, targets, num_masks)
                # aux_query_loss = inter_group_diversity_loss(aux_pred_query, 12)
                # total_loss_class = total_loss_class + aux_pred_class_loss
                total_loss_dice = total_loss_dice + aux_dice
                total_loss_ce = total_loss_ce + aux_ce
                # total_loss_query = total_loss_query + aux_query_loss

        return {"loss_dice": 5 * total_loss_dice, "loss_ce": 5 * total_loss_ce}
