import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss for semantic segmentation.
    """
    def __init__(self, smooth=1.0, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode target
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dice_loss = 0.0
        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            prob_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            intersection = torch.sum(prob_c * target_c, dim=(1, 2))
            cardinality = torch.sum(prob_c, dim=(1, 2)) + torch.sum(target_c, dim=(1, 2))
            
            score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_loss += (1.0 - score.mean())
            
        return dice_loss / num_classes


class CombinedCrossEntropyDiceLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice Loss as specified in Capstone Proposal.
    Loss = alpha * CrossEntropy + beta * DiceLoss
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=3, class_weights=None):
        super(CombinedCrossEntropyDiceLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weights_tensor = None
            
        self.ce_loss = nn.CrossEntropyLoss(weight=weights_tensor)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        
        total_loss = (self.ce_weight * loss_ce) + (self.dice_weight * loss_dice)
        return total_loss, loss_ce, loss_dice
