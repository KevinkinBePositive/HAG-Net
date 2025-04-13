import torch
import torch.nn.functional as F

def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()

def bce_cel_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    eps = 1e-6
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    numerator = (pred - inter).sum() + (mask - inter).sum()
    denominator = pred.sum() + mask.sum()
    weighted_cel = numerator / (denominator + eps)
    return (weighted_bce + weighted_cel).mean()

def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()

def boundary_gradient_loss(pred, mask):
    # 使用sigmoid将预测值转换为0到1之间的值
    pred = torch.sigmoid(pred)
    #print('pred', pred)
    # 计算Dice系数损失
    inter = pred * mask
    union = pred + mask
    dice_loss = 1 - (2. * inter + 1) / (union + 1)
    #print('dice_loss',dice_loss)
    
    # 计算梯度损失 (Sobel算子)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    
    # 对预测和mask图像计算梯度
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    mask_grad_x = F.conv2d(mask, sobel_x, padding=1)
    mask_grad_y = F.conv2d(mask, sobel_y, padding=1)
    
    # 计算梯度损失，比较预测和真实mask的梯度差异
    pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
    mask_grad = torch.sqrt(mask_grad_x ** 2 + mask_grad_y ** 2)
    
    grad_loss = torch.mean((pred_grad - mask_grad) ** 2)
    
    # 组合Dice损失和梯度损失
    #total_loss = dice_loss + grad_loss
    
    return (dice_loss + grad_loss).mean()

def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return (1 - Tversky) ** gamma

def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return bce + (1 - Tversky) ** gamma