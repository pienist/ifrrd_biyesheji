import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ground-truth 热图生成
# ---------------------------------------------------------------------------
def gaussian2d(shape, sigma=2.0):
    """生成 2D 高斯核。shape = (H, W)"""
    h, w = shape
    y = torch.arange(h, dtype=torch.float32)
    x = torch.arange(w, dtype=torch.float32)
    y, x = torch.meshgrid(y, x, indexing="ij")
    center_y, center_x = h / 2, w / 2
    g = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    return g


def draw_gaussian(heatmap, center, radius=2, sigma=2.0):
    """
    在 heatmap 上画一个高斯热点。
    heatmap: (H, W) tensor，原地修改
    center:  (cy, cx) int tuple
    radius:  高斯半径（决定绘制范围）
    """
    h, w = heatmap.shape
    cy, cx = int(center[0]), int(center[1])

    left, right = max(0, cx - radius), min(w - 1, cx + radius)
    top, bottom = max(0, cy - radius), min(h - 1, cy + radius)

    local_h = bottom - top + 1
    local_w = right - left + 1
    g = gaussian2d((local_h, local_w), sigma=sigma)

    heatmap[top:bottom + 1, left:right + 1] = torch.max(
        heatmap[top:bottom + 1, left:right + 1],
        g.to(heatmap.device),
    )


def make_heatmap_gt(heatmaps, targets, radius=2, sigma=2.0):
    """
    为一批样本生成检测 ground truth 热图。
    heatmaps: (B, H, W) tensor，初始化为 0
    targets:  List[Tensor]，每个元素 (N, 4) → [cx, cy, w, h]（绝对像素坐标，float）
    """
    B, H, W = heatmaps.shape
    for b in range(B):
        for box in targets[b]:
            cx, cy, w, h = box
            draw_gaussian(heatmaps[b], (cy, cx), radius=radius, sigma=sigma)


# ---------------------------------------------------------------------------
# Loss 函数
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    CenterNet 风格 Focal Loss。
    L = -((1 - p)^{gamma} * log(p))    at y = 1 (positive)
      = -(p^{gamma} * log(1 - p))    at y = 0 (negative)
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        pred: (B, 1, H, W) sigmoid 输出
        gt:   (B, 1, H, W) 0~1 热图 GT
        """
        pos_mask = gt.eq(1.0).float()
        neg_mask = gt.lt(1.0).float()

        neg_weights = torch.pow(1.0 - gt, self.beta)

        pos_loss = torch.log(pred + 1e-6) * torch.pow(1.0 - pred, self.alpha) * pos_mask
        neg_loss = torch.log(1.0 - pred + 1e-6) * torch.pow(pred, self.alpha) * neg_weights * neg_mask

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        loss = -(pos_loss + neg_loss) / max(num_pos, 1)
        return loss


class DiceLoss(nn.Module):
    """Dice Loss，用于主分割头（PSPHead）。支持 class_weight 来处理小目标。"""

    def __init__(self, eps=1e-6, use_focal=False, alpha=2.0, beta=4.0, class_weight=None):
        super().__init__()
        self.eps = eps
        self.use_focal = use_focal
        self.alpha = alpha
        self.beta = beta
        self.class_weight = class_weight  # [背景权重, 前景权重]，如 [1.0, 50.0]

    def forward(self, pred, gt):
        """
        pred: (B, 1, H, W) sigmoid
        gt:   (B, 1, H, W) 二值掩码 0~1
        """
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        intersection = (pred * gt).sum(dim=1)  # (B,)
        union = pred.sum(dim=1) + gt.sum(dim=1)

        if self.use_focal:
            # Focal Dice Loss：对难分样本给予更高权重
            dice = (2.0 * intersection + self.eps) / (union + self.eps)
            focal_weight = torch.pow(1.0 - dice, self.alpha)
            loss = focal_weight * (1.0 - dice)
            return loss.mean()
        else:
            # 标准 Dice Loss
            dice = (2.0 * intersection + self.eps) / (union + self.eps)
            loss = 1.0 - dice
            
            # 如果提供了 class_weight，对前景（正类）给予更高权重
            if self.class_weight is not None:
                bg_weight, fg_weight = self.class_weight
                # 计算每个样本的类别权重
                # 前景比例越高的样本，对应的正类权重影响越大
                pos_ratio = gt.sum(dim=1) / (gt.size(1) + self.eps)
                sample_weights = bg_weight + (fg_weight - bg_weight) * pos_ratio
                loss = loss * sample_weights
            
            return loss.mean()


class SoftIoULoss(nn.Module):
    """借鉴 MSHNet 的 SoftIoU Loss，支持 class_weight"""
    
    def __init__(self, eps=1e-6, class_weight=None):
        super().__init__()
        self.eps = eps
        self.class_weight = class_weight  # [背景权重, 前景权重]
    
    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)
        
        intersection = (pred * gt).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        target_sum = gt.sum(dim=1)
        
        iou = (intersection + self.eps) / (pred_sum + target_sum - intersection + self.eps)
        loss = 1 - iou
        
        # 应用 class_weight
        if self.class_weight is not None:
            bg_weight, fg_weight = self.class_weight
            pos_ratio = gt.sum(dim=1) / (gt.size(1) + self.eps)
            sample_weights = bg_weight + (fg_weight - bg_weight) * pos_ratio
            loss = loss * sample_weights
        
        return loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss - 专为小目标设计
    结合了 Dice Loss 和 Focal Loss 的思想
    """

    def __init__(self, gamma=2.0, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, gt):
        """
        pred: (B, 1, H, W) sigmoid
        gt:   (B, 1, H, W) 二值掩码 0~1
        """
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        # 标准 Dice
        intersection = (pred * gt).sum(dim=1)
        union = pred.sum(dim=1) + gt.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)

        # Focal 权重：对低 dice 值（难例）给予更高权重
        focal_weight = torch.pow(1.0 - dice, self.gamma)

        # 加权 Dice Loss
        loss = focal_weight * (1.0 - dice)

        return loss.mean()


class WeightedBCEDiceLoss(nn.Module):
    """
    加权 BCE + Dice Loss
    BCE 提供像素级监督，Dice 提供全局优化
    对正类（目标）给予更高权重
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=10.0, eps=1e-6):
        """
        pos_weight: 正类权重，用于处理类不平衡
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, pred, gt):
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        # 加权 BCE
        bce = -self.pos_weight * gt * torch.log(pred + self.eps) \
              - (1 - gt) * torch.log(1 - pred + self.eps)
        bce_loss = bce.mean()

        # Dice
        intersection = (pred * gt).sum(dim=1)
        union = pred.sum(dim=1) + gt.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失 - 增强小目标边界的学习
    支持 class_weight 处理类不平衡
    """

    def __init__(self, eps=1e-6, class_weight=None):
        super().__init__()
        self.eps = eps
        self.class_weight = class_weight  # [背景权重, 前景权重]

    def forward(self, pred, gt):
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        # 标准 Dice
        intersection = (pred * gt).sum(dim=1)
        union = pred.sum(dim=1) + gt.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice.mean()

        # 简单 BCE（带正类权重）
        pos_weight = 50.0  # 高权重强调正类
        bce = -pos_weight * gt * torch.log(pred + self.eps) \
              - (1 - gt) * torch.log(1 - pred + self.eps)
        bce_loss = bce.mean()

        loss = 0.5 * bce_loss + 0.5 * dice_loss
        
        # 应用 class_weight
        if self.class_weight is not None:
            bg_weight, fg_weight = self.class_weight
            pos_ratio = gt.sum(dim=1) / (gt.size(1) + self.eps)
            sample_weights = bg_weight + (fg_weight - bg_weight) * pos_ratio
            loss = loss * sample_weights
        
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Dice Loss 的泛化，对假阳性/假阴性有不同权重
    适合小目标检测（需要减少假阴性）
    """

    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        """
        alpha: FP 权重
        beta:  FN 权重
        对于小目标，FN 比 FP 更重要，所以 beta > alpha
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, gt):
        """
        pred: (B, 1, H, W) sigmoid
        gt:   (B, 1, H, W) 二值掩码 0~1
        """
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        TP = (pred * gt).sum(dim=1)
        FP = (pred * (1 - gt)).sum(dim=1)
        FN = ((1 - pred) * gt).sum(dim=1)

        tversky = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return (1.0 - tversky).mean()


class ComboLoss(nn.Module):
    """
    Combo Loss：结合 BCE 和 Dice Loss
    BCE 提供像素级监督，Dice 提供全局优化
    """

    def __init__(self, alpha=0.5, beta=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha  # BCE 权重
        self.beta = beta    # Dice 权重
        self.eps = eps

    def forward(self, pred, gt):
        pred = pred.contiguous().view(pred.size(0), -1)
        gt = gt.contiguous().view(gt.size(0), -1)

        # BCE
        bce = -gt * torch.log(pred + self.eps) - (1 - gt) * torch.log(1 - pred + self.eps)
        bce_loss = bce.mean()

        # Dice
        intersection = (pred * gt).sum(dim=1)
        union = pred.sum(dim=1) + gt.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice.mean()

        return self.alpha * bce_loss + self.beta * dice_loss


class BCELoss(nn.Module):
    """BCE Loss，用于辅助分割头（FCNHead）。"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="mean")

    def forward(self, pred, gt):
        return self.bce(pred, gt)


def reg_l1_loss(pred, gt, mask):
    """
    L1 regression loss for wh / offset。
    pred:  (B, 2, H, W)
    gt:    (B, 2, H, W)  目标值
    mask:  (B, H, W)    只在 mask=1 处计算
    """
    mask = mask.unsqueeze(1).float()
    loss = torch.abs(pred - gt) * mask
    return loss.sum() / max(mask.sum(), 1.0)


# ---------------------------------------------------------------------------
# 多任务 Loss 汇总
# ---------------------------------------------------------------------------
class MTLLoss(nn.Module):
    """
    多任务损失汇总。
    训练模式 (task):
      'seg'  → SLSIoULoss(PSP) + w_fcn * BCE(FCN)
      'det'  → Focal(heatmap) + w_wh * L1(wh)
      'mtl'  → 全部
    
    改进：
    - 使用 SLSIoULoss（借鉴 MSHNet）：IoU Loss + 位置敏感损失
    - 支持 class_weight 处理小目标不平衡
    - Warm-up 机制：前 warm_epoch 只用标准 IoU，后面加入 LLoss
    """

    def __init__(self, task="mtl", w_fcn=0.5, w_wh=0.1, loss_type="slsiou",
                 bg_weight=1.0, fg_weight=50.0, warm_epoch=5):
        super().__init__()
        self.task = task
        self.w_fcn = w_fcn
        self.w_wh = w_wh
        self.loss_type = loss_type
        self.warm_epoch = warm_epoch
        
        # class_weight: [背景权重, 前景权重]
        class_weight = [bg_weight, fg_weight] if fg_weight > bg_weight else None

        self.focal_loss = FocalLoss()
        self.bce_loss = BCELoss()

        # 根据 loss_type 选择分割损失
        if loss_type == "dice":
            self.seg_loss = DiceLoss(use_focal=False, class_weight=class_weight)
        elif loss_type == "focal_dice":
            self.seg_loss = DiceLoss(use_focal=True, class_weight=class_weight)
        elif loss_type == "tversky":
            self.seg_loss = TverskyLoss(alpha=0.3, beta=0.7)
        elif loss_type == "weighted_bce_dice":
            self.seg_loss = WeightedBCEDiceLoss(pos_weight=fg_weight)
        elif loss_type == "boundary":
            self.seg_loss = BoundaryAwareLoss(class_weight=class_weight)
        elif loss_type == "softiou":
            self.seg_loss = SoftIoULoss(class_weight=class_weight)
        elif loss_type == "slsiou":
            # 借鉴 MSHNet 的 SLSIoULoss（推荐）
            self.seg_loss = SLSIoULoss(
                warm_epoch=warm_epoch, 
                class_weight=class_weight,
                with_shape=True
            )
        else:
            self.seg_loss = SLSIoULoss(warm_epoch=warm_epoch, class_weight=class_weight)
        
        print(f"[Loss] type={loss_type}, class_weight={class_weight}, warm_epoch={warm_epoch}")

    def forward(self, outputs, targets, epoch=0):
        """
        outputs: dict with keys:
          seg_main:  (B, 1, 320, 320) sigmoid
          seg_aux:   (B, 1, 320, 320) sigmoid
          heatmap:   (B, 1, 320, 320) sigmoid
          wh:        (B, 2, 320, 320) relu
        targets: dict with keys:
          seg_mask:  (B, 1, 320, 320) 二值掩码
          det_hm:    (B, 1, 320, 320) 热图 GT
          det_wh:    (B, 2, 320, 320) wh GT（只在 center 位置有值）
          det_mask:  (B, 320, 320)   center 位置 mask
        epoch: 当前 epoch（用于 SLSIoULoss 的 warm-up）
        """
        losses = {}
        total = 0.0

        if self.task in ("seg", "mtl"):
            # 主分割损失：使用改进的损失函数
            # 如果是 SLSIoULoss，需要传递 epoch 参数
            if hasattr(self.seg_loss, 'forward') and 'epoch' in self.seg_loss.forward.__code__.co_varnames[:self.seg_loss.forward.__code__.co_argcount]:
                L_seg = self.seg_loss(outputs["seg_main"], targets["seg_mask"], epoch=epoch)
            else:
                L_seg = self.seg_loss(outputs["seg_main"], targets["seg_mask"])

            # 辅助分割损失：使用加权 BCE
            L_bce = self.bce_loss(outputs["seg_aux"], targets["seg_mask"])

            losses["seg"] = L_seg
            losses["bce"] = L_bce

            # 组合损失：seg 权重更高（因为辅助头只提供额外监督）
            total = total + L_seg + self.w_fcn * L_bce

        if self.task in ("det", "mtl"):
            L_focal = self.focal_loss(outputs["heatmap"], targets["det_hm"])
            L_wh = reg_l1_loss(outputs["wh"], targets["det_wh"], targets["det_mask"])
            losses["focal"] = L_focal
            losses["wh"] = L_wh
            total = total + L_focal + self.w_wh * L_wh

        losses["total"] = total
        return losses


# ---------------------------------------------------------------------------
# 借鉴 MSHNet 的 SLSIoULoss（Scale and Location Sensitive IoU Loss）
# ---------------------------------------------------------------------------

def LLoss(pred, target, eps=1e-8):
    """
    位置敏感损失：结合角度损失和长度损失
    pred: (B, 1, H, W) sigmoid
    target: (B, 1, H, W) 二值掩码
    """
    B, _, H, W = pred.shape
    loss = torch.tensor(0.0, requires_grad=True).to(pred.device)

    x_index = torch.arange(0, W, 1, device=pred.device).view(1, 1, W).repeat((1, H, 1)) / W
    y_index = torch.arange(0, H, 1, device=pred.device).view(1, H, 1).repeat((1, 1, W)) / H

    for i in range(B):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        # 角度损失
        angle_loss = (4 / (torch.pi ** 2)) * (
            torch.square(torch.arctan(pred_centery / (pred_centerx + eps))
                       - torch.arctan(target_centery / (target_centerx + eps)))
        )

        # 长度损失
        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + eps)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + eps)

        length_loss = torch.min(pred_length, target_length) / (torch.max(pred_length, target_length) + eps)

        loss = loss + (1 - length_loss + angle_loss) / B

    return loss


class SLSIoULoss(nn.Module):
    """
    借鉴 MSHNet 的 SLSIoULoss
    IoU Loss + 位置敏感损失（LLoss）
    前 warm_epoch 个 epoch 只用标准 IoU，后面加入 LLoss
    """

    def __init__(self, warm_epoch=5, eps=1e-6, class_weight=None, with_shape=True):
        super().__init__()
        self.warm_epoch = warm_epoch
        self.eps = eps
        self.class_weight = class_weight
        self.with_shape = with_shape

    def forward(self, pred, target, epoch=0):
        """
        pred: (B, 1, H, W) sigmoid
        target: (B, 1, H, W) 二值掩码
        epoch: 当前 epoch，用于控制 warm-up
        """
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))

        # 距离因子
        dis = torch.pow((pred_sum - target_sum) / 2, 2)

        # IoU
        iou = (intersection_sum + self.eps) / (pred_sum + target_sum - intersection_sum + self.eps)

        # 位置敏感权重 alpha
        alpha = (torch.min(pred_sum, target_sum) + dis + self.eps) / (torch.max(pred_sum, target_sum) + dis + self.eps)

        siou_loss = alpha * iou

        if epoch > self.warm_epoch:
            lloss = LLoss(pred, target, self.eps)
            if self.with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - iou.mean()

        # 应用 class_weight（对每个样本单独加权，增强前景学习信号）
        if self.class_weight is not None and loss.requires_grad:
            bg_weight, fg_weight = self.class_weight
            pos_ratio = target.sum(dim=(1, 2, 3)) / (target.size(1) * target.size(2) * target.size(3) + self.eps)
            # 用 min 确保样本权重至少为 fg_weight（所有样本都能收到前景梯度）
            sample_weights = torch.clamp(bg_weight + (fg_weight - bg_weight) * pos_ratio, min=fg_weight)
            loss = loss * sample_weights.mean()

        return loss
