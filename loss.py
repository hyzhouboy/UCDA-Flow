import torch
import os
import torch.nn.functional as F
MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid_ = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid_[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid_.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def disp_pyramid_loss(disp_preds, disp_gt, gt_mask, pseudo_gt_disp, pseudo_mask, load_pseudo_gt=False):
    """ Loss function defined over sequence of flow predictions """
    pyramid_loss = []
    pseudo_pyramid_loss = []
    disp_loss = 0
    pseudo_disp_loss = 0
    # Loss weights
    if len(disp_preds) == 5:
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
    elif len(disp_preds) == 4:
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
    elif len(disp_preds) == 3:
        pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
    elif len(disp_preds) == 1:
        pyramid_weight = [1.0]  # highest loss only
    else:
        raise NotImplementedError

    for k in range(len(disp_preds)):
        pred_disp = disp_preds[k]
        weight = pyramid_weight[k]

        if pred_disp.size(-1) != disp_gt.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, size=(disp_gt.size(-2), disp_gt.size(-1)),
                                        mode='bilinear', align_corners=False) * (disp_gt.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        curr_loss = F.smooth_l1_loss(pred_disp[gt_mask], disp_gt[gt_mask],
                                        reduction='mean')
        disp_loss += weight * curr_loss
        pyramid_loss.append(curr_loss)

        # Pseudo gt loss
        if load_pseudo_gt:
            pseudo_curr_loss = F.smooth_l1_loss(pred_disp[pseudo_mask], pseudo_gt_disp[pseudo_mask],
                                                reduction='mean')
            pseudo_disp_loss += weight * pseudo_curr_loss

            pseudo_pyramid_loss.append(pseudo_curr_loss)
            
    total_loss = disp_loss + pseudo_disp_loss
    metrics = {
        'disp_epe': pyramid_loss[-1],
    }

    return total_loss, metrics



def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()




def motion_consis_loss(flow_preds, rigid_flow_preds, mask, max_flow=300):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0
    motion_consis = 0.0
    loss_weight = 0.05

    mag = torch.sum(rigid_flow_preds**2, dim=1).sqrt()
    mask_ = (mask >= 0.5) & (mag < max_flow)

    motion_consis = (flow_preds[-1] - rigid_flow_preds).abs()
    motion_consis = loss_weight * (mask_[:, None] * motion_consis).mean()

    motion_epe = loss_weight * torch.sum((flow_preds[-1] - rigid_flow_preds)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)[mask_.view(-1)]

    metrics = {
        'motion_consis': motion_epe.mean().item(),
    }

    return motion_consis, metrics



def cross_domain_consis_loss(foggy_flow_preds, clean_flow_preds):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0

    loss_weight = 0.4
    loss = (foggy_flow_preds - clean_flow_preds).abs()
    loss = loss_weight * loss.mean()

    motion_epe = loss_weight * torch.sum((foggy_flow_preds - clean_flow_preds)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)

    metrics = {
        'foggy_domain_epe': motion_epe.mean().item(),
        'foggy_domain_1px': (motion_epe < 1).float().mean().item(),
        'foggy_domain_3px': (motion_epe < 3).float().mean().item(),
        'foggy_domain_5px': (motion_epe < 5).float().mean().item(),
    }

    return loss, metrics



def self_supervised_loss(flow_preds, flow_pseudo):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0

    loss_weight = 0.5
    loss = (flow_preds - flow_pseudo).abs()
    loss = loss_weight * loss.mean()

    motion_epe = loss_weight * torch.sum((flow_preds - flow_pseudo)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)

    metrics = {
        'pseudo_epe': motion_epe.mean().item(),
        'pseudo_1px': (motion_epe < 1).float().mean().item(),
        'pseudo_3px': (motion_epe < 3).float().mean().item(),
        'pseudo_5px': (motion_epe < 5).float().mean().item(),
    }

    return loss, metrics


# tensor_b: 指导作用的
def loss_KL_div(tensor_a, tensor_b, reduction='mean'):
    log_a = F.log_softmax(tensor_a)
    softmax_b = F.softmax(tensor_b,dim=-1)

    kl_mean = F.kl_div(log_a, softmax_b, reduction=reduction)

    return kl_mean