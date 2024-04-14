import numpy as np
import torch
import torch.nn.functional as F

def to_one_hot(y, num_classes):
    return torch.eye(num_classes).to(y.device)[y]

def binary_cross_entropy_with_logits(y_true, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)

def cross_entropy(y_true, logits):
    return F.cross_entropy(logits, y_true)

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, logits, num_classes):
        y_true = to_one_hot(y_true, num_classes)
        # Clipping predictions and true values to avoid log(0)
        y_pred = F.softmax(logits, dim=-1)
        y_pred_clipped = torch.clamp(y_pred, 1e-7, 1.0)
        y_true_clipped = torch.clamp(y_true, 1e-4, 1.0)
        ce_loss = -torch.sum(y_true * torch.log(y_pred_clipped), dim=-1)
        rev_ce_loss = -torch.sum(y_pred_clipped * torch.log(y_true_clipped), dim=-1)

        return alpha * torch.mean(ce_loss) + beta * torch.mean(rev_ce_loss)
    return loss

def label_smoothing_regularization(y_true, logits, num_classes):
    epsilon = 0.1
    y_true = to_one_hot(y_true, num_classes)
    y_pred = F.softmax(logits, dim=-1)
    y_smoothed_true = y_true * (1 - epsilon - epsilon / 10.0) + epsilon / 10.0
    y_pred_clipped = torch.clamp(y_pred, 1e-7, 1.0)
    return torch.mean(-torch.sum(y_smoothed_true * torch.log(y_pred_clipped), dim=-1))

def generalized_cross_entropy(y_true, logits, num_classes):
    q = 0.9
    y_true = to_one_hot(y_true, num_classes)
    y_pred = F.softmax(logits, dim=-1)
    t_loss = (1 - torch.pow(torch.sum(y_true * y_pred, dim=-1), q)) / q
    return torch.mean(t_loss)

def joint_optimization_loss(y_true, logits, num_classes):
    y_true = to_one_hot(y_true, num_classes)
    y_pred = F.softmax(logits, dim=-1)
    y_pred_avg = torch.mean(y_pred, dim=0)
    p = torch.ones(10) / 10.
    l_p = - torch.sum(torch.log(y_pred_avg) * p)
    l_e = F.cross_entropy(y_pred, y_pred)
    return F.cross_entropy(y_true, y_pred) + 1.2 * l_p + 0.8 * l_e

def boot_soft(y_true, logits, num_classes):
    beta = 0.95
    y_true = to_one_hot(y_true, num_classes)
    y_pred = F.softmax(logits, dim=-1)
    y_pred = y_pred / torch.sum(y_pred, dim=-1, keepdim=True)
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    return -torch.sum((beta * y_true + (1. - beta) * y_pred) * torch.log(y_pred), dim=-1)

def boot_hard(y_true, logits, num_classes):
    beta = 0.8
    y_true = to_one_hot(y_true, num_classes)
    y_pred = F.softmax(logits, dim=-1)
    y_pred = y_pred / torch.sum(y_pred, dim=-1, keepdim=True)
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    pred_labels = F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_true.shape[1])
    return -torch.sum((beta * y_true + (1. - beta) * pred_labels) * torch.log(y_pred), dim=-1)

def forward(P):
    P = torch.tensor(P, dtype=torch.float32)
    def loss(y_true, logits, num_classes):
        y_true = to_one_hot(y_true, num_classes)
        y_pred = F.softmax(logits, dim=-1)
        y_pred = y_pred / torch.sum(y_pred, axis=-1, keepdim=True)
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        return -torch.sum(y_true * torch.log(torch.matmul(y_pred, P)), axis=-1)
    return loss

def backward(P):
    P_inv = torch.tensor(np.linalg.inv(P), dtype=torch.float32)
    def loss(y_true, logits, num_classes):
        y_true = to_one_hot(y_true, num_classes)
        y_pred = F.softmax(logits, dim=-1)
        y_pred = y_pred / torch.sum(y_pred, dim=-1, keepdim=True)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1 - 1e-7)
        loss = -torch.sum(torch.matmul(y_true, P_inv) * torch.log(y_pred), dim=-1)
        return loss
    return loss


# Define a dictionary mapping short names to loss functions
loss_functions = {
    'bce': binary_cross_entropy_with_logits, 
    'ce': cross_entropy,
    'sce': symmetric_cross_entropy, # Add parameters as needed
    'lsr': label_smoothing_regularization,
    'gce': generalized_cross_entropy,
    'jol': joint_optimization_loss,
    'bs': boot_soft,
    'bh': boot_hard,
    'fwd': forward, # Define parameters for forward function
    'bwd': backward, # Define parameters for backward function
}

def get_loss_function(loss_name):
    if loss_name in loss_functions:
        return loss_functions[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")



