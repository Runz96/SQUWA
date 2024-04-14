import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self, device, task_num):
        super(AbsWeighting, self).__init__()
        self.device = device
        self.task_num = task_num

    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass

class EW(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """
    def __init__(self, device, task_num):
        super(EW, self).__init__(device, task_num)
        
    def backward(self, losses, **kwargs):
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        return loss, np.ones(self.task_num)

class GLS(AbsWeighting):
    r"""Geometric Loss Strategy (GLS).
    
    This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
    and implemented by us.

    """
    def __init__(self, device, task_num):
        super(GLS, self).__init__(device, task_num)
        
    def backward(self, losses, **kwargs):
        loss = torch.pow(losses.prod(), 1./self.task_num)
        batch_weight = losses / (self.task_num * losses.prod())
        return loss, batch_weight.detach().cpu().numpy()

class RLW(AbsWeighting):
    r"""Random Loss Weighting (RLW).
    
    This method is proposed in `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_ \
    and implemented by us.

    """
    def __init__(self, device, task_num):
        super(RLW, self).__init__(device, task_num)
        
    def backward(self, losses, **kwargs):
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        # print("losses_device: ", losses.device)
        # print("batch_weight_device: ", batch_weight.device)
        loss = torch.mul(losses, batch_weight).sum()
        return loss, batch_weight.detach().cpu().numpy()
    
class UW(AbsWeighting):
    r"""Uncertainty Weights (UW).
    
    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_ \
    and implemented by us. 

    """
    def __init__(self, device, task_num):
        super(UW, self).__init__(device, task_num)
        
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        loss = (losses/(2*self.loss_scale.exp())+self.loss_scale/2).sum()
        # loss.backward()
        return loss, (1/(2*torch.exp(self.loss_scale))).detach().cpu().numpy()

class DWA(AbsWeighting):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    Args:
        T (float, default=2.0): The softmax temperature.

    """
    def __init__(self, device, task_num):
        super(DWA, self).__init__(device, task_num)
        
    def backward(self, losses, **kwargs):
        T = kwargs['T']
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:,self.epoch-1]/self.train_loss_buffer[:,self.epoch-2]).to(self.device)
            batch_weight = self.task_num*F.softmax(w_i/T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        # loss.backward()
        return loss, batch_weight.detach().cpu().numpy()
    
class CMC:
    def __init__(self, device, task_num):
        self.device = device
        self.task_num = task_num
        self.grad_index = []
        self.grad_dim = 0

    def init_param(self):
        self.weight_ce = 2
        self.weight_cmc = 1

    def backward(self, losses, **kwargs):
        loss = losses[0] * self.weight_ce + losses[1] * self.weight_cmc
        return loss, [self.weight_ce, self.weight_cmc]

        
def Weighting_Adapt(method_name, device, task_num):
    """
    Returns an instance of the loss weighting method class based on the provided method name.

    Args:
    method_name (str): The name of the loss weighting method class.

    Returns:
    An instance of the specified loss weighting method class.
    """
    method_classes = {
        "EW": EW,
        "GLS": GLS,
        "RLW": RLW,
        "UW": UW,
        "DWA": DWA,
        "CMC": CMC
    }

    if method_name in method_classes:
        return method_classes[method_name](device, task_num)
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    







