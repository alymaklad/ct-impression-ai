# src/moco3d.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import resnet as monai_resnet

class MoCo3D(nn.Module):
    def __init__(self, backbone, feat_dim=128, K=8192, m=0.999, T=0.07, use_mlp=True):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # encoders
        self.encoder_q = backbone
        self.encoder_k = copy.deepcopy(backbone)

        # projectors
        if use_mlp:
            self.encoder_q_proj = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, feat_dim))
            self.encoder_k_proj = copy.deepcopy(self.encoder_q_proj)
        else:
            self.encoder_q_proj = nn.Linear(2048, feat_dim)
            self.encoder_k_proj = nn.Linear(2048, feat_dim)
        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # freeze key encoder params
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.encoder_k_proj.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def momentum_update_key(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.encoder_q_proj.parameters(), self.encoder_k_proj.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys) if torch.distributed.is_initialized() else keys
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (for simplicity assume K % batch_size == 0)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
        else:
            part = self.K - ptr
            self.queue[:, ptr:] = keys[:part].T
            self.queue[:, :batch_size - part] = keys[part:].T
            ptr = (batch_size - part) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # im_q/im_k shape: (B, C, D, H, W)
        qf = self.encoder_q(im_q)
        qf = qf.view(qf.size(0), -1)
        q = self.encoder_q_proj(qf)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self.momentum_update_key()
            kf = self.encoder_k(im_k)
            kf = kf.view(kf.size(0), -1)
            k = self.encoder_k_proj(kf)
            k = F.normalize(k, dim=1)

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # update queue
        self.dequeue_and_enqueue(k)
        return loss, logits


# utilities for DDP (copy from MoCo official repo)
import torch.distributed as dist

def concat_all_gather(tensor):
    """Gathers tensors from all processes, supporting autograd."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output