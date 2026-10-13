import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


def get_similarity(mk: torch.Tensor,
                   qk: torch.Tensor,
                   add_batch_dim: bool = False,
                   rank: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x h * w    - Memory keys
    # qk: B x CK x h * w - Query keys
    # Dimensions in [] are flattened
    # mk.shape:torch.Size([8, 384, 1, 32, 32])
    # qk.shape:torch.Size([8, 384, 32, 32])
    if add_batch_dim:
        mk = mk.unsqueeze(0)
        qk = qk.unsqueeze(0)

    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    # mk.shape:torch.Size([8, 384, 1024T])
    qk = qk.flatten(start_dim=2)
    # [8, 384, 1024]

    a_sq = mk.pow(2).sum(1).unsqueeze(2)
    two_ab = 2 * (mk.transpose(1, 2) @ qk)
    similarity = (-a_sq + two_ab)
    similarity = similarity / math.sqrt(CK)  # B*N*HW
    # (B, T×HW, HW)
    if not rank:
        return similarity
    else:
        B, T_HW, HW = similarity.shape  # mk: B x CK x (T * H * W)
        T = T_HW // (HW)
        # similarity: B x (T*HW) x HW
        rank_sim = similarity.view(B, T, HW, HW)  # B x T x HW x HW

        #
        frame_sim = rank_sim.mean(dim=(2, 3))  # B x T
        best_match_idx = frame_sim.argmax(dim=1)
        # similarity = torch.mean(similarity, dim=2)
        return similarity, best_match_idx


def do_softmax(
        similarity: torch.Tensor,
        top_k: Optional[int] = None,
        inplace: bool = False,
        return_usage: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
    else:
        # similarity (B, T×HW, HW)
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        # maxes[b, :, j] 表示：第 b 个 batch 中，query 的位置 j 在所有 memory 中最相似的那个值
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity


def get_affinity(mk: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
    # shorthand used in training with no top-k
    # mk.shape:torch.Size([8, 1, 384, 32, 32])
    # qk.shape:torch.Size([8, 384, 32, 32])
    mk = mk.transpose(1, 2)
    # mk.shape:torch.Size([8, 384, 1, 32, 32])
    similarity = get_similarity(mk, qk)
    affinity = do_softmax(similarity)
    return affinity


def readout(affinity: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    mv = mv.transpose(1, 2).contiguous()
    B, CV, R, H, W = mv.shape
    mo = mv.view(B, CV, R * H * W)
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)
    return mem
