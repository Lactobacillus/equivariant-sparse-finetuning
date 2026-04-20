import os
import sys
import math
from typing import List, Dict, Tuple, Set, Union, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn.util import prod
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct, Irreps


class EquiSparseDeltaSTR(nn.Module):

    def __init__(self,
            model: nn.Module,
            init_threshold: float = 1e-4,
            per_instruction: bool = False,
            target: List[str] = ['linear', 'tp', 'fctp']) -> None:

        super(EquiSparseDeltaSTR, self).__init__()

        self.model = model

        self.init_threshold = init_threshold
        self.per_instruction = per_instruction
        self.target = target

        self._wrap_layers(model)

    def _wrap_layers(self,
            module: nn.Module) -> None:

        for name, child in list(module.named_children()):

            if type(child) in {LinearSparseDeltaSTR, TensorProductSparseDeltaSTR, FullyConnectedTensorProductSparseDeltaSTR}:

                continue

            # skip already wrapped layers
            if any(key in name for key in {'adapter', 'lora', 'delta', 'spin', 'mag'}):

                continue

            # do not use 'isinstasnce' here
            # inheritance causes unintended behaviors
            if type(child) is Linear and 'linear' in self.target and getattr(child, 'internal_weights', False):

                wrapped_layer = LinearSparseDeltaSTR(child, self.init_threshold, self.per_instruction)
                setattr(module, name, wrapped_layer)
                print('[info] SparseDeltaSTR-Linear: {} (init_threshold: {}, per_instruction: {})'.format(name, self.init_threshold, self.per_instruction))

            elif type(child) is TensorProduct and 'tp' in self.target and getattr(child, 'internal_weights', False):

                wrapped_layer = TensorProductSparseDeltaSTR(child, self.init_threshold, self.per_instruction)
                setattr(module, name, wrapped_layer)
                print('[info] SparseDeltaSTR-TP: {} (init_threshold: {}, per_instruction: {})'.format(name, self.init_threshold, self.per_instruction))

            elif type(child) is FullyConnectedTensorProduct and 'fctp' in self.target and getattr(child, 'internal_weights', False):

                wrapped_layer = FullyConnectedTensorProductSparseDeltaSTR(child, self.init_threshold, self.per_instruction)
                setattr(module, name, wrapped_layer)
                print('[info] SparseDeltaSTR-FCTP: {} (init_threshold: {}, per_instruction: {})'.format(name, self.init_threshold, self.per_instruction))

            else:

                self._wrap_layers(child)

    def forward(self,
            *args,
            **kwargs) -> Any:

        return self.model(*args, **kwargs)

    def get_sparsity_reg(self) -> torch.Tensor:

        reg = 0

        for module in self.model.modules():

            if hasattr(module, 'get_sparsity_reg'):

                reg = reg + module.get_sparsity_reg()

        return reg

    def prune(self,
            threshold: float = 0) -> None:

        # threshold does not affect pruning for STR
        for module in self.model.modules():

            if hasattr(module, 'prune'):

                module.prune()

    def get_sparsity(self) -> float:

        cnt_zero = 0
        cnt_nonzero = 0

        for module in self.model.modules():

            if hasattr(module, 'count_zero_weight'):

                cnt_zero = cnt_zero + module.count_zero_weight()

            if hasattr(module, 'count_nonzero_weight'):

                cnt_nonzero = cnt_nonzero + module.count_nonzero_weight()

        cnt_all = cnt_zero + cnt_nonzero

        if cnt_all == 0:

            sparsity = 0

        else:

            sparsity = cnt_zero / cnt_all

        return sparsity

    def get_mean_score(self) -> float:

        score = list()

        for module in self.model.modules():

            if hasattr(module, 'get_score'):

                if self.per_instruction:

                    score.extend(module.get_score().detach().cpu().tolist())

                else:

                    score.append(module.get_score().item())

        mean_score = sum(score) / len(score)

        return mean_score

    def get_mean_threshold(self) -> float:

        threshold = list()

        for module in self.model.modules():

            if hasattr(module, 'get_threshold'):

                if self.per_instruction:

                    threshold.extend(module.get_threshold().detach().cpu().tolist())

                else:

                    threshold.append(module.get_threshold().item())

        mean_threshold = sum(threshold) / len(threshold)

        return mean_threshold

    def count_zero_weight(self) -> int:

        cnt = 0

        for module in self.model.modules():

            if hasattr(module, 'count_zero_weight'):

                cnt = cnt + module.count_zero_weight()

        return cnt

    def count_nonzero_weight(self) -> int:

        cnt = 0

        for module in self.model.modules():

            if hasattr(module, 'count_nonzero_weight'):

                cnt = cnt + module.count_nonzero_weight()

        return cnt

    def merge(self) -> None:

        for module in self.model.modules():

            if hasattr(module, 'merge'):

                module.merge()

    def unmerge(self) -> None:

        for module in self.model.modules():

            if hasattr(module, 'merge'):

                module.unmerge()


class LinearSparseDeltaSTR(Linear):

    def __init__(self,
            original_layer: Linear,
            init_threshold: float = 1e-4,
            per_instruction: bool = False) -> None:

        super(LinearSparseDeltaSTR, self).__init__(
            irreps_in = original_layer.irreps_in,
            irreps_out = original_layer.irreps_out,
            internal_weights = False,
            shared_weights = getattr(original_layer, 'shared_weights', False),
            instructions = [(ins.i_in, ins.i_out) for ins in original_layer.instructions if ins.i_in != -1],
            biases = False,
            path_normalization = getattr(original_layer, 'path_normalization', 'element'),
            _optimize_einsums = getattr(original_layer, '_optimize_einsums', None))

        self.original_layer = original_layer

        self.per_instruction = per_instruction

        # find s, where sigmoid(s) = init_threshold
        # torch.special.logit: inv. function of sigmoid
        score = torch.special.logit(torch.tensor(init_threshold), eps = 1e-12) # hard-coded epsilon

        if per_instruction:

            # multiple threshold, per instruction
            num_threshold = sum(1 for ins in self.instructions if ins.i_in != -1)
            self.score = nn.Parameter(torch.full((num_threshold, ), score))

        else:

            # single threshold
            self.score = nn.Parameter(score)

        for p in self.original_layer.parameters():

            p.requires_grad = False 

        self.delta_weight = nn.Parameter(torch.randn(self.weight_numel) * 1e-2)
        self.register_buffer('delta_weight_prune', torch.zeros(self.weight_numel))

        self.merged = False

    def get_delta_weight(self) -> torch.Tensor:

        return self.delta_weight

    def get_sparse_delta_weight(self) -> torch.Tensor:

        delta_weight = self.get_delta_weight()

        if self.per_instruction:

            sparse_delta_weight = torch.zeros_like(delta_weight)
            weight_idx = 0
            threshold_idx = 0

            for ins_i, ins in enumerate(self.instructions):

                if ins.i_in != -1:

                    in_dim, out_dim = ins.path_shape

                    start_idx = weight_idx
                    end_idx = start_idx + in_dim * out_dim

                    threshold = torch.sigmoid(self.score[threshold_idx])
                    delta_segment = delta_weight[start_idx:end_idx]
                    sparse_delta_segment = torch.sign(delta_segment) * F.relu(torch.abs(delta_segment) - threshold)
                    sparse_delta_weight[start_idx:end_idx] = sparse_delta_segment

                    weight_idx = end_idx
                    threshold_idx = threshold_idx + 1

        else:

            threshold = torch.sigmoid(self.score)
            sparse_delta_weight = torch.sign(delta_weight) * F.relu(torch.abs(delta_weight) - threshold)

        return sparse_delta_weight

    def get_delta_weight_prune(self) -> torch.Tensor:

        return self.delta_weight_prune

    def get_score(self) -> torch.Tensor:

        return self.score

    def get_threshold(self) -> torch.Tensor:

        return torch.sigmoid(self.score)

    def get_weight_views(self,
            merge: bool = False) -> Dict[str, Tuple[Irreps, Irreps, torch.Tensor]]:

        views = dict()
        weight_idx = 0

        if merge:

            self.merge()

        weight = self.original_layer.weight

        for ins_i, ins in enumerate(self.instructions):

            if ins.i_in != -1:

                in_dim, out_dim = ins.path_shape
                key = '{}_{}'.format(ins.i_in, ins.i_out)

                start_idx = weight_idx
                end_idx = start_idx + in_dim * out_dim
                delta = weight[start_idx:end_idx].view(out_dim, in_dim)

                views[key] = (self.irreps_in[ins.i_in], self.irreps_out[ins.i_out], delta.detach().clone())

                weight_idx = end_idx

        if merge:

            self.unmerge()

        return views

    def get_delta_weight_views(self) -> Dict[str, Tuple[Irreps, Irreps, torch.Tensor]]:

        views = dict()
        weight_idx = 0

        delta_weight = self.get_delta_weight()

        for ins_i, ins in enumerate(self.instructions):

            if ins.i_in != -1:

                in_dim, out_dim = ins.path_shape
                key = '{}_{}'.format(ins.i_in, ins.i_out)

                start_idx = weight_idx
                end_idx = start_idx + in_dim * out_dim
                delta = delta_weight[start_idx:end_idx].view(out_dim, in_dim)

                views[key] = (self.irreps_in[ins.i_in], self.irreps_out[ins.i_out], delta)

                weight_idx = end_idx

        return views

    def get_delta_weight_prune_views(self) -> Dict[str, Tuple[Irreps, Irreps, torch.Tensor]]:

        views = dict()
        weight_idx = 0

        delta_weight_prune = self.get_delta_weight_prune()

        for ins_i, ins in enumerate(self.instructions):

            if ins.i_in != -1:

                in_dim, out_dim = ins.path_shape
                key = '{}_{}'.format(ins.i_in, ins.i_out)

                start_idx = weight_idx
                end_idx = start_idx + in_dim * out_dim
                delta = delta_weight_prune[start_idx:end_idx].view(out_dim, in_dim)

                views[key] = (self.irreps_in[ins.i_in], self.irreps_out[ins.i_out], delta)

                weight_idx = end_idx

        return views

    def forward(self,
            x: torch.Tensor) -> torch.Tensor:

        original_out = self.original_layer(x)

        if self.merged:

            return original_out

        delta_out = super().forward(x, weight = self.get_sparse_delta_weight())

        out = original_out + delta_out

        return out

    def merge(self) -> None:

        if self.merged:

            return

        with torch.no_grad():

            delta_weight = self.get_delta_weight_prune()
            self.original_layer.weight.add_(delta_weight)

        self.merged = True

    def unmerge(self) -> None:

        if not self.merged:

            return

        with torch.no_grad():

            delta_weight = self.get_delta_weight_prune()
            self.original_layer.weight.sub_(delta_weight)

        self.merged = False

    def train(self,
            mode: bool = True):

        self.original_layer.train(False)

        # if mode:

        #     self.unmerge()

        # else:

        #     self.merge()

        return super().train(mode)

    def eval(self):

        self.original_layer.eval()
        # self.merge()

        return super().eval()

    def get_sparsity_reg(self) -> torch.Tensor:

        reg = torch.sum(torch.abs(self.score))

        return reg

    @torch.no_grad()
    def prune(self) -> None:

        sparse_delta_weight = self.get_sparse_delta_weight()
        self.delta_weight_prune.copy_(sparse_delta_weight)

    @torch.no_grad()
    def count_zero_weight(self) -> int:

        sparse_delta_weight = self.get_sparse_delta_weight()
        cnt = sparse_delta_weight.numel() - torch.count_nonzero(sparse_delta_weight).item()

        return cnt

    @torch.no_grad()
    def count_nonzero_weight(self) -> int:

        sparse_delta_weight = self.get_sparse_delta_weight()
        cnt = torch.count_nonzero(sparse_delta_weight).item()

        return cnt


class TensorProductSparseDeltaSTR(TensorProduct):

    def __init__(self,
            original_layer: TensorProduct,
            init_threshold: float = 1e-4,
            per_instruction: bool = False) -> None:

        super(TensorProductSparseDeltaSTR, self).__init__(
            irreps_in1 = original_layer.irreps_in1,
            irreps_in2 = original_layer.irreps_in2,
            irreps_out = original_layer.irreps_out,
            instructions = [(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, getattr(ins, 'path_weight', None)) for ins in original_layer.instructions],
            internal_weights = False,
            shared_weights = getattr(original_layer, 'shared_weights', False),
            compile_left_right = True,
            compile_right = getattr(original_layer, '_did_compile_right', False),
            _specialized_code = getattr(original_layer, '_specialized_code', None),
            _optimize_einsums = getattr(original_layer, '_optimize_einsums', None))

        self.original_layer = original_layer

        self.per_instruction = per_instruction

        # find s, where sigmoid(s) = init_threshold
        # torch.special.logit: inv. function of sigmoid
        score = torch.special.logit(torch.tensor(init_threshold), eps = 1e-12) # hard-coded epsilon

        if per_instruction:

            # multiple threshold, per instruction
            num_threshold = sum(1 for ins in self.original_layer.instructions if ins.has_weight)
            self.score = nn.Parameter(torch.full((num_threshold, ), score))

        else:

            # single threshold
            self.score = nn.Parameter(score)

        for p in self.original_layer.parameters():

            p.requires_grad = False 

        self.delta_weight = nn.Parameter(torch.randn(self.weight_numel) * 1e-2)
        self.register_buffer('delta_weight_prune', torch.zeros(self.weight_numel))

        self.merged = False

    def get_delta_weight(self) -> torch.Tensor:

        return self.delta_weight

    def get_sparse_delta_weight(self) -> torch.Tensor:

        delta_weight = self.get_delta_weight()

        if self.per_instruction:

            sparse_delta_weight = torch.zeros_like(delta_weight)
            cursor = 0
            threshold_idx = 0

            for ins_i, ins in enumerate(self.original_layer.instructions):

                if ins.has_weight:

                    flat_dim = torch.tensor(ins.path_shape).prod().item()

                    threshold = torch.sigmoid(self.score[threshold_idx])
                    delta_segment = delta_weight.narrow(-1, cursor, flat_dim)
                    sparse_delta_segment = torch.sign(delta_segment) * F.relu(torch.abs(delta_segment) - threshold)
                    sparse_delta_weight.narrow(-1, cursor, flat_dim).copy_(sparse_delta_segment)

                    cursor = cursor + flat_dim
                    threshold_idx = threshold_idx + 1

        else:

            threshold = torch.sigmoid(self.score)
            sparse_delta_weight = torch.sign(delta_weight) * F.relu(torch.abs(delta_weight) - threshold)

        return sparse_delta_weight

    def get_delta_weight_prune(self) -> torch.Tensor:

        return self.delta_weight_prune

    def get_score(self) -> torch.Tensor:

        return self.score

    def get_threshold(self) -> torch.Tensor:

        return torch.sigmoid(self.score)

    def get_weight_views(self,
            merge: bool = False) -> Dict[str, Tuple[Irreps, Irreps, Irreps, torch.Tensor]]:

        views = dict()
        cursor = 0

        if merge:

            self.merge()

        weight = self.original_layer.weight

        for ins_i, ins in enumerate(self.original_layer.instructions):

            if ins.has_weight:

                flat_dim = torch.tensor(ins.path_shape).prod().item()
                key = '{}_{}_{}'.format(ins.i_in1, ins.i_in2, ins.i_out)

                delta = weight.narrow(-1, cursor, flat_dim).view(ins.path_shape)
                views[key] = (self.irreps_in1[ins.i_in1], self.irreps_in2[ins.i_in2], self.irreps_out[ins.i_out], delta.detach().clone())

                cursor = cursor + flat_dim

        if merge:

            self.unmerge()

        return views

    def get_delta_weight_views(self) -> Dict[str, Tuple[Irreps, Irreps, Irreps, torch.Tensor]]:

        views = dict()
        cursor = 0

        delta_weight = self.get_delta_weight()

        for ins_i, ins in enumerate(self.original_layer.instructions):

            if ins.has_weight:

                flat_dim = torch.tensor(ins.path_shape).prod().item()
                key = '{}_{}_{}'.format(ins.i_in1, ins.i_in2, ins.i_out)

                delta = delta_weight.narrow(-1, cursor, flat_dim).view(ins.path_shape)
                views[key] = (self.irreps_in1[ins.i_in1], self.irreps_in2[ins.i_in2], self.irreps_out[ins.i_out], delta)

                cursor = cursor + flat_dim

        return views

    def get_delta_weight_prune_views(self) -> Dict[str, Tuple[Irreps, Irreps, Irreps, torch.Tensor]]:

        views = dict()
        cursor = 0

        delta_weight_prune = self.get_delta_weight_prune()

        for ins_i, ins in enumerate(self.original_layer.instructions):

            if ins.has_weight:

                flat_dim = torch.tensor(ins.path_shape).prod().item()
                key = '{}_{}_{}'.format(ins.i_in1, ins.i_in2, ins.i_out)

                delta = delta_weight_prune.narrow(-1, cursor, flat_dim).view(ins.path_shape)
                views[key] = (self.irreps_in1[ins.i_in1], self.irreps_in2[ins.i_in2], self.irreps_out[ins.i_out], delta)

                cursor = cursor + flat_dim

        return views

    def forward(self,
            x: torch.Tensor,
            y: torch.Tensor) -> torch.Tensor:

        original_out = self.original_layer(x, y)

        if self.merged:

            return original_out

        delta_out = super().forward(x, y, weight = self.get_sparse_delta_weight())

        out = original_out + delta_out

        return out

    def merge(self) -> None:

        if self.merged:

            return

        with torch.no_grad():

            delta_weight = self.get_delta_weight_prune()
            self.original_layer.weight.add_(delta_weight)

        self.merged = True

    def unmerge(self) -> None:

        if not self.merged:

            return

        with torch.no_grad():

            delta_weight = self.get_delta_weight_prune()
            self.original_layer.weight.sub_(delta_weight)

        self.merged = False

    def train(self,
            mode: bool = True):

        self.original_layer.train(False)

        # if mode:

        #     self.unmerge()

        # else:

        #     self.merge()

        return super().train(mode)

    def eval(self):

        self.original_layer.eval()
        # self.merge()

        return super().eval()

    def get_sparsity_reg(self) -> torch.Tensor:

        reg = torch.sum(torch.abs(self.score))

        return reg

    @torch.no_grad()
    def prune(self) -> None:

        sparse_delta_weight = self.get_sparse_delta_weight()
        self.delta_weight_prune.copy_(sparse_delta_weight)

    @torch.no_grad()
    def count_zero_weight(self) -> int:

        sparse_delta_weight = self.get_sparse_delta_weight()
        cnt = sparse_delta_weight.numel() - torch.count_nonzero(sparse_delta_weight).item()

        return cnt

    @torch.no_grad()
    def count_nonzero_weight(self) -> int:

        sparse_delta_weight = self.get_sparse_delta_weight()
        cnt = torch.count_nonzero(sparse_delta_weight).item()

        return cnt


class FullyConnectedTensorProductSparseDeltaSTR(FullyConnectedTensorProduct):

    def __init__(self,
            original_layer: FullyConnectedTensorProduct,
            init_threshold: float = 1e-4,
            per_instruction: bool = False) -> None:

        super(FullyConnectedTensorProductSparseDeltaSTR, self).__init__(
            irreps_in1 = original_layer.irreps_in1,
            irreps_in2 = original_layer.irreps_in2,
            irreps_out = original_layer.irreps_out,
            irrep_normalization = getattr(original_layer, 'irrep_normalization', 'component'),
            path_normalization = getattr(original_layer, 'path_normalization', 'element'),
            internal_weights = False,
            shared_weights = getattr(original_layer, 'shared_weights', False),
            compile_left_right = True,
            compile_right = getattr(original_layer, '_did_compile_right', False),
            _specialized_code = getattr(original_layer, '_specialized_code', None),
            _optimize_einsums = getattr(original_layer, '_optimize_einsums', None))

        self.original_layer = original_layer

        self.per_instruction = per_instruction

        # find s, where sigmoid(s) = init_threshold
        # torch.special.logit: inv. function of sigmoid
        score = torch.special.logit(torch.tensor(init_threshold), eps = 1e-12) # hard-coded epsilon

        if per_instruction:

            # multiple threshold, per instruction
            num_threshold = sum(1 for ins in self.original_layer.instructions if ins.has_weight)
            self.score = nn.Parameter(torch.full((num_threshold, ), score))

        else:

            # single threshold
            self.score = nn.Parameter(score)

        for p in self.original_layer.parameters():

            p.requires_grad = False

        self.delta_weight = nn.Parameter(torch.randn(self.weight_numel) * 1e-2)
        self.register_buffer('delta_weight_prune', torch.zeros(self.weight_numel))

        self.merged = False

    def get_delta_weight(self) -> torch.Tensor:

        return self.delta_weight

    def get_sparse_delta_weight(self) -> torch.Tensor:

        delta_weight = self.get_delta_weight()

        if self.per_instruction:

            sparse_delta_weight = torch.zeros_like(delta_weight)
            cursor = 0
            threshold_idx = 0

            for ins_i, ins in enumerate(self.original_layer.instructions):

                if ins.has_weight:

                    flat_dim = torch.tensor(ins.path_shape).prod().item()

                    threshold = torch.sigmoid(self.score[threshold_idx])
                    delta_segment = delta_weight.narrow(-1, cursor, flat_dim)
                    sparse_delta_segment = torch.sign(delta_segment) * F.relu(torch.abs(delta_segment) - threshold)
                    sparse_delta_weight.narrow(-1, cursor, flat_dim).copy_(sparse_delta_segment)

                    cursor = cursor + flat_dim
                    threshold_idx = threshold_idx + 1

        else:

            threshold = torch.sigmoid(self.score)
            sparse_delta_weight = torch.sign(delta_weight) * F.relu(torch.abs(delta_weight) - threshold)

        return sparse_delta_weight

    def get_delta_weight_prune(self) -> torch.Tensor:

        return self.delta_weight_prune

    def get_score(self) -> torch.Tensor:

        return self.score

    def get_threshold(self) -> torch.Tensor:

        return torch.sigmoid(self.score)

    def get_weight_views(self,
            merge: bool = False) -> Dict[str, Tuple[Irreps, Irreps, Irreps, torch.Tensor]]:

        views = dict()
        cursor = 0

        if merge:

            self.merge()

        weight = self.original_layer.weight

        for ins_i, ins in enumerate(self.original_layer.instructions):

            if ins.has_weight:

                flat_dim = torch.tensor(ins.path_shape).prod().item()
                key = '{}_{}_{}'.format(ins.i_in1, ins.i_in2, ins.i_out)

                delta = weight.narrow(-1, cursor, flat_dim).view(ins.path_shape)
                views[key] = (self.irreps_in1[ins.i_in1], self.irreps_in2[ins.i_in2], self.irreps_out[ins.i_out], delta.detach().clone())

                cursor = cursor + flat_dim

        if merge:

            self.unmerge()

        return views

    def get_delta_weight_views(self) -> Dict[str, Tuple[Irreps, Irreps, Irreps, torch.Tensor]]:

        views = dict()
        cursor = 0

        delta_weight = self.get_delta_weight()

        for ins_i, ins in enumerate(self.original_layer.instructions):

            if ins.has_weight:

                flat_dim = torch.tensor(ins.path_shape).prod().item()
                key = '{}_{}_{}'.format(ins.i_in1, ins.i_in2, ins.i_out)

                delta = delta_weight.narrow(-1, cursor, flat_dim).view(ins.path_shape)
                views[key] = (self.irreps_in1[ins.i_in1], self.irreps_in2[ins.i_in2], self.irreps_out[ins.i_out], delta)

                cursor = cursor + flat_dim

        return views

    def get_delta_weight_prune_views(self) -> Dict[str, Tuple[Irreps, Irreps, Irreps, torch.Tensor]]:

        views = dict()
        cursor = 0

        delta_weight_prune = self.get_delta_weight_prune()

        for ins_i, ins in enumerate(self.original_layer.instructions):

            if ins.has_weight:

                flat_dim = torch.tensor(ins.path_shape).prod().item()
                key = '{}_{}_{}'.format(ins.i_in1, ins.i_in2, ins.i_out)

                delta = delta_weight_prune.narrow(-1, cursor, flat_dim).view(ins.path_shape)
                views[key] = (self.irreps_in1[ins.i_in1], self.irreps_in2[ins.i_in2], self.irreps_out[ins.i_out], delta)

                cursor = cursor + flat_dim

        return views

    def forward(self,
            x: torch.Tensor,
            y: torch.Tensor) -> torch.Tensor:

        original_out = self.original_layer(x, y)

        if self.merged:

            return original_out

        delta_out = super().forward(x, y, weight = self.get_sparse_delta_weight())

        out = original_out + delta_out

        return out

    def merge(self) -> None:

        if self.merged:

            return

        with torch.no_grad():

            delta_weight = self.get_delta_weight_prune()
            self.original_layer.weight.add_(delta_weight)

        self.merged = True

    def unmerge(self) -> None:

        if not self.merged:

            return

        with torch.no_grad():

            delta_weight = self.get_delta_weight_prune()
            self.original_layer.weight.sub_(delta_weight)

        self.merged = False

    def train(self,
            mode: bool = True):

        self.original_layer.train(False)

        # if mode:

        #     self.unmerge()

        # else:

        #     self.merge()

        return super().train(mode)

    def eval(self):

        self.original_layer.eval()
        # self.merge()

        return super().eval()

    def get_sparsity_reg(self) -> torch.Tensor:

        reg = torch.sum(torch.abs(self.score))

        return reg

    @torch.no_grad()
    def prune(self) -> None:

        sparse_delta_weight = self.get_sparse_delta_weight()
        self.delta_weight_prune.copy_(sparse_delta_weight)

    @torch.no_grad()
    def count_zero_weight(self) -> int:

        sparse_delta_weight = self.get_sparse_delta_weight()
        cnt = sparse_delta_weight.numel() - torch.count_nonzero(sparse_delta_weight).item()

        return cnt

    @torch.no_grad()
    def count_nonzero_weight(self) -> int:

        sparse_delta_weight = self.get_sparse_delta_weight()
        cnt = torch.count_nonzero(sparse_delta_weight).item()

        return cnt
