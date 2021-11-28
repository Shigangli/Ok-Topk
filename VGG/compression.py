# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import time

import math
import utils
from scipy import stats

class NoneCompressor():
    name = 'dense'
    @staticmethod
    def compress(tensor, name=None, sigma_scale=None, ratio=None):
        return tensor, None
        #return tensor, tensor.dtype

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    residuals = {}
    c = 0
    sparsities = []
    t = 0.
    zero_conditions = {}
    values = {} 
    indexes = {} 
    name = 'topk'
    @staticmethod
    def compress_org(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]
            if name not in TopKCompressor.zero_conditions:
                TopKCompressor.zero_conditions[name] = torch.ones(numel, dtype=torch.float32, device=tensor.device) 
            zero_condition = TopKCompressor.zero_conditions[name]
            zero_condition.fill_(1.0)
            zero_condition[indexes] = 0.0

            TopKCompressor.residuals[name].data.fill_(0.)
            TopKCompressor.residuals[name].data = tensor.data * zero_condition
            tensor.data.sub_(TopKCompressor.residuals[name].data)

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes

    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k, sorted=False)

            TopKCompressor.residuals[name].data = tensor.data + 0.0
            TopKCompressor.residuals[name].data[indexes] = 0.0

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes 

    @staticmethod
    def ratio2threshold(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(TopKCompressor.residuals[name].data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)

            TopKCompressor.residuals[name].data = tensor.data + 0.0
            TopKCompressor.residuals[name].data[indexes] = 0.0

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            #print("local topk elements: ", torch.numel(values))

            threshold = float(values[values.numel()-1].item())
            return threshold

    @staticmethod
    def ratio2globalthreshold(tensor, ratio=0.05):
        with torch.no_grad():
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)

            threshold = float(values[values.numel()-1].item())
            print("global topk elements: ", torch.numel(values), "threshold: ", threshold)
            return threshold

    @staticmethod
    def compressbythreshold(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    @staticmethod
    def compressbythresholdlong(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            return indexes, values

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return TopKCompressor.residuals[name]

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = TopKCompressor.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).cuda(residuals.device).long()
            else:
                indexes_t = included_indexes
            values = TopKCompressor.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[TopKCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 

class GaussianCompressor():
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {} 
    indexes = {} 
    c = 0
    t = 0.
    name = 'gaussionk'
    #inc_factor = 0.02
    #dec_factor = 1.8

    counter = 0
    local_threshold = 0.0

    @staticmethod
    def clear():
        GaussianCompressor.residuals = {}
        GaussianCompressor.sparsities = []
        GaussianCompressor.zero_conditions = {}
        GaussianCompressor.values = {} 
        GaussianCompressor.indexes = {}

    #@staticmethod
    #def compress(tensor, name=None, ratio=0.05, counter=-1, rank=-1):
    #    with torch.no_grad():
    #        if name not in GaussianCompressor.residuals:
    #            GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
    #        numel = tensor.numel()
    #        k = max(int(numel * ratio), 1)

    #        tensor.add_(GaussianCompressor.residuals[name].data)

    #        std = torch.std(tensor)
    #        mean = torch.mean(tensor)
    #        left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
    #        abs_tensor = torch.abs(tensor)
    #        one_indexes = abs_tensor > right_thres
    #        indexes = one_indexes.nonzero().data.squeeze().view(-1)

    #        #one_indexes = abs_tensor > right_thres
    #        #indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #        #indexes = indexes #[0:k]
    #        values = tensor.data[indexes] 
    #        GaussianCompressor.residuals[name].data = tensor.data + 0.0 
    #        GaussianCompressor.residuals[name].data[indexes] = 0.0

    #        indexes = indexes.type(torch.IntTensor)
    #        return indexes, values

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, counter=-1, rank=-1):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianCompressor.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > right_thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            init_num = indexes.numel()

            if init_num < 3*k//4:
                loops = 0
                while loops < 20:
                    if indexes.numel() < 3*k//4:
                        right_thres /= 1.02
                    else:
                        break
                    one_indexes = abs_tensor > right_thres
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    loops += 1
            elif init_num > 5*k//4:
                loops = 0
                while loops < 20:
                    if indexes.numel() > 5*k//4:
                        right_thres *= 1.02
                    else:
                        break
                    one_indexes = abs_tensor > right_thres
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    loops += 1
            else:
                pass

            values = tensor.data[indexes] 
            GaussianCompressor.residuals[name].data = tensor.data + 0.0 
            GaussianCompressor.residuals[name].data[indexes] = 0.0

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    #@staticmethod
    #def compress(tensor, name=None, ratio=0.05, counter=-1, rank=-1):
    #    with torch.no_grad():
    #        if name not in GaussianCompressor.residuals:
    #            GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
    #        numel = tensor.numel()
    #        k = max(int(numel * ratio), 1)

    #        tensor.add_(GaussianCompressor.residuals[name].data)

    #        std = torch.std(tensor)
    #        mean = torch.mean(tensor)
    #        left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
    #        abs_tensor = torch.abs(tensor)
    #        loops = 0
    #        while loops < 3:
    #            one_indexes = abs_tensor > right_thres
    #            indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #            if indexes.numel() < 2*k/3:
    #                right_thres *= 0.5
    #            elif indexes.numel() > 4*k/3:
    #                right_thres *= 1.5
    #            else:
    #                break
    #            loops += 1

    #        #print("local mean: ", mean, "local std: ", std, "adapt loops: ", loops)
    #        #one_indexes = abs_tensor > right_thres
    #        #indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #        #indexes = indexes #[0:k]
    #        values = tensor.data[indexes] 
    #        #print('gaussion vs topk: ', indexes.numel(), k)
    #        GaussianCompressor.residuals[name].data = tensor.data + 0.0 
    #        GaussianCompressor.residuals[name].data[indexes] = 0.0

    #        indexes = indexes.type(torch.IntTensor)
    #        return indexes, values

    @staticmethod
    def predictratio2threshold(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            one_indexes = abs_tensor > right_thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            pre_topk = indexes.numel()

            return right_thres, pre_topk


    @staticmethod
    def compressbythreshold(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    @staticmethod
    def compressbythreshold_residual(tensor, name, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)

            GaussianCompressor.residuals[name].data[indexes] = 0.0
            values = tensor.data[indexes]

            indexes = indexes.type(torch.IntTensor)
            return indexes, values

    @staticmethod
    def compressbythresholdlong(tensor, thres=0.0):
        with torch.no_grad():
            abs_tensor = torch.abs(tensor)

            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            return indexes

    #@staticmethod
    #def compressbythresholdlong(tensor, thres=0.0):
    #    with torch.no_grad():
    #        abs_tensor = torch.abs(tensor)

    #        one_indexes = abs_tensor > thres
    #        indexes = one_indexes.nonzero().data.squeeze().view(-1)
    #        values = tensor.data[indexes]

    #        return indexes, values

    @staticmethod
    def ratio2threshold(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.data.add_(GaussianCompressor.residuals[name].data)
            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            GaussianCompressor.residuals[name].data = tensor.data + 0.0

        return float(values[values.numel()-1].item())

    @staticmethod
    def add2residual(tensor=None, name=None, thrd=None, tk=None):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)

            tensor.data.add_(GaussianCompressor.residuals[name].data)
            GaussianCompressor.residuals[name].data = tensor.data + 0.0

            abs_tensor = torch.abs(tensor)
            loops = 0
            thres = thrd
            while loops < 5:
                one_indexes = abs_tensor > thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() > 4*tk//3:
                    thres *= 1.03
                else:
                    break
                loops += 1

            return thres

    @staticmethod
    def k2globalthreshold(tensor, k=0):
        numel = tensor.numel()
        kk = min(numel, k)
        with torch.no_grad():
            values, indexes = torch.topk(torch.abs(tensor.data), k=kk)
            global_threshold = float(values[values.numel()-1].item())
            values = tensor[indexes]
            #indexes = indexes.type(torch.IntTensor)
        return values, indexes, global_threshold

    @staticmethod
    def ratio2thresholdresidual(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianCompressor.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 3:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            GaussianCompressor.residuals[name].data = tensor.data + 0.0 
            GaussianCompressor.residuals[name].data[indexes] = 0.0 
        return right_thres

    #@staticmethod
    #def globalratio2threshold(sparse_tensor, ratio=0.05, num_workers=1):
    #    with torch.no_grad():
    #        mean = torch.mean(sparse_tensor)*num_workers
    #        std = torch.std(sparse_tensor)*math.sqrt(num_workers)

    #        print("global mean: ", mean, "global std: ", std)
    #        left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
    #        return right_thres

    @staticmethod
    def globalratio2threshold(sparse_tensor, ratio=0.05, num_workers=1):
        with torch.no_grad():
            mean = torch.mean(sparse_tensor)*num_workers
            std = torch.std(sparse_tensor)*math.sqrt(num_workers)

            print("global mean: ", mean, "global std: ", std)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            return right_thres

    @staticmethod
    def update_residuals(involved_indexes, name):
        with torch.no_grad():
            #indexes_t = torch.from_numpy(involved_indexes).to(device=GaussianCompressor.residuals[name].device)
            indexes_t = torch.from_numpy(involved_indexes).to(device=GaussianCompressor.residuals[name].device).long()
            GaussianCompressor.residuals[name].data[indexes_t] = 0.0

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in GaussianCompressor.residuals:
            GaussianCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return GaussianCompressor.residuals[name]

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 

class TopKACompressor(TopKCompressor):
    name = 'topkA'

class TopKACompressor2(TopKCompressor):
    name = 'topkA2'

class TopKSACompressor(TopKCompressor):
    name = 'topkSA'

class gTopKCompressor(TopKCompressor):
    name = 'gtopk'

class GaussianKCompressor(GaussianCompressor):
    name = 'gaussiank'

class GaussianKCCCompressor(GaussianCompressor):
    name = 'gaussiankconcat'

class GaussianKSACompressor(GaussianCompressor):
    name = 'gaussiankSA'

class OKTopKCompressor(GaussianCompressor):
    name = 'oktopk'

class TopKAoptCompressor(GaussianCompressor):
    name = 'topkAopt'


compressors = {
        'topkA': TopKACompressor,
        'topkAopt': TopKAoptCompressor,
        'topkA2': TopKACompressor2,
        'topkSA': TopKSACompressor,
        'gtopk': gTopKCompressor,
        'gaussiank': GaussianKCompressor,
        'gaussiankconcat': GaussianKCCCompressor,
        'gaussiankSA': GaussianKSACompressor,
        'oktopk': OKTopKCompressor,
        'none': NoneCompressor
        }
