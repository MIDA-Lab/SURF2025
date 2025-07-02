# RayS_Single.py
import time
import numpy as np
import torch

class RayS(object):
    def __init__(self, model, ds_mean, ds_std, order=np.inf, epsilon=0.3, early_stopping=True):
        self.model = model
        self.order = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}
        self.early_stopping = early_stopping

        # Add these two lines:
        self.ds_mean = torch.tensor(ds_mean).view(1, -1, 1, 1)
        self.ds_std = torch.tensor(ds_std).view(1, -1, 1, 1)

    def get_xadv(self, x, v, d, lb=0., rb=1.):
        out = x + d * v
        return torch.clamp(out, lb, rb)

    def attack_hard_label(self, x, y, target=None, query_limit=10000, seed=None):
        # ====== Add this line for reverse standardization ======
        x = (x * self.ds_std.to(x.device)) + self.ds_mean.to(x.device)
        # now x_i should be \in  [0, 1] \forall i as RayS expects. 

        shape = list(x.shape)
        dim = int(np.prod(shape[1:]))
        if seed is not None:
            np.random.seed(seed)
        self.queries = 0
        self.d_t = np.inf
        self.sgn_t = torch.sign(torch.ones(shape, device=x.device))
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
        dist = torch.tensor(np.inf, device=x.device)
        block_level = 0
        block_ind = 0

        for i in range(query_limit):
            block_num  = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start = block_ind * block_size
            end   = min(dim, (block_ind + 1) * block_size)

            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm(self.x_final - x, self.order)
            if self.early_stopping and (dist <= self.epsilon):
                break
            if self.queries >= query_limit:
                break
            if i % 10 == 0:
                print(f"Iter {i+1:3d} d_t {self.d_t:.6f} dist {dist:.6f} queries {self.queries}")

        print(f"Done – d_t {self.d_t:.6f} dist {dist:.6f} queries {self.queries}")
        return self.x_final, self.queries, dist, (dist <= self.epsilon).float()

    def search_succ(self, x, y, target):
        self.queries += 1
        if target is not None:
            return self.model.predict_label(x) == target
        else:
            return self.model.predict_label(x) != y

    def lin_search(self, x, y, target, sgn):
        for d in range(1, self.lin_search_rad + 1):
            if self.search_succ(self.get_xadv(x, sgn, d), y, target):
                return d
        return np.inf

    def binary_search(self, x, y, target, sgn, tol=1e-3):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)
        d_start = 0

        if self.d_t < np.inf:
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                return False
            d_end = self.d_t
        else:
            d_lin = self.lin_search(x, y, target, sgn)
            if d_lin == np.inf:
                return False
            d_end = d_lin * sgn_norm

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            if self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target):
                d_end = d_mid
            else:
                d_start = d_mid

        if d_end < self.d_t:
            self.d_t     = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t   = sgn
            return True
        return False

    def __call__(self, data, label, target=None, seed=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, seed=seed, query_limit=query_limit)

