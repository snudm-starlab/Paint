import random
import time
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from ._utils import UnlabeledImageDataset, DataIter, ImagePool
from .base import BaseSynthesis
from .hooks import DeepInversionHook

def Gaussian_low_pass_filter(images, cutoff_ratio=0.8):
    B, C, H, W =images.shape

    fft_images = torch.fft.fftshift(
        torch.fft.fft2(images, dim=(-2, -1)),
        dim=(-2, -1)
    )

    Y, X = torch.meshgrid(
        torch.arange(H, device=images.device),
        torch.arange(W, device=images.device),
        indexing='ij'
    )

    freq_grid = torch.stack([Y - H // 2, X - W // 2], dim=-1).float()
    distance = torch.norm(freq_grid, dim=-1)

    max_dist = distance.max()
    sigma_f = cutoff_ratio * max_dist

    gaussian_mask = torch.exp(-(distance ** 2) / (2 * sigma_f ** 2))
    gaussian_mask = gaussian_mask[None, None, :, :] # broadcast

    filtered_fft = fft_images * gaussian_mask
    filtered = torch.fft.ifft2(
        torch.fft.ifftshift(filtered_fft, dim=(-2, -1)),
        dim=(-2, -1)
    ).real

    return filtered

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{:.2f}s'.format(x)

def clip_images(image_tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1,loss_var_l2

def jsdiv( logits, targets, T=1.0, reduction='batchmean' ):
    P = F.softmax(logits / T, dim=1)
    Q = F.softmax(targets / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction=reduction) + 0.5 * F.kl_div(torch.log(Q), M, reduction=reduction)

def jitter_and_flip(inputs_jit, lim=1./8., do_flip=True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)
    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
    # Flipping
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit,off1,off2,flip and do_flip

def jitter_and_flip_index(pre_index_matrix, off1, off2, flip, patch_size=16, num_patches_per_dim=14):
    off1_int, off1_frac = int(off1 // patch_size), off1 % patch_size / patch_size
    off2_int, off2_frac = int(off2 // patch_size), off2 % patch_size / patch_size
    patch_indices = torch.arange(1, num_patches_per_dim * num_patches_per_dim + 1).reshape(num_patches_per_dim, num_patches_per_dim).to(pre_index_matrix.device)
    patch_indices = torch.roll(patch_indices, shifts=(off1_int, off2_int), dims=(0, 1))
    if abs(off1_frac) >= 0.5:
        direction = 1 if off1_frac > 0 else -1
        patch_indices = torch.roll(patch_indices, shifts=(direction, 0), dims=(0, 1))
    if abs(off2_frac) >= 0.5:
        direction = 1 if off2_frac > 0 else -1
        patch_indices = torch.roll(patch_indices, shifts=(0, direction), dims=(0, 1))
    if flip:
        patch_indices = torch.flip(patch_indices, dims=[1])
    flat_patch_indices = patch_indices.flatten()
    non_zero_mask = pre_index_matrix != 0
    indices = (flat_patch_indices == pre_index_matrix[non_zero_mask].unsqueeze(-1)).nonzero(as_tuple=True)
    rows = indices[1] // num_patches_per_dim
    cols = indices[1] % num_patches_per_dim
    new_indices = rows * num_patches_per_dim + cols + 1
    new_index_matrix = torch.zeros_like(pre_index_matrix)
    new_index_matrix[non_zero_mask] = new_indices
    return new_index_matrix

class PAINT(BaseSynthesis):
    def __init__(
        self, teacher, teacher_name, student,
        num_classes, img_shape=(3, 224, 224),patch_size=16,
        iterations=2000, lr_g=0.25,
        synthesis_batch_size=128, sample_batch_size=128, 
        adv=0.0, bn=0, oh=1,tv1=0.0, tv2=1e-5, l2=0.0,
        save_dir='', transform=None,
        normalizer=None, device='cpu',
        bnsource='resnet50v2',init_dataset=None
    ):
        super(PAINT, self).__init__(teacher, student)
        assert len(img_shape)==3, "image size should be a 3-dimension tuple"

        self.save_dir = save_dir
        self.img_size = img_shape
        self.patch_size=patch_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        
        # Data pool
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset=init_dataset

        # Scaling factors
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.tv1 = tv1
        self.tv2 = tv2
        self.l2 = l2
        self.num_classes = num_classes

        if self.bn != 0:
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            if bnsource == 'resnet50v2':
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).cuda(
                    device)
                print(count_parameters(self.prior),'resnet50v2')
            elif bnsource == 'resnet50v1':
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda(
                    device)
                print(count_parameters(self.prior),'resnet50v1')
            else:
                raise NotImplementedError
            self.prior.eval()
            self.prior.cuda()

        # training configs
        self.device = device

        # Idea 2. Peripheral Sparsification (Sort patch indeices)
        self.num_patches_per_dim = int(self.img_size[1] // self.patch_size)
        Hp, Wp = self.num_patches_per_dim, self.num_patches_per_dim

        yy, xx = torch.meshgrid(
            torch.arange(Hp, device=self.device),
            torch.arange(Wp, device=self.device),
            indexing='ij'
        )
        cy, cx = (Hp - 1) / 2.0, (Wp - 1) / 2.0
        r2 = (yy.float() - cy)**2 + (xx.float() - cx)**2

        radial_order_flat = torch.argsort(r2.view(-1))
        self.patch_radial_order = radial_order_flat

        # setup hooks for BN regularization
        if self.bn!=0:
            self.bn_hooks = []
            for m in self.prior.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.bn_hooks.append( DeepInversionHook(m) )
            assert len(self.bn_hooks)>0, 'input model should contains at least one BN layer for DeepInversion'

    # Idea 2. Saliency-to-anchor alignment
    def _center_rad2(self, B):
        H, W = self.img_size[1], self.img_size[2]
        
        # Use calculated anchor coordinates
        cy = self.anchor_cy
        cx = self.anchor_cx

        y = self._coord_y - cy
        x = self._coord_x - cx
        rad2 = (y*y + x*x) / self._norm_denom

        return rad2.expand(B, 1, H, W)

    def _saliency_p(self, x, targets, current_abs_index, next_relative_index):
        x_req = x.clone().requires_grad_(True)

        logits, _, _ = self.teacher(x_req, current_abs_index, next_relative_index)
        if targets.dtype == torch.long and targets.dim() == 1:
            score = logits.gather(1, targets.view(-1, 1)).sum()
        else:
            hard_targets = targets.argmax(dim=1).long()
            score = logits.gather(1, hard_targets.view(-1, 1)).sum()

        grad = torch.autograd.grad(score, x_req, create_graph=True, retain_graph=True)[0]
        sal = grad.abs().amax(dim=1, keepdim=True) # [B, 1, H, W]

        sal_sum = sal.sum(dim=(2, 3), keepdim=True) + 1e-8
        p = sal / sal_sum # [B, 1, H, W]
        return p

    def synthesize(
        self, num_patches=197, prune_it=[-1], prune_ratio=[0],
        lpf_every=100, cutoff_ratio=0.9,
        sc_every=100, lambda_anchor=1.0,
        soft_label_alpha=1.0
    ):

        # Idea 1. Low-pass Filter
        self.lpf_every = lpf_every
        self.cutoff_ratio = cutoff_ratio

        # Idea 2. Saliency Map Centering
        self.sc_every = sc_every
        self.lambda_anchor = lambda_anchor

        H, W = self.img_size[1], self.img_size[2]
        self._coord_y = torch.linspace(0, H-1, H, device=self.device).view(1,1,H,1)
        self._coord_x = torch.linspace(0, W-1, W, device=self.device).view(1,1,1,W)
        self._norm_denom = float(H*H + W*W)

        self.anchor_cy, self.anchor_cx = (H - 1) / 2.0, (W - 1) / 2.0

        # Idea 3. Soft label
        self.soft_label_alpha = float(soft_label_alpha)

        self.student.eval()
        self.teacher.eval()

        best_cost = 1e6
        inputs = torch.randn(
            size=[self.synthesis_batch_size, *self.img_size],
            device=self.device
        ).requires_grad_()

        targets = torch.randint(
                low=0, high=self.num_classes,
                size=(self.synthesis_batch_size,)
            ).to(self.device)

        # Idea 3. Soft targets
        if not 0.0 <= self.soft_label_alpha <= 1.0:
            raise ValueError("soft_label_alpha should be between 0.0 and 1.0")
        off_value = (1.0 - self.soft_label_alpha) / (self.num_classes - 1)
        targets_soft = torch.full(
            (targets.size(0), self.num_classes),
            off_value,
            device=self.device,
            dtype=inputs.dtype
        )
        targets_soft.scatter_(1, targets.view(-1, 1), self.soft_label_alpha)
        targets = targets_soft

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])
        best_inputs = inputs.data

        current_abs_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        next_relative_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        inputs_aug = inputs
        for it in tqdm(range(self.iterations)):
            if it+1 in prune_it:
                inputs_aug = inputs
                current_abs_index_aug = current_abs_index

                # Idea 1. Low-pass Filter
                if ((it + 1) % self.lpf_every == 0):
                    inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)

                t_out, attention_weights, _ = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index)

            elif it in prune_it:
                prune_ratio_value = prune_ratio[prune_it.index(it)]

                num_spatial = self.num_patches_per_dim * self.num_patches_per_dim
                top_K = int(num_spatial * (1.0 - prune_ratio_value))
                top_K = max(1, min(num_spatial, top_K))

                print(f"top_K (Radial Order): {top_K} ### Iteration: {it}")

                B = inputs.shape[0]

                keep_flat_idx = self.patch_radial_order[:top_K]
                keep_patch_ids = (keep_flat_idx + 1).long()

                cls_col = torch.zeros(B, 1, dtype=torch.long, device=self.device)
                keep_col = keep_patch_ids.view(1, -1).expand(B, -1)
                next_relative_index = torch.cat([cls_col, keep_col], dim=1)

                inputs_aug = (inputs)
                current_abs_index_aug = current_abs_index

                # Idea 1. Low-pass Filter
                if ((it + 1) % self.lpf_every == 0):
                    inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)

                t_out, attention_weights, current_abs_index = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index)

            else:
                inputs_aug,off1,off2,flip = jitter_and_flip(inputs)
                if current_abs_index.shape[1]==num_patches:
                    current_abs_index_aug = current_abs_index
                else:
                    current_abs_index_aug =jitter_and_flip_index(current_abs_index,off1,off2,flip,self.patch_size,int(224//self.patch_size))

                # Idea 1. Low-pass Filter
                if ((it + 1) % self.lpf_every == 0):
                    inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)

                t_out,attention_weights,_ = self.teacher(inputs_aug,current_abs_index_aug,next_relative_index)

            # Loss
            if self.bn!=0:
                _ = self.prior(inputs_aug)
                rescale = [10.0] + [1. for _ in range(len(self.bn_hooks) - 1)]
                loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.bn_hooks)])
            else:
                loss_bn=0

            loss_oh = F.cross_entropy( t_out, targets )

            if self.adv>0:
                s_out = self.student(inputs_aug)
                loss_adv = -jsdiv(s_out, t_out, T=3)
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss_tv1,loss_tv2 = get_image_prior_losses(inputs)
            loss_l2 = torch.norm(inputs, 2)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv1 * loss_tv1 + self.tv2*loss_tv2 + self.l2 * loss_l2

            # Idea 2. Saliency-to-anchor alignment
            if ((it + 1) % self.sc_every == 0) and (it+1 < prune_it[-1]):
                B = inputs_aug.shape[0]
                p_sal = self._saliency_p(
                    inputs_aug, targets,
                    current_abs_index_aug,
                    next_relative_index
                )
                rad2 = self._center_rad2(B)

                L_center = (rad2 * p_sal).sum(dim=(1, 2, 3)).mean()
                loss = loss + self.lambda_anchor * L_center

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)

        print(f"Losses: loss_bn: {loss_bn}, loss_oh: {loss_oh}, loss_adv: {loss_adv}, loss_tv1: {loss_tv1}, loss_tv2: {loss_tv2}, loss_l2: {loss_l2}, loss_anchor: {L_center}")

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        if len(prune_ratio)==1 and prune_ratio[0]==0: #add non-masked image
            self.data_pool.add( best_inputs )

        with torch.no_grad():
            t_out, attention_weights, current_abs_index = self.teacher(
                best_inputs.detach(),
                torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device),
                torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
            )

        # Idea 3. Sparsification
        num_patches_per_dim = int(sqrt(num_patches))
        num_spatial = num_patches - 1

        def cumulative_mul(lst):
            current_mul = 1
            for num in lst:
                current_mul = current_mul*(1.-num)
            return current_mul

        top_K = int(num_spatial * cumulative_mul(prune_ratio))
        top_K = max(1, min(num_spatial, top_K))

        B = best_inputs.shape[0]
        keep_flat_idx = self.patch_radial_order[:top_K]
        keep_patch_ids = (keep_flat_idx + 1).long()

        cls_col = torch.zeros(B, 1, dtype=torch.long, device=self.device)
        keep_col = keep_patch_ids.view(1, -1).expand(B, -1)
        next_relative_index = torch.cat([cls_col, keep_col], dim=1)

        mask = torch.zeros(B, num_patches_per_dim, num_patches_per_dim, device=self.device)
        for b in range(B):
            for pid in next_relative_index[b, 1:]:
                idx = int(pid.item()) - 1
                r = idx // num_patches_per_dim
                c = idx % num_patches_per_dim
                mask[b, r, c] = 1.0
        
        expanded_mask = mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        masked_best_inputs = best_inputs * expanded_mask.unsqueeze(1)

        if not(len(prune_ratio)==1 and prune_ratio[0]==0): #add masked image
            self.data_pool.add( masked_best_inputs )

        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {
            'synthetic': best_inputs,
            'masked_synthetic':masked_best_inputs
        }
        
    def sample(self):
        return self.data_iter.next()
