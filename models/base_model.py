import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.searchable_modules = []

    def calculate_search_threshold(self, budget_attn, budget_mlp, budget_patch):
        zetas_attn, zetas_mlp, zetas_patch = self.give_zetas()
        zetas_attn = sorted(zetas_attn)
        zetas_mlp = sorted(zetas_mlp)
        zetas_patch = sorted(zetas_patch)
        threshold_attn = zetas_attn[int((1.-budget_attn)*len(zetas_attn))]
        threshold_mlp = zetas_mlp[int((1.-budget_mlp)*len(zetas_mlp))]
        threshold_patch = zetas_patch[int((1.-budget_patch)*len(zetas_patch))]
        return threshold_attn, threshold_mlp, threshold_patch
    
    def n_remaining(self, m):
        if hasattr(m, 'num_heads'):
            return  (m.searched_zeta if m.is_searched else m.zeta).sum(), (m.searched_patch_zeta if m.is_searched else torch.tanh(m.patch_zeta)).sum()
        return (m.searched_zeta if m.is_searched else m.get_zeta()).sum()
    
    def get_remaining(self):
        """return the fraction of active zeta""" 
        n_rem_attn = 0
        n_total_attn = 0
        n_rem_mlp = 0
        n_total_mlp = 0
        n_rem_patch = 0
        n_total_patch = 0
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                attn, patch = self.n_remaining(l_block)
                n_rem_attn += attn
                n_total_attn += l_block.num_gates*l_block.num_heads
                n_rem_patch += patch
                n_total_patch += self.num_patches
            else:
                n_rem_mlp += self.n_remaining(l_block)
                n_total_mlp += l_block.num_gates
        return n_rem_attn/n_total_attn, n_rem_mlp/n_total_mlp, n_rem_patch/n_total_patch

    def get_sparsity_loss(self, device):
        loss_attn = torch.FloatTensor([]).to(device)
        loss_mlp = torch.FloatTensor([]).to(device)
        loss_patch = torch.FloatTensor([]).to(device)
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                loss_attn = torch.cat([loss_attn, torch.abs(zeta_attn.view(-1))])
                loss_patch = torch.cat([loss_patch, torch.abs(zeta_patch.view(-1))])
            else:
                loss_mlp = torch.cat([loss_mlp, torch.abs(l_block.get_zeta().view(-1))])
        return torch.sum(loss_attn).to(device), torch.sum(loss_mlp).to(device), torch.sum(loss_patch).to(device)

    def get_sparsity_loss_layerwise(self, device, w):
        loss_attn = torch.FloatTensor([]).to(device)
        loss_mlp = torch.FloatTensor([]).to(device)
        loss_patch = torch.FloatTensor([]).to(device)
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                loss_attn = torch.cat([loss_attn.to(device), torch.FloatTensor([torch.sum(torch.abs(zeta_attn.view(-1)))]).to(device)])
                loss_patch = torch.cat([loss_patch.to(device), torch.FloatTensor([torch.sum(torch.abs(zeta_patch.view(-1)))]).to(device)])
            else:
                loss_mlp = torch.cat([loss_mlp.to(device), torch.FloatTensor([torch.sum(torch.abs(l_block.get_zeta().view(-1)))]).to(device)])
        
        loss_layerwise = torch.FloatTensor([]).to(device)
        loss_layerwise = torch.cat([loss_layerwise, loss_attn, loss_mlp, loss_patch])
        loss_layerwise = torch.FloatTensor(w).to(device)*loss_layerwise
        return torch.sum(loss_layerwise).to(device)
    
    def get_sparsity_loss_channelwise(self, device, w):
        loss_attn = torch.FloatTensor([]).to(device)
        loss_mlp = torch.FloatTensor([]).to(device)
        loss_patch = torch.FloatTensor([]).to(device)
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                loss_attn = torch.cat([loss_attn.to(device), torch.abs(zeta_attn.view(-1)).to(device)])
                loss_patch = torch.cat([loss_patch.to(device), torch.abs(zeta_patch.view(-1)).to(device)])
            else:
                loss_mlp = torch.cat([loss_mlp.to(device), torch.abs(l_block.get_zeta().view(-1)).to(device)])
        
        loss_channelwise = torch.FloatTensor([]).to(device)
        loss_channelwise = torch.cat([loss_channelwise, loss_attn, loss_mlp, loss_patch])
        loss_channelwise = torch.FloatTensor(w).to(device)*loss_channelwise
        return torch.sum(loss_channelwise).to(device)

    def get_sparsity_loss_channelwise_one(self, device, w):
        loss_attn = torch.FloatTensor([]).to(device)
        loss_mlp = torch.FloatTensor([]).to(device)
        loss_patch = torch.FloatTensor([]).to(device)
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                loss_attn = torch.cat([loss_attn.to(device), torch.abs(torch.abs(torch.sub(zeta_attn.view(-1), 1))).to(device)])
                loss_patch = torch.cat([loss_patch.to(device), torch.abs(torch.abs(torch.sub(zeta_patch.view(-1), 1))).to(device)])
            else:
                loss_mlp = torch.cat([loss_mlp.to(device), torch.abs(torch.abs(torch.sub(l_block.get_zeta().view(-1), 1))).to(device)])
        
        loss_channelwise = torch.FloatTensor([]).to(device)
        loss_channelwise = torch.cat([loss_channelwise, loss_attn, loss_mlp, loss_patch])
        loss_channelwise = torch.FloatTensor(w).to(device)*loss_channelwise
        return torch.sum(loss_channelwise).to(device)

    def get_discreteness_loss(self, device):
        loss_attn = torch.FloatTensor([]).to(device)
        loss_mlp = torch.FloatTensor([]).to(device)
        loss_patch = torch.FloatTensor([]).to(device)
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                loss_attn = torch.cat([loss_attn, torch.abs(zeta_attn.view(-1))])
                loss_patch = torch.cat([loss_patch, torch.abs(zeta_patch.view(-1))])
            else:
                loss_mlp = torch.cat([loss_mlp, torch.abs(l_block.get_zeta().view(-1))])
        
        loss_attn = torch.min(torch.abs(torch.sub(loss_attn,1)), torch.abs(loss_attn))
        loss_patch = torch.min(torch.abs(torch.sub(loss_patch,1)), torch.abs(loss_patch))
        loss_mlp = torch.min(torch.abs(torch.sub(loss_mlp,1)), torch.abs(loss_mlp))

        return torch.sum(loss_attn).to(device), torch.sum(loss_mlp).to(device), torch.sum(loss_patch).to(device)  

    def add_hooks_zetas(self, grad_masks, device):
        masks_attn = grad_masks[0]
        masks_mlp = grad_masks[1]
        masks_patch = grad_masks[2]
        cnt_attn = 0
        cnt_mlp = 0
        cnt_patch = 0
        hooks_attn = []
        hooks_mlp = []
        hooks_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                mask_attn_tensor = torch.tensor(masks_attn[cnt_attn]).reshape(zeta_attn.size())
                mask_patch_tensor = torch.tensor(masks_patch[cnt_attn]).reshape(zeta_patch.size())
                cnt_attn += 1
                cnt_patch += 1
                hook_attn = zeta_attn.register_hook(lambda grad: grad.mul_(mask_attn_tensor.to(device)))
                hook_patch = zeta_patch.register_hook(lambda grad: grad.mul_(mask_patch_tensor.to(device)))
                hooks_attn.append(hook_attn)
                hooks_patch.append(hook_patch)
            else:
                zeta_mlp = l_block.get_zeta()
                mask_mlp_tensor = torch.tensor(masks_mlp[cnt_mlp]).reshape(zeta_mlp.size())
                cnt_mlp += 1
                hook_mlp = zeta_mlp.register_hook(lambda grad: grad.mul_(mask_mlp_tensor.to(device)))
                hooks_mlp.append(hook_mlp)
        return hooks_attn, hooks_mlp, hooks_patch

    def remove_hooks_zetas(self, hooks_attn, hooks_mlp, hooks_patch):
        for hook_attn in hooks_attn:
            hook_attn.remove()
        for hook_mlp in hooks_mlp:
            hook_mlp.remove()
        for hook_patch in hooks_patch:
            hook_patch.remove()

    def give_zetas(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                zetas_attn.append(zeta_attn.cpu().detach().reshape(-1).numpy().tolist())
                zetas_patch.append(zeta_patch.cpu().detach().reshape(-1).numpy().tolist())
            else:
                zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
        zetas_attn = [z for k in zetas_attn for z in k ]
        zetas_mlp = [z for k in zetas_mlp for z in k ]
        zetas_patch = [z for k in zetas_patch for z in k ]
        return zetas_attn, zetas_mlp, zetas_patch

    def give_zetas_layerwise(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                zetas_attn.append(zeta_attn.cpu().detach().reshape(-1).numpy().tolist())
                zetas_patch.append(zeta_patch.cpu().detach().reshape(-1).numpy().tolist())
            else:
                zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
        return zetas_attn, zetas_mlp, zetas_patch

    def give_zetas_channelwise(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                zetas_attn.append(zeta_attn.cpu().detach().reshape(-1).numpy().tolist())
                zetas_patch.append(zeta_patch.cpu().detach().reshape(-1).numpy().tolist())
            else:
                zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
        zetas_attn = [z for k in zetas_attn for z in k ]
        zetas_mlp = [z for k in zetas_mlp for z in k ]
        zetas_patch = [z for k in zetas_patch for z in k ]
        zetas = []
        zetas.extend(zetas_attn)
        zetas.extend(zetas_mlp)
        zetas.extend(zetas_patch)
        return zetas

    def give_pis_layerwise(self):
        pis_attn = []
        pis_mlp = []
        pis_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                l_attn = zeta_attn.cpu().detach().reshape(-1).numpy().tolist()
                l_patch = zeta_patch.cpu().detach().reshape(-1).numpy().tolist()
                pi_attn = np.mean(list(map(np.exp, l_attn)))
                pi_patch = np.mean(list(map(np.exp, l_patch)))
                pis_attn.append(pi_attn)
                pis_patch.append(pi_patch)
            else:
                l_mlp = l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist()
                pi_mlp = np.mean(list(map(np.exp, l_mlp)))
                pis_mlp.append(pi_mlp)
        return pis_attn, pis_mlp, pis_patch

    def give_pis_channelwise(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                zetas_attn.append(zeta_attn.cpu().detach().reshape(-1).numpy().tolist())
                zetas_patch.append(zeta_patch.cpu().detach().reshape(-1).numpy().tolist())
            else:
                zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
        zetas_attn = [z for k in zetas_attn for z in k ]
        zetas_mlp = [z for k in zetas_mlp for z in k ]
        zetas_patch = [z for k in zetas_patch for z in k ]

        pis_attn = list(map(np.exp, zetas_attn))
        pis_mlp = list(map(np.exp, zetas_mlp))
        pis_patch = list(map(np.exp, zetas_patch))

        return pis_attn, pis_mlp, pis_patch

    def plot_zt(self):
        """plots the distribution of zeta_t and returns the same"""
        zetas_attn, zetas_mlp, zetas_patch = self.give_zetas()
        zetas = zetas_attn + zetas_mlp + zetas_patch
        exactly_zeros = np.sum(np.array(zetas)==0.0)
        exactly_ones = np.sum(np.array(zetas)==1.0)
        plt.hist(zetas)
        plt.show()
        return exactly_zeros, exactly_ones

    def compress(self, budget_attn, budget_mlp, budget_patch):
        """compress the network to make zeta exactly 1 and 0"""
        if self.searchable_modules == []:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        thresh_attn, thresh_mlp, thresh_patch = self.calculate_search_threshold(budget_attn, budget_mlp, budget_patch)
                
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.compress(thresh_attn)
            else:
                l_block.compress(thresh_mlp)
        self.compress_patch(thresh_patch)
        return thresh_attn, thresh_mlp, 0
    
    def compress_with_thresh(self, thresh_attn, thresh_mlp, thresh_patch):
        """compress the network with fixed threshold to make zeta exactly 1 and 0"""
        if self.searchable_modules == []:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        # thresh_attn, thresh_mlp, thresh_patch = self.calculate_search_threshold(budget_attn, budget_mlp, budget_patch)
                
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.compress(thresh_attn)
            else:
                l_block.compress(thresh_mlp)
        self.compress_patch(thresh_patch)
        # return thresh_attn, thresh_mlp, 0
        return thresh_attn, thresh_mlp, thresh_patch
    
    def compress_patch(self, threshold):
        zetas = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                _, zeta_patch = l_block.get_zeta()
                zeta_patch = zeta_patch.cpu().detach().numpy()
                zetas.append(zeta_patch)
        mask = np.zeros_like(zeta_patch)
        for i in range(len(zetas)-1, -1, -1):
            temp_mask = zetas[i]>=threshold
            mask = np.logical_or(mask, temp_mask).astype(np.float32)
            zetas[i] = mask
        i = 0
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.compress_patch(threshold, zetas[i])
                i+=1

    def correct_require_grad(self, w1, w2, w3):
        if w1==0:
            for l_block in self.searchable_modules:
                if hasattr(l_block, 'num_heads'):
                    l_block.zeta.requires_grad = False
        if w2==0:
            for l_block in self.searchable_modules:
                if not hasattr(l_block, 'num_heads'):
                    l_block.zeta.requires_grad = False
        if w3==0:
            for l_block in self.searchable_modules:
                if hasattr(l_block, 'num_heads'):
                    l_block.patch_zeta.requires_grad = False

    def decompress(self):
        for l_block in self.searchable_modules:
            l_block.decompress()
    
    def get_channels(self):
        active_channels_attn = []
        active_channels_mlp = []
        active_channels_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                active_channels_attn.append(l_block.searched_zeta.numpy())
                active_channels_patch.append(l_block.searched_patch_zeta.numpy())
            else:
                active_channels_mlp.append(l_block.searched_zeta.sum().item())
        return np.squeeze(np.array(active_channels_attn)), np.array(active_channels_mlp), np.squeeze(np.array(active_channels_patch))

    def get_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        searched_params = total_params
        for l_block in self.searchable_modules:
            searched_params-=l_block.get_params_count()[0]
            searched_params+=l_block.get_params_count()[1]
        return total_params, searched_params.item()