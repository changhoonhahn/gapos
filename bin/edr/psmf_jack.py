'''

script to estimate the jackknife probabilistic SMF by fitting a GMM 


'''
import os, sys 
import numpy as np
from tqdm.auto import trange
import astropy.table as aTable
from astropy.cosmology import Planck13

import copy
from nflows import transforms, distributions, flows

import torch
from torch import nn
from torch import optim
import torch.distributions as D

if torch.cuda.is_available():
    device = torch.device(type='cuda', index=0)
else:
    device = 'cpu'

#########################################################
# input
#########################################################
targ = sys.argv[1]
zmin = float(sys.argv[2])
zmax = float(sys.argv[3]) 
#########################################################
# jackknife fields 
#########################################################
fields = [
    ((bgs['RA'] > 140) & (bgs['RA'] < 160)), 
    ((bgs['RA'] > 160) & (bgs['RA'] < 185) & (bgs['DEC'] > -10) & (bgs['DEC'] < 10)), 
    ((bgs['RA'] > 205) & (bgs['RA'] < 212) & (bgs['DEC'] > 2) & (bgs['DEC'] < 8)), 
    ((bgs['RA'] > 210) & (bgs['RA'] < 224) & (bgs['DEC'] > -5) & (bgs['DEC'] < 5) & ~((bgs['RA'] > 205) & (bgs['RA'] < 212) & (bgs['DEC'] > 2) & (bgs['DEC'] < 8))),
    ((bgs['RA'] > 190) & (bgs['RA'] < 200) & (bgs['DEC'] > 20) & (bgs['DEC'] < 30)), 
    ((bgs['RA'] > 210) & (bgs['RA'] < 225) & (bgs['DEC'] > 30) & (bgs['DEC'] < 40)),     
    ((bgs['RA'] > 250) & (bgs['RA'] < 260) & (bgs['DEC'] > 30) & (bgs['DEC'] < 40)),
    ((bgs['RA'] > 230) & (bgs['RA'] < 255) & (bgs['DEC'] > 40) & (bgs['DEC'] < 45)),    
    ((bgs['RA'] > 210) & (bgs['RA'] < 225) & (bgs['DEC'] > 45) & (bgs['DEC'] < 58)),        
    ((bgs['RA'] > 235) & (bgs['RA'] < 255) & (bgs['DEC'] > 50) & (bgs['DEC'] < 60)), 
    ((bgs['RA'] > 180) & (bgs['RA'] < 200) & (bgs['DEC'] > 55) & (bgs['DEC'] < 70)), 
    ((bgs['RA'] > 260) & (bgs['RA'] < 280) & (bgs['DEC'] > 55) & (bgs['DEC'] < 70))    
]

A_fields = [164.965, 156.064, 164.917, 138.785, 155.972, 164.848, 165.120, 148.024, 
            165.015, 165.030, 164.890, 156.806]
#########################################################
# read in BGS data
#########################################################
dat_dir = '/tigress/chhahn/provabgs/svda'
bgs = aTable.Table.read(os.path.join(dat_dir, 'BGS_ANY_full.provabgs.hdf5'))

has_posterior = (bgs['provabgs_z_max'].data != -999.)

if targ == 'bgs_bright': 
    is_bgs = bgs['is_bgs_bright']
elif targ == 'bgs_any': 
    is_bgs = (bgs['is_bgs_bright'] | bgs['is_bgs_faint'])

bgs = bgs[has_posterior & is_bgs]
print('%i %s galaxies with posteriors' % (len(bgs), targ))
#########################################################

class GaussianMixtureModel(nn.Module):
    def __init__(self, n_components: int=2):
        super().__init__()
        logweights = torch.zeros(n_components, )
        means   = torch.randn(n_components, ) + 10.
        logstdevs  = 0.1 * torch.tensor(np.random.randn(n_components, ))
        self.logweights = torch.nn.Parameter(logweights)
        self.means   = torch.nn.Parameter(means)
        self.logstdevs  = torch.nn.Parameter(logstdevs)

    def forward(self, x):
        mix  = D.Categorical(torch.exp(self.logweights))
        comp = D.Normal(self.means, torch.exp(self.logstdevs))
        gmm  = D.MixtureSameFamily(mix, comp)
        return - gmm.log_prob(x).mean()

    def log_prob(self, x):
        mix  = D.Categorical(torch.exp(self.logweights))
        comp = D.Normal(self.means, torch.exp(self.logstdevs))
        gmm  = D.MixtureSameFamily(mix, comp)
        return gmm.log_prob(x)

    def sample(self, N):
        mix  = D.Categorical(torch.exp(self.logweights))
        comp = D.Normal(self.means, torch.exp(self.logstdevs))
        gmm  = D.MixtureSameFamily(mix, comp)

        return gmm.sample(N)


def Loss(qphi, post, w):
    ''' calculate loss

    \sum_i^Ng w_i * \log \sum_j^Ns qphi(\theta_ij)

    '''
    logqphi = qphi.log_prob(post.flatten()[:,None]).reshape(post.shape)
    return -torch.sum(w * torch.logsumexp(logqphi, axis=1))


for i_jack, field in enumerate(fields): # loop through jackknife fields
    fgmm = os.path.join(dat_dir, 'psmf.gmm.%s.z%.2f_%.2f.jack%i.best.pt' % (targ, zmin, zmax, i_jack))
    if os.path.isfile(fgmm): continue
    
    # Vmax based importance weights
    zlim = (bgs['Z_HP'].data > zmin) & (bgs['Z_HP'].data < zmax) & ~field & (bgs['provabgs_w_zfail'].data > 0)

    v_zmin = Planck13.comoving_volume(zmin).value * Planck13.h**3 # (Mpc/h)^3
    v_zmax = Planck13.comoving_volume(zmax).value * Planck13.h**3 # (Mpc/h)^3

    w_import = (v_zmax - v_zmin) / (vmaxes.clip(v_zmin, v_zmax) - v_zmin) 
    w_import *= bgs['provabgs_w_zfail'].data * bgs['provabgs_w_fibassign']

    x_data = torch.tensor(logM_posteriors[zlim].astype(np.float32)).to(device)
    w_data = torch.tensor(w_import[zlim].astype(np.float32)).to(device)

    batch_size = 128
    Ntrain = int(0.9 * x_data.shape[0])
    Nvalid = x_data.shape[0] - Ntrain 

    trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_data[:Ntrain], w_data[:Ntrain]),
            batch_size=batch_size,
            shuffle=True)

    validloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_data[Ntrain:], w_data[Ntrain:]),
            batch_size=batch_size)

    best_flows, best_valid_losses = [], []
    for i in range(3): 
        ncomp = int(np.random.uniform(5, 100))
        flow = GaussianMixtureModel(n_components=ncomp)
        flow.to(device)
        print('GMM with %i components' % ncomp)

        # parameters = [weights, means, stdevs]
        optimizer = optim.Adam(flow.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=num_iter)

        best_epoch, best_valid_loss = 0, np.inf
        valid_losses = []

        t = trange(num_iter, leave=False)
        for epoch in t:
            train_loss = 0.
            for batch in trainloader: 
                optimizer.zero_grad()
                _post, _w = batch
                _post = _post.to(device)
                _w = _w.to(device)

                loss = Loss(flow, _post, _w)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_loss /= len(trainloader.dataset)

            with torch.no_grad():
                valid_loss = 0.
                for batch in validloader: 
                    _post, _w = batch
                    _post = _post.to(device)
                    _w = _w.to(device)

                    loss = Loss(flow, _post, _w)                
                    valid_loss += loss.item()
                valid_loss /= len(validloader.dataset)           
                valid_losses.append(valid_loss)

            scheduler.step()

            t.set_description('Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % 
                              (epoch, train_loss, valid_loss), refresh=False)

            if valid_loss < best_valid_loss: 
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_flow = copy.deepcopy(flow)
            else: 
                if best_epoch < epoch - patience: 
                    print('>>>%i \t %.5e' % (epoch, best_valid_loss))
                    break

        best_flows.append(best_flow)
        best_valid_losses.append(best_valid_loss)
        
    ibest = np.argmin(best_valid_losses)
    torch.save(best_flows[ibest], fgmm)
