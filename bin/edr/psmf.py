'''

script to estimate the probabilistic SMF by fitting a GMM 


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

import matplotlib as mpl
import matplotlib.pyplot as plt

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

#########################################################
# read in BGS data
#########################################################
dat_dir = '/tigress/chhahn/provabgs/svda'
bgs = aTable.Table.read(os.path.join(dat_dir, 'BGS_ANY_full.provabgs.lite.hdf5'))

if targ == 'bgs_bright': 
    bgs = bgs[bgs['is_bgs_bright']]
elif targ =='bgs_faint': 
    bgs = bgs[bgs['is_bgs_faint']]
elif targ == 'bgs_any': 
    pass
elif targ == 'bgs_bright_q': 
    is_quiescent = ((np.log10(np.median(bgs['provabgs_avgSFR1Gyr_mcmc'].data, axis=1)) - np.median(bgs['provabgs_logMstar'].data, axis=1)) < -11.2) 
    bgs = bgs[bgs['is_bgs_bright'] & is_quiescent]
elif targ == 'bgs_bright_sf': 
    is_quiescent = ((np.log10(np.median(bgs['provabgs_avgSFR1Gyr_mcmc'].data, axis=1)) - np.median(bgs['provabgs_logMstar'].data, axis=1)) < -11.2) 
    bgs = bgs[bgs['is_bgs_bright'] & ~is_quiescent]

# redshift limit 
zlim = (bgs['Z_HP'].data > zmin) & (bgs['Z_HP'].data < zmax) & (bgs['provabgs_w_zfail'].data > 0)
bgs = bgs[zlim]

print('%i %s galaxies with posteriors' % (len(bgs), targ))
#########################################################


class GaussianMixtureModel(nn.Module):
    def __init__(self, n_components: int=2, mmin=7., mmax=13.):
        super().__init__()
        logweights = torch.zeros(n_components, )
        means   = (mmax-mmin)*torch.rand(n_components, ) + mmin
        logstdevs  = 0.1*torch.tensor(np.random.uniform(size=n_components)) - 1.
        
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

# calculate vmax
f_area = (173.641/(4.*np.pi*(180/np.pi)**2))
v_zmin = Planck13.comoving_volume(zmin).value * Planck13.h**3 * f_area # (Mpc/h)^3
v_zmax = Planck13.comoving_volume(zmax).value * Planck13.h**3 * f_area # (Mpc/h)^3

# calculate weights 
w_import = (v_zmax - v_zmin) / (bgs['Vmax'].data.clip(v_zmin, v_zmax) - v_zmin) 
w_import *= bgs['provabgs_w_zfail'].data * bgs['provabgs_w_fibassign'].data

x_data = torch.tensor(bgs['provabgs_logMstar'].data.astype(np.float32)).to(device)
w_data = torch.tensor(w_import.astype(np.float32)).to(device)

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

num_iter    = 1000
patience    = 20
n_model     = 10 
lr          = 1e-2


fig = plt.figure(figsize=(6,4))
sub = fig.add_subplot(111)

best_flows, best_valid_losses, vls = [], [], []
last_flows = [] 
for i in range(n_model): 
    ncomp = int(np.random.uniform(5, 100))
    flow = GaussianMixtureModel(n_components=ncomp, 
                                mmin=bgs['provabgs_logMstar'].min(), 
                                mmax=bgs['provabgs_logMstar'].max())
    flow.to(device)
    print('GMM with %i components' % ncomp)

    best_epoch, best_valid_loss = 0, np.inf
    valid_losses = []
    
    optimizer = optim.Adam(flow.parameters(), lr=lr)            
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr=1e-4)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_iter, steps_per_epoch=len(trainloader))

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
            # scheduler.step()
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
        
        scheduler.step(valid_loss)
        t.set_description('Epoch: %i LR: %.2e TRAINING Loss: %.2e VALIDATION Loss: %.2e' % 
                          (epoch, scheduler._last_lr[0], train_loss, valid_loss), refresh=False)

        if valid_loss < best_valid_loss: 
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_flow = copy.deepcopy(flow)
        #else: 
        #    if best_epoch < epoch - patience: 
        #        print('>>>%i \t %.5e' % (epoch, best_valid_loss))
        #        break
    print('>>> %.5e' % best_valid_loss)
    with torch.no_grad(): 
        post_prime = best_flow.sample((10000,))
        _ = sub.hist(np.array(post_prime.detach().cpu()), range=(7., 13.), bins=40, histtype='step', linewidth=1)

    last_flows.append(copy.deepcopy(flow))
    best_flows.append(best_flow)
    best_valid_losses.append(best_valid_loss)
    vls.append(valid_losses)
    
ibest = np.argmin(best_valid_losses)
fgmm = os.path.join(dat_dir, 'psmf.gmm.%s.z%.2f_%.2f.pt' % (targ, zmin, zmax))
torch.save(best_flows[ibest], fgmm)
torch.save(last_flows[np.argmin([vl[-1] for vl in vls])], fgmm.replace('.pt', '.last.pt'))

with torch.no_grad(): 
    post_prime = best_flows[ibest].sample((10000,))
    _ = sub.hist(np.array(post_prime.detach().cpu()), range=(7., 13.), bins=40, histtype='step', linewidth=2, color='k')

_ = sub.hist(np.median(bgs['provabgs_logMstar'], axis=1),
        weights=w_import*10000./np.sum(w_import),
        range=(7., 13.), bins=40, histtype='step', color='k', linestyle='--', linewidth=2)

sub.set_xlabel(r'$\log M_*$', fontsize=25)
sub.set_xlim(7., 13.)
sub.set_ylabel(r'$p(\log M_*)$', fontsize=25)
sub.set_yscale('log')
fig.savefig(os.path.join(dat_dir, 'psmf.gmm.%s.z%.2f_%.2f.png' % (targ, zmin, zmax)), bbox_inches='tight') 

fig = plt.figure(figsize=(5,5))
sub = fig.add_subplot(111)
for vl in vls:
    sub.plot(np.arange(len(vl)), vl)
sub.plot(np.arange(len(vls[ibest])), vls[ibest], c='k')
sub.set_xlim(0, np.max([len(vl) for vl in vls]))
fig.savefig(os.path.join(dat_dir, 'psmf.gmm.%s.z%.2f_%.2f.loss.png' % (targ, zmin, zmax)), bbox_inches='tight') 
