import numpy as np
from Fundamental.Bands import *
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_vc(p,alpha,numberksamples,savedir):
    energy_list = diracq3_HBT_Energies_Interactions_vectorized(p,alpha,0,numberksamples,savedir)
    np.save(savedir+'/p{}a{}nk{}_vcritcalc'.format(p,alpha,numberksamples), energy_list)
    vc = 1 / ((3/(4*numberksamples))*np.sum(np.abs(1/energy_list)))
    np.save(savedir+'/p{}a{}nk{}_vcritvalue'.format(p,alpha,numberksamples), vc)
    # file = open(savedir+'/p{}a{}nk{}_vcritvalue'.format(p,alpha,numberksamples),'w')
    # file.write('{}'.format(vc))
    return(vc)

def get_vs(p,alpha,delta_list,numberksamples,savedir):
    delta_vlist_mat = np.zeros((1,2))
    vlist = np.array([])
    for delta in delta_list:
        energy_list = diracq3_HBT_Energies_Interactions_vectorized(p,alpha,delta,numberksamples,savedir)
        np.save(savedir+'/p{}a{}d{}nk{}_vcalc'.format(p,alpha,delta,numberksamples), energy_list)
        vlist = np.append(vlist, 1 / ((3/(4*numberksamples))*np.sum(np.abs(1/energy_list))))
        delta_vlist_mat = np.vstack((delta_vlist_mat, np.array([delta, 1 / ((3/(4*numberksamples))*np.sum(np.abs(1/energy_list)))])))
    delta_vlist_mat = delta_vlist_mat[1:,:]
    np.save(savedir + '/p{}a{}nk{}_vvalues'.format(p, alpha, numberksamples), delta_vlist_mat)
    # file = open(savedir + '/p{}a{}nk{}_vvalues'.format(p, alpha, numberksamples), 'w')
    # file.write('vlist: {}\n deltalist: {}'.format(vlist,delta_list))
    return(vlist)

def plot_vdata(vdata_file_address, vcrit_file_address):
    data_mat = np.load(vdata_file_address)
    deltavals = data_mat[:,0]
    vvals = data_mat[:,1]
    vcrit = np.load(vcrit_file_address)
    alldeltas = np.append(deltavals,0)
    allvs = np.append(vvals,vcrit)
    plt.scatter(allvs, alldeltas, s=5)
    plt.show()

def get_highdelta_slope(vsdatafile_address,cutoff_delta):
    data = np.load(vsdatafile_address)
    deltavals = data[:,0]
    vvals = data[:,1]
    highdelta_locs = np.where(deltavals>cutoff_delta)[0]
    highdeltavals = deltavals[highdelta_locs]
    highvvals = vvals[highdelta_locs]
    def linfit(v,m,b): return(m*v + b)
    fits = curve_fit(linfit, highvvals, highdeltavals)[0]
    print('Fit Slope: {}'.format(fits[0]))
    print('Fit intercept: {}'.format(fits[1]))
    return(fits)

def get_nonherm_alpha_vcritvalues(p,alpha_list,numberksamples,savedir):
    data_mat = np.zeros((1,2))
    for al in alpha_list:
        energies = diracq3_HBT_Energies_Interactions_vectorized(p,al,10**(-3),numberksamples,savedir)
        vc = 1 / ((3 / (4 * numberksamples)) * np.sum(np.abs(1 / energies)))
        data_mat = np.vstack((data_mat, np.array([al, vc])))
    data_mat = data_mat[1:,:]
    np.save(savedir + '/p{}Nk{}_alphavcrit_data'.format(p,numberksamples), data_mat)
    return(data_mat)

def get_alpha_vcrit_trend(data_mat):
    alphas = data_mat[:,0]
    vcsratio = data_mat[:,1]/data_mat[0,1]
    def acvfit(als): return(np.sqrt(1-als**2))
    curve_fit(acvfit,alphas,vcsratio)

def plot_alpha_vcritvalues_trend(data_mat):
    alphas = data_mat[:,0]
    vcs = data_mat[:,1]
    plt.scatter(alphas, vcs/vcs[0])
    def alphacrit_trend(a): return(np.sqrt(1-a**2))
    afoo = np.linspace(alphas[0],alphas[-1],1000)
    plt.plot(afoo,alphacrit_trend(afoo),color='red')
    plt.show()