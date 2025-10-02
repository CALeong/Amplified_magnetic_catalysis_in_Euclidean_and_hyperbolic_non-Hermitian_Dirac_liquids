import os
import numpy as np
import matplotlib.pyplot as plt

def schf_manual_plot(vlist,dir_arr):
    for da in dir_arr:
        data = np.load(da)
        plt.scatter(vlist,data,label=da.split('/')[-1])
    plt.legend()
    plt.show()
def schf_plot_preset_slices(p,q,n,alpha,subdir,vlist):
    if alpha == 0:
        alpha = 0
    center_data = np.array([])
    add_data = np.load(subdir+'/'+'center_delta_p{}q{}n{}a{}_Vstart0.npy'.format(p,q,n,alpha))
    center_data = np.concatenate((center_data,add_data))
    add_data = np.load(subdir+'/'+'center_delta_p{}q{}n{}a{}_Vstart0.6.npy'.format(p, q, n, alpha))
    center_data = np.concatenate((center_data, add_data))
    add_data = np.load(subdir+'/'+'center_delta_p{}q{}n{}a{}_Vstart1.npy'.format(p, q, n, alpha))
    center_data = np.concatenate((center_data, add_data))
    add_data = np.load(subdir+'/'+'center_delta_p{}q{}n{}a{}_Vstart1.4.npy'.format(p, q, n, alpha))
    center_data = np.concatenate((center_data, add_data))
    system_data = np.array([])
    add_data = np.load(subdir+'/'+'system_delta_p{}q{}n{}a{}_Vstart0.npy'.format(p, q, n, alpha))
    system_data = np.concatenate((system_data, add_data))
    add_data = np.load(subdir+'/'+'system_delta_p{}q{}n{}a{}_Vstart0.6.npy'.format(p, q, n, alpha))
    system_data = np.concatenate((system_data, add_data))
    add_data = np.load(subdir+'/'+'system_delta_p{}q{}n{}a{}_Vstart1.npy'.format(p, q, n, alpha))
    system_data = np.concatenate((system_data, add_data))
    add_data = np.load(subdir+'/'+'system_delta_p{}q{}n{}a{}_Vstart1.4.npy'.format(p, q, n, alpha))
    system_data = np.concatenate((system_data, add_data))
    plt.title('a={}'.format(alpha))
    plt.scatter(vlist,system_data,label='System')
    plt.scatter(vlist,center_data,label='Center')
    plt.show()

def schf_multi_plot(subdir,vlist,system=True,center=False):
    os.chdir(subdir)
    files = os.listdir()
    if system==True:
        for f in files:
            if 'System' in f:
                data = np.load(f)
                plt.scatter(vlist,data,label=f)
    if center==True:
        for f in files:
            if 'Center' in f:
                data = np.load(f)
                plt.scatter(vlist,data,label=f)
    plt.legend()
    plt.show()

def schf_PeierlsSub_manual_plot(subdir_list, selectedVs, hfcoeff_list, alist, totalnumplaquets):
    data_mat = hfcoeff_list
    for sl in subdir_list:
        data = np.load(sl)
        data_mat = np.vstack((data_mat,data))
    magflux = totalnumplaquets*alist
    for sv in selectedVs:
        selected_data = data_mat[:,np.where(np.around(hfcoeff_list,8)==round(sv,8))[0][0]]
        selected_data = selected_data[1:]
        plt.scatter(magflux, selected_data, label=round(sv,2))
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()
