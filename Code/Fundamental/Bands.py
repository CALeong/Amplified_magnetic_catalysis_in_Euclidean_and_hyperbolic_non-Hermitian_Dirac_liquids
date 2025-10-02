import numpy as np
import matplotlib.pyplot as plt
from Fundamental.Number_Points import points

#################################################
###Core dirac q3 code stuff for no interaction###
#################################################

def diracq3_klist(ind_klist):
    klist_altsum = 0
    for i in range(len(ind_klist)):
        klist_altsum += ((-1)**i)*ind_klist[i]
    # genus = diracq3_getgenus(2*(len(ind_klist)+1))
    k_dependent = -1*(klist_altsum)
    return(np.append(ind_klist,k_dependent))

def diracq3_ham(p,klist,savedir):
    #For Dirac q=3, number of independent k are (p/2)-1
    #Unit cell just contains first generation ring
    #Brillouin zone wraps across opposite edges
    #t=1 between nearest neighbors
    #e^ik for across unit cell boundaries
    H0 = np.zeros((p,p),dtype=np.complex_)
    t=1
    for n in range(p):
        if n != p-1:
            H0[n][n+1] = t
            H0[n+1][n] = t
        else:
            H0[n][0] = t
            H0[0][n] = t
    for i in range(int(p/2)):
        H0[i][int(i+p/2)] = np.exp(1j*klist[i])
        H0[int(i+p/2)][i] = np.exp(-1j*klist[i])
    # np.save(savedir + '/p{}_klist{}'.format(p,klist),H0)
    return(H0)

def diracq3_diagonalcut(p, rangemin, rangemax, numsteps, savedir):
    step_size = (rangemax-rangemin)/numsteps
    kvals = np.arange(rangemin,rangemax+step_size,step_size)
    energies = np.zeros((1,p+1))
    for n in range(len(kvals)):
        ham = diracq3_ham(p, diracq3_klist(np.repeat(kvals[n],int((p/2)-1))),savedir)
        eigvals = np.linalg.eigvalsh(ham)
        energies = np.vstack((energies,np.append(kvals[n],eigvals)))
    energies = energies[1:,:]
    np.save(savedir + '/p{}_energyspectrum'.format(p), energies)

def diracq3_generalcut(p, rangemin, rangemax, numsteps, klist, savedir):
    klist = diracq3_klist(klist)
    step_size = (rangemax-rangemin)/numsteps
    lam_vals = np.arange(rangemin, rangemax+step_size, step_size)
    energies = np.zeros((1,p+1))
    for n in range(len(lam_vals)):
        ham = diracq3_ham(p, lam_vals[n]*klist, savedir)
        eigvals = np.linalg.eigvalsh(ham)
        energies = np.vstack((energies, np.append(lam_vals[n],eigvals)))
    energies = energies[1:,:]
    np.save(savedir+'/p{}_energyspectrum'.format(p), energies)

def energy_spectrum_plot(energy_spectrum_file):
    energy_spectrum = np.load(energy_spectrum_file)
    num_energies_per_k = np.size(energy_spectrum,1)-1
    kvals = np.array([])
    energies = np.array([])
    for row in range(np.size(energy_spectrum,0)):
        kvals = np.append(kvals, np.repeat(energy_spectrum[row,:][0],num_energies_per_k))
        energies = np.append(energies, energy_spectrum[row,:][1:])
    plt.scatter(kvals, energies, s=10)
    plt.show()

def energy_spectrum_plot_multi(energy_spectrum_files):
    for esf in energy_spectrum_files:
        energy_spectrum = np.load(esf)
        num_energies_per_k = np.size(energy_spectrum, 1) - 1
        kvals = np.array([])
        energies = np.array([])
        for row in range(np.size(energy_spectrum, 0)):
            kvals = np.append(kvals, np.repeat(energy_spectrum[row, :][0], num_energies_per_k))
            energies = np.append(energies, energy_spectrum[row, :][1:])
        plt.scatter(kvals, energies, s=10)

# def gaussian_smoothing_OLD(E, eta, nk, dbloch, energy_list):
#     def gaussian(arg, eta):
#         return((1/(eta*np.sqrt(2*np.pi)))*np.exp(-(arg**2)/(2*(eta**2))))
#     # vec_gaussian = np.vectorize(gaussian)
#     sumterms = np.array([])
#     for j in range(nk*dbloch):
#         sumterms = np.append(sumterms, gaussian(E-energy_list[j],eta))
#     return((1/(nk*dbloch))*np.sum(sumterms))
def gaussian_smoothing(E, eta, nk, dbloch, energy_list):
    def gaussian(arg, eta):
        return((1/(eta*np.sqrt(2*np.pi)))*np.exp(-(arg**2)/(2*(eta**2))))
    vec_gaussian = np.vectorize(gaussian)
    # sumterms = np.array([])
    # for j in range(nk*dbloch):
    #     sumterms = np.append(sumterms, gaussian(E-energy_list[j],eta))
    return((1/(nk*dbloch))*np.sum(vec_gaussian(E-energy_list, eta)))

def diracq3_HBT_DOS(p, eta, sample_size, rangemin, rangemax, num_steps, savedir):
    energies = np.zeros((1,p))
    for s in range(sample_size):
        print('Iteration: {}'.format(s+1))
        random_ks = np.array([])
        for i in range(int((p/2)-1)):
            random_ks = np.append(random_ks, np.random.uniform(-np.pi,np.pi,1))
        klist = diracq3_klist(random_ks)
        ham = diracq3_ham(p,klist,savedir)
        eigvals = np.linalg.eigvalsh(ham)
        energies = np.vstack((energies, eigvals))
    energies = energies[1:,:]
    energy_list = energies.reshape(-1)
    DOS_vals = np.zeros((1,2))
    step_size = (rangemax-rangemin)/num_steps
    E_vals = np.arange(rangemin, rangemax+step_size, step_size)
    for E in E_vals:
        print('E: {}'.format(E))
        DOS_vals = np.vstack((DOS_vals,np.array([E, gaussian_smoothing(E, eta, sample_size, p, energy_list)])))
    DOS_vals = DOS_vals[1:,:]
    np.save(savedir+'/p{}_DOS_Energies'.format(p), energies)
    np.save(savedir+'/p{}_DOS_Data'.format(p), DOS_vals)

def diracq3_HBT_Energies_vectorized(p, sample_size, savedir):
    random_ks = list(np.random.uniform(-np.pi, np.pi, int(sample_size * (p / 2 - 1))).reshape(sample_size, int(p / 2 - 1)))
    dq3_klist_v = np.vectorize(diracq3_klist, signature='(n)->(m)')
    multi_ks_list = list(dq3_klist_v(random_ks))

    dq3_ham_v = np.vectorize(diracq3_ham, signature='(),(n),()->(m,m)')
    multi_ham_list = list(dq3_ham_v(p, multi_ks_list, savedir))

    eigvals_v = np.vectorize(np.linalg.eigvals, signature='(n,n)->(n)')
    eigvals_list = eigvals_v(multi_ham_list)

    energy_list = np.concatenate(eigvals_list)

    return(energy_list)

def diracq3_HBT_DOS_vectorized(p, eta, sample_size, rangemin, rangemax, num_steps, savedir):

    random_ks = list(np.random.uniform(-np.pi,np.pi,int(sample_size*(p/2 - 1))).reshape(sample_size,int(p/2 - 1)))
    dq3_klist_v = np.vectorize(diracq3_klist, signature='(n)->(m)')
    multi_ks_list = list(dq3_klist_v(random_ks))

    dq3_ham_v = np.vectorize(diracq3_ham, signature='(),(n),()->(m,m)')
    multi_ham_list = list(dq3_ham_v(p, multi_ks_list, savedir))

    eigvalsh_v = np.vectorize(np.linalg.eigvalsh, signature='(n,n)->(n)')
    eigvals_list = eigvalsh_v(multi_ham_list)

    energy_list = np.concatenate(eigvals_list)

    DOS_vals = np.zeros((1,2))
    step_size = (rangemax-rangemin)/num_steps
    E_vals = np.arange(rangemin, rangemax+step_size, step_size)
    for E in E_vals:
        print('E: {}'.format(E))
        DOS_vals = np.vstack((DOS_vals,np.array([E, gaussian_smoothing(E, eta, sample_size, p, energy_list)])))
    DOS_vals = DOS_vals[1:,:]

    np.save(savedir + '/p{}_DOS_Energies'.format(p), eigvals_list)
    np.save(savedir + '/p{}_DOS_Data'.format(p), DOS_vals)

def plot_DOS_data(DOS_data_file):
    raw_data = np.load(DOS_data_file)
    E_vals = raw_data[:,0]
    DOS_vals = raw_data[:,1]
    plt.plot(E_vals,DOS_vals)
    plt.scatter(E_vals,DOS_vals)
    plt.show()

def plot_multiple_DOS_data(DOS_data_files):
    for ddf in DOS_data_files:
        raw_data = np.load(ddf)
        E_vals = raw_data[:, 0]
        DOS_vals = raw_data[:, 1]
        plt.scatter(E_vals, DOS_vals)
    plt.show()

def DOS_from_energy_file(p, energy_file, eta, num_steps, savedir):
    rangemin=-3
    rangemax=3
    energies = np.load(energy_file)
    energy_list = energies.reshape(-1)
    DOS_vals = np.zeros((1, 2))
    step_size = (rangemax - rangemin) / num_steps
    E_vals = np.arange(rangemin, rangemax + step_size, step_size)
    sample_size = np.size(energies,0)
    for E in E_vals:
        print('E: {}'.format(E))
        DOS_vals = np.vstack((DOS_vals, np.array([E, gaussian_smoothing(E, eta, sample_size, p, energy_list)])))
    DOS_vals = DOS_vals[1:, :]
    np.save(savedir + '/p{}_eta{}_DOS_Data'.format(p,eta), DOS_vals)

#################################################
###Dirac q3 stuff but now with interaction###
#################################################

def diracq3_hdel(p,delta):
    hdel=np.zeros((p,p), dtype=np.complex_)
    for i in range(int(p/2)):
        hdel[i,i] = delta
    for i in range(int(p/2),p):
        hdel[i,i] = -1*delta
    return(hdel)

def diracq3_sublatticebasis(p):
    asites = np.array([],dtype=int)
    bsites = np.array([],dtype=int)
    for i in range(p):
        if i % 2 == 0:
            asites = np.append(asites, i)
        else:
            bsites = np.append(bsites, i)
    return(np.concatenate((asites,bsites)))

def diracq3_nonherm_ham(p,klist,alpha,savedir):
    h0oldbasis = diracq3_ham(p,klist,savedir)
    newbasis = diracq3_sublatticebasis(p)
    h0newbasis = h0oldbasis[newbasis,:][:,newbasis]
    hcdw = np.zeros((p,p),dtype=np.complex_)
    for i in range(int(p/2)):
        hcdw[i,i] = 1
    for i in range(int(p/2),p):
        hcdw[i,i] = -1
    return(h0newbasis + alpha*np.matmul(hcdw,h0newbasis))

def diracq3_nonherm_ham_interaction(p,klist,alpha,delta,savedir):
    hamnoint = diracq3_nonherm_ham(p,klist,alpha,savedir)
    hdel = diracq3_hdel(p,delta)
    return(hamnoint+hdel)


def diracq3_HBT_Energies_Interactions_vectorized(p, alpha, delta, sample_size, savedir):
    random_ks = list(np.random.uniform(-np.pi, np.pi, int(sample_size * (p / 2 - 1))).reshape(sample_size, int(p / 2 - 1)))
    dq3_klist_v = np.vectorize(diracq3_klist, signature='(n)->(m)')
    multi_ks_list = list(dq3_klist_v(random_ks))

    dq3_ham_v = np.vectorize(diracq3_nonherm_ham_interaction, signature='(),(n),(),(),()->(m,m)')
    multi_ham_list = list(dq3_ham_v(p, multi_ks_list, alpha, delta, savedir))

    eigvals_v = np.vectorize(np.linalg.eigvals, signature='(n,n)->(n)')
    eigvals_list = eigvals_v(multi_ham_list)

    energy_list = np.concatenate(eigvals_list)

    return (energy_list)

####################
###Specific Cases###
####################
def p8q4_chenetal(klist, savedir):
    t = 1
    ham_r1 = np.array([0, t*(1+np.exp(1j*(klist[0]-klist[1]))), 0, t*(np.exp(1j*klist[0])+np.exp(-1j*klist[3]))])
    ham_r2 = np.array([t*(1+np.exp(1j*(-klist[0]+klist[1]))), 0, t*(1+np.exp(1j*(klist[1]-klist[2]))), 0])
    ham_r3 = np.array([0, t*(1+np.exp(1j*(-klist[1]+klist[2]))), 0, t*(1+np.exp(1j*(klist[2]-klist[3])))])
    ham_r4 = np.array([t*(np.exp(-1j*klist[0])+np.exp(1j*klist[3])), 0, t*(1+np.exp(1j*(-klist[2]+klist[3]))), 0])
    ham = np.vstack((ham_r1,ham_r2,ham_r3,ham_r4))
    # np.save(savedir + '/Chenetal_p8q4_klist{}'.format(klist), ham)
    return(ham)

def p8q4_chenetal_generalcut(rangemin, rangemax, numsteps, klist, savedir):
    p=8
    step_size = (rangemax - rangemin) / numsteps
    lam_vals = np.arange(rangemin, rangemax + step_size, step_size)
    energies = np.zeros((1, int(p/2 + 1)))
    for n in range(len(lam_vals)):
        ham = p8q4_chenetal(lam_vals[n]*klist, savedir)
        eigvals = np.linalg.eigvalsh(ham)
        energies = np.vstack((energies, np.append(lam_vals[n], eigvals)))
    energies = energies[1:, :]
    np.save(savedir + '/p8q4_Chenetal_energyspectrum', energies)

def p8q4_chenetal_HBT_DOS(eta, sample_size, rangemin, rangemax, num_steps, savedir):
    p=8
    energies = np.zeros((1,int(p/2)))
    for s in range(sample_size):
        print('Iteration: {}'.format(s+1))
        random_ks = np.array([])
        for i in range(int((p/2))):
            random_ks = np.append(random_ks, np.random.uniform(-np.pi,np.pi,1))
        klist = random_ks
        ham = p8q4_chenetal(klist,savedir)
        eigvals = np.linalg.eigvalsh(ham)
        energies = np.vstack((energies, eigvals))
    energies = energies[1:,:]
    energy_list = energies.reshape(-1)
    DOS_vals = np.zeros((1,2))
    step_size = (rangemax-rangemin)/num_steps
    E_vals = np.arange(rangemin, rangemax+step_size, step_size)
    for E in E_vals:
        print('E: {}'.format(E))
        DOS_vals = np.vstack((DOS_vals,np.array([E, gaussian_smoothing(E, eta, sample_size, 4, energy_list)])))
    DOS_vals = DOS_vals[1:,:]
    np.save(savedir+'/p{}_DOS_Energies'.format(p), energies)
    np.save(savedir+'/p{}_DOS_Data'.format(p), DOS_vals)

def p8q4_chenetal_HBT_DOS_vectorized(eta, sample_size, rangemin, rangemax, num_steps, savedir):

    random_ks = list(np.random.uniform(-np.pi, np.pi, int(sample_size * 4)).reshape(sample_size, 4))
    multi_ks_list = random_ks

    p8q4_chenetal_ham_v = np.vectorize(p8q4_chenetal, signature='(n),()->(m,m)')
    multi_ham_list = list(p8q4_chenetal_ham_v(multi_ks_list, savedir))

    eigvalsh_v = np.vectorize(np.linalg.eigvalsh, signature='(n,n)->(n)')
    eigvals_list = eigvalsh_v(multi_ham_list)

    energy_list = np.concatenate(eigvals_list)

    DOS_vals = np.zeros((1, 2))
    step_size = (rangemax - rangemin) / num_steps
    E_vals = np.arange(rangemin, rangemax + step_size, step_size)
    for E in E_vals:
        print('E: {}'.format(E))
        DOS_vals = np.vstack((DOS_vals, np.array([E, gaussian_smoothing(E, eta, sample_size, 4, energy_list)])))
    DOS_vals = DOS_vals[1:, :]

    p=8
    np.save(savedir + '/p{}_DOS_Energies'.format(p), eigvals_list)
    np.save(savedir + '/p{}_DOS_Data'.format(p), DOS_vals)

def p10q5_chenetal(klist, savedir):
    t=1
    def beta(klist):
        return(1+np.exp(1j*(klist[0]-klist[1]))+np.exp(1j*(klist[1]-klist[2]))+np.exp(1j*(klist[0]-klist[2]+klist[3]))+np.exp(1j*(klist[1]-klist[2]-klist[2]+klist[3])))
    ham = np.array([[0, t*beta(klist)],[t*np.conj(beta(klist)), 0]])
    # np.save(savedir + '/p10q5_klist{}_chenetal'.format(klist), ham)
    return(ham)

def p10q5_chenetal_generalcut(rangemin, rangemax, numsteps, klist, savedir):
    step_size = (rangemax - rangemin) / numsteps
    lam_vals = np.arange(rangemin, rangemax + step_size, step_size)
    energies = np.zeros((1, int(2+1)))
    for n in range(len(lam_vals)):
        ham = p10q5_chenetal(lam_vals[n]*klist, savedir)
        eigvals = np.linalg.eigvalsh(ham)
        energies = np.vstack((energies, np.append(lam_vals[n], eigvals)))
    energies = energies[1:, :]
    np.save(savedir + '/p10q5_Chenetal_energyspectrum', energies)

def p10q5_chenetal_HBT_DOS(eta, sample_size, rangemin, rangemax, num_steps, savedir):
    energies = np.zeros((1,2))
    for s in range(sample_size):
        print('Iteration: {}'.format(s+1))
        random_ks = np.array([])
        for i in range(4):
            random_ks = np.append(random_ks, np.random.uniform(-np.pi,np.pi,1))
        klist = random_ks
        ham = p10q5_chenetal(klist,savedir)
        eigvals = np.linalg.eigvalsh(ham)
        energies = np.vstack((energies, eigvals))
    energies = energies[1:,:]
    energy_list = energies.reshape(-1)
    DOS_vals = np.zeros((1,2))
    step_size = (rangemax-rangemin)/num_steps
    E_vals = np.arange(rangemin, rangemax+step_size, step_size)
    for E in E_vals:
        print('E: {}'.format(E))
        DOS_vals = np.vstack((DOS_vals,np.array([E, gaussian_smoothing(E, eta, sample_size, 2, energy_list)])))
    DOS_vals = DOS_vals[1:,:]
    p=10
    np.save(savedir+'/p{}_DOS_Energies'.format(p), energies)
    np.save(savedir+'/p{}_DOS_Data'.format(p), DOS_vals)

def p10q5_chenetal_HBT_DOS_vectorized(eta, sample_size, rangemin, rangemax, num_steps, savedir):

    random_ks = list(np.random.uniform(-np.pi, np.pi, int(sample_size * 4)).reshape(sample_size, 4))
    multi_ks_list = random_ks

    p10q5_chenetal_ham_v = np.vectorize(p10q5_chenetal, signature='(n),()->(m,m)')
    multi_ham_list = list(p10q5_chenetal_ham_v(multi_ks_list, savedir))

    eigvalsh_v = np.vectorize(np.linalg.eigvalsh, signature='(n,n)->(n)')
    eigvals_list = eigvalsh_v(multi_ham_list)

    energy_list = np.concatenate(eigvals_list)

    DOS_vals = np.zeros((1, 2))
    step_size = (rangemax - rangemin) / num_steps
    E_vals = np.arange(rangemin, rangemax + step_size, step_size)
    for E in E_vals:
        print('E: {}'.format(E))
        DOS_vals = np.vstack((DOS_vals, np.array([E, gaussian_smoothing(E, eta, sample_size, 2, energy_list)])))
    DOS_vals = DOS_vals[1:, :]

    p=10
    np.save(savedir + '/p{}_DOS_Energies'.format(p), eigvals_list)
    np.save(savedir + '/p{}_DOS_Data'.format(p), DOS_vals)