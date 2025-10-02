import numpy as np

def generate_square_hamiltonian_check_doc(ham, nl):
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Square/nl{}_squareCheckDoc.txt'.format(nl)
    f = open(newfile, 'w')
    for n in range(1,np.size(ham,0)+1):
        checkindex = n
        checks = np.append(checks,'{} {} \n'.format(n, np.where(ham[:,checkindex-1]!=0)[0]+1))
        # print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
    f.writelines(checks)
    f.close()

def generate_square_hamiltonian_peierlssub_check_doc(ham, nl, amag):
    t = 1
    ps_xvals = np.array([])
    ps_tvals = np.array([])
    for nlev in range(nl):
        if nlev == 0:
            ps_xvals = np.append(ps_xvals, 0)  # Any choice for hopping on first level; just pick t1=t
            ps_tvals = np.append(ps_tvals, t)
        else:
            ps_xvals = np.append(ps_xvals, ps_xvals[-1] - amag)  # Pattern is alpha = x_n - x_{n+1}
            ps_tvals = np.append(ps_tvals, t * np.exp(1j * ps_xvals[-1]))
    ps_tvals_numbers = np.concatenate((ps_tvals, np.conjugate(ps_tvals)))
    ps_tvals_words_part1 = np.array([])
    ps_tvals_words_part2 = np.array([])
    for i in range(1,nl):
        ps_tvals_words_part1 = np.append(ps_tvals_words_part1, 't{}'.format(i+1))
        ps_tvals_words_part2 = np.append(ps_tvals_words_part2, 't{}_conj'.format(i+1))
    ps_tvals_words_part1 = np.concatenate((np.array(['t']), ps_tvals_words_part1))
    ps_tvals_words_part2 = np.concatenate((np.array(['t']), ps_tvals_words_part2))
    ps_tvals_words = np.concatenate((ps_tvals_words_part1, ps_tvals_words_part2))
    connections_in_words = np.array([])
    for row in range(np.size(ham,0)):
        nonzerovalues = ham[row, np.where(ham[row,:] != 0)[0]]
        thisrowconns = np.array([])
        for nzv in nonzerovalues:
            matchloc = np.where(np.around(nzv,12)==np.around(ps_tvals_numbers,12))[0]
            matchword = np.unique(ps_tvals_words[matchloc])
            thisrowconns = np.append(thisrowconns, matchword)
        connections_in_words = np.append(connections_in_words, '{} {}\n'.format(row+1, thisrowconns))
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Square/nl{}amag{}_square_peierlssub_conjugatespecific_CheckDoc.txt'.format(nl,amag)
    f = open(newfile, 'w')
    f.writelines(connections_in_words)
    f.close()

from Fundamental.Square_Lattice import square_site_assignment
from Fundamental.Square_Lattice import number_square_points
def nonherm_h0block(ham, nl):
    totnumpoints = number_square_points(nl)
    interblock = np.zeros((int(totnumpoints/2), int(totnumpoints/2)), dtype=np.complex_)
    asites, bsites = square_site_assignment(nl)
    for row in range(np.size(ham,0)):
        nonzerolocs = np.where(ham[row,:]!=0)[0]
        if (row in asites) == True:
            newrow = np.where(row==asites)[0]
            newcol = [np.where(m==bsites)[0][0] for m in nonzerolocs]
            for nc in range(len(newcol)):
                interblock[newrow, newcol[nc]] = ham[row, nonzerolocs[nc]]
    return(interblock)
def nonherm_alternate_construct_check(ham, nl, interblock, alpha):
    totnumpoints = number_square_points(nl)
    zeroblock = np.zeros((int(totnumpoints/2), int(totnumpoints/2)))
    h0 = np.block([[zeroblock, interblock],[np.conjugate(np.transpose(interblock)), zeroblock]])
    hcdw = np.block([[np.eye(int(totnumpoints/2)), zeroblock],[zeroblock, -1*np.eye(int(totnumpoints/2))]])
    alt_nonherm_ham = h0 + alpha*np.matmul(hcdw,h0)
    print('Does the alternate construction match? {}'.format(np.all((np.around(ham,8)==np.around(alt_nonherm_ham,8))==True)))

def alternate_site_assign(nl, ham):
    asites = [i for i in range(nl) if i%2==0]
    bsites = [i for i in range(nl) if i%2==1]
    while len(asites) != (nl**2)/2 or len(bsites) != (nl**2)/2:
        for a in asites:
            bsites = np.append(bsites, np.where(ham[a,:]!=0)[0])
            bsites = np.unique(bsites)
        for b in bsites:
            asites = np.append(asites, np.where(ham[b, :] != 0)[0])
            asites = np.unique(asites)
    return(asites, bsites)

def generate_square_pbc_hamiltonian_check_doc(ham, nl):
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Square/nl{}_squarePBCCheckDoc.txt'.format(nl)
    f = open(newfile, 'w')
    for n in range(1, np.size(ham, 0) + 1):
        checkindex = n
        checks = np.append(checks, '{} {} \n'.format(n, np.where(ham[:, checkindex - 1] != 0)[0] + 1))
        # print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
    f.writelines(checks)
    f.close()
