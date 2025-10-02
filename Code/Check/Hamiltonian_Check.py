import numpy as np
from Fundamental.NonHermitian_Hamiltonian import site_assignment
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian
from Fundamental.NonHermitian_Hamiltonian import inter_sublattice_block
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb
from Fundamental.NonHermitian_Hamiltonian import inter_sublattice_block_honeycomb
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb_periodic_boundary
from Fundamental.NonHermitian_Hamiltonian import inter_sublattice_block_honeycomb_periodic_boundary
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_PeierlsSub_Hamiltonian
from Fundamental.NonHermitian_Hamiltonian import inter_sublattice_block_PeierlsSub

def interactive_hamiltonian_spot_check(ham):
    still_running = True
    while still_running == True:
        checkindex = input('What site to check?\n')
        checkindex = int(checkindex)
        print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
        keeprunquestion = input('Check another index (y/n)?\n')
        if keeprunquestion == 'n': still_running = False

def interactive_nonhermitian_hamiltonian_spot_check(nonhermham,p,q,n,a,hermham):
    still_running = True
    asites_ogind, bsites_ogind = site_assignment(p,q,n,hermham)
    asites_ogind = asites_ogind + 1
    bsites_ogind = bsites_ogind + 1
    asites_newind = np.linspace(1,int(np.size(nonhermham,0)/2),int(np.size(nonhermham,0)/2))
    bsites_newind = np.linspace(int(np.size(nonhermham,0)/2 + 1),int(np.size(nonhermham,0)),int(np.size(nonhermham,0)/2))
    def get_og_index(new_index):
        og_ind = np.array([])
        for n in new_index:
            if n in asites_newind:
                og_ind = np.append(og_ind, asites_ogind[np.where(n == asites_newind)[0]])
            if n in bsites_newind:
                og_ind = np.append(og_ind, bsites_ogind[np.where(n == bsites_newind)[0]])
        return(og_ind)
    def get_new_index(og_index):
        if og_index in asites_ogind:
            n_ind = asites_newind[np.where(og_index == asites_ogind)[0]]
        if og_index in bsites_ogind:
            n_ind = bsites_newind[np.where(og_index == bsites_ogind)[0]]
        return(n_ind)
    while still_running == True:
        checkindex = input('What site to check?\n')
        checkindex = get_new_index(int(checkindex))
        print('Connections are to:\n {}'.format(get_og_index(np.where(nonhermham[:,int(checkindex-1)]!=0)[0]+1)))
        keeprunquestion = input('Check another index (y/n)?\n')
        if keeprunquestion == 'n': still_running = False
    print('hi')

def generate_hamiltonian_check_doc(ham,p,q,n):
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Hermitian/p{}q{}n{}_CheckDoc.txt'.format(p, q, n)
    f = open(newfile, 'w')
    for n in range(1,np.size(ham,0)+1):
        checkindex = n
        checks = np.append(checks,'{} {} \n'.format(n, np.where(ham[:,checkindex-1]!=0)[0]+1))
        # print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
    f.writelines(checks)
    f.close()

def generate_nonhermhamiltonian_check_doc(nonhermham,p,q,n,a,hermham):
    asites_ogind, bsites_ogind = site_assignment(p, q, n, hermham)
    asites_ogind = asites_ogind + 1
    bsites_ogind = bsites_ogind + 1
    asites_newind = np.linspace(1, int(np.size(nonhermham, 0) / 2), int(np.size(nonhermham, 0) / 2))
    bsites_newind = np.linspace(int(np.size(nonhermham, 0) / 2 + 1), int(np.size(nonhermham, 0)), int(np.size(nonhermham, 0) / 2))
    def get_og_index(new_index):
        og_ind = np.array([])
        for n in new_index:
            if n in asites_newind:
                og_ind = np.append(og_ind, asites_ogind[np.where(n == asites_newind)[0]])
            if n in bsites_newind:
                og_ind = np.append(og_ind, bsites_ogind[np.where(n == bsites_newind)[0]])
        return (og_ind)
    def get_new_index(og_index):
        if og_index in asites_ogind:
            n_ind = asites_newind[np.where(og_index == asites_ogind)[0]]
        if og_index in bsites_ogind:
            n_ind = bsites_newind[np.where(og_index == bsites_ogind)[0]]
        return (n_ind)
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/NonHermitian/p{}q{}n{}a{}_CheckDoc.txt'.format(p, q, n, a)
    f = open(newfile, 'w')
    for n in range(1,np.size(nonhermham,0)+1):
        checkindex = get_new_index(n)
        checks = np.append(checks,'{} {} \n'.format(n, (get_og_index(np.where(nonhermham[:, int(checkindex - 1)] != 0)[0] + 1)).astype(int)))
        # print('Connections are to:\n {}'.format(get_og_index(np.where(nonhermham[:, int(checkindex - 1)] != 0)[0] + 1)))
    f.writelines(checks)
    f.close()

def compare_generated_check_docs(f1address, f2address):
    f1 = open(f1address,'r')
    f2 = open(f2address,'r')
    f1_data = f1.read()
    f2_data = f2.read()
    print('Are both files the same? {}'.format(f1_data == f2_data))

def random_site_check_hermham(ham,nsamps):
    prev_checkindex = np.array([])
    for n in range(nsamps):
        checkindex = np.random.randint(1,np.size(ham,0)+1)
        checkindex_is_unique = False
        while checkindex_is_unique == False:
            if checkindex not in prev_checkindex:
                checkindex_is_unique = True
            else:
                checkindex = np.random.randint(1,np.size(ham,0)+1)
        prev_checkindex = np.append(prev_checkindex, checkindex)
        print('Connections to {} are: {} \n'.format(checkindex, np.where(ham[:, checkindex - 1] != 0)[0] + 1))

def random_site_check_nonhermham(nonhermham,p,q,n,a,hermham,nsamps):
    asites_ogind, bsites_ogind = site_assignment(p,q,n,hermham)
    asites_ogind = asites_ogind + 1
    bsites_ogind = bsites_ogind + 1
    asites_newind = np.linspace(1,int(np.size(nonhermham,0)/2),int(np.size(nonhermham,0)/2))
    bsites_newind = np.linspace(int(np.size(nonhermham,0)/2 + 1),int(np.size(nonhermham,0)),int(np.size(nonhermham,0)/2))
    def get_og_index(new_index):
        og_ind = np.array([])
        for n in new_index:
            if n in asites_newind:
                og_ind = np.append(og_ind, asites_ogind[np.where(n == asites_newind)[0]])
            if n in bsites_newind:
                og_ind = np.append(og_ind, bsites_ogind[np.where(n == bsites_newind)[0]])
        return(og_ind)
    def get_new_index(og_index):
        if og_index in asites_ogind:
            n_ind = asites_newind[np.where(og_index == asites_ogind)[0]]
        if og_index in bsites_ogind:
            n_ind = bsites_newind[np.where(og_index == bsites_ogind)[0]]
        return(n_ind)
    prev_checkindex = np.array([])
    for n in range(nsamps):
        checkindex = np.random.randint(1,np.size(nonhermham,0)+1)
        checkindex_is_unique = False
        while checkindex_is_unique == False:
            if checkindex not in prev_checkindex:
                checkindex_is_unique = True
            else:
                checkindex = np.random.randint(1, np.size(nonhermham, 0) + 1)
        prev_checkindex = np.append(prev_checkindex, checkindex)
        print('Connections to {} are: {} \n'.format(get_og_index([checkindex]), get_og_index(np.where(nonhermham[:,int(checkindex-1)]!=0)[0]+1)))

def NonHermitian_algebraic_equivalence_check(p,q,n,a):
    original_nhh = NonHermitian_Hamiltonian(p,q,n,a,1)
    tblock = inter_sublattice_block(p,q,n,1)[0]
    zeroblock = np.zeros((np.size(tblock,0),np.size(tblock,1)))
    algebraic_equal = np.block([[zeroblock, (1+a)*tblock],[(1-a)*np.transpose(tblock), zeroblock]])
    print('=================================== \n')
    print('Are the two equal? {}'.format(np.all(original_nhh == algebraic_equal)==True))

def NonHermitian_Honeycomb_algebraic_equivalence_check(nl, a):
    original_nhh = NonHermitian_Honeycomb(nl, 1, a)
    tblock = inter_sublattice_block_honeycomb(nl, 1)[0]
    zeroblock = np.zeros((np.size(tblock, 0), np.size(tblock, 1)))
    algebraic_equal = np.block([[zeroblock, (1 + a) * tblock], [(1 - a) * np.transpose(tblock), zeroblock]])
    print('=================================== \n')
    print('Are the two equal? {}'.format(np.all(original_nhh == algebraic_equal) == True))

def coordination_number_check(ham):
    num_connect = np.array([])
    for n in range(np.size(ham,0)):
        num_connect = np.append(num_connect, len(np.where(ham[n,:] != 0)[0]))
    print('Number of possible connections to a point: {}'.format(np.unique(num_connect)))

def NonHermitian_Honeycomb_PBC_algebraic_equivalence_check(nl, a):
    original_nhh = NonHermitian_Honeycomb_periodic_boundary(nl, 1, a)
    tblock = inter_sublattice_block_honeycomb_periodic_boundary(nl, 1)[0]
    zeroblock = np.zeros((np.size(tblock, 0), np.size(tblock, 1)))
    algebraic_equal = np.block([[zeroblock, (1 + a) * tblock], [(1 - a) * np.transpose(tblock), zeroblock]])
    print('=================================== \n')
    print('Are the two equal? {}'.format(np.all(original_nhh == algebraic_equal) == True))

def NonHermitian_PeierlsSub_algebraic_equivalence_check(p,q,n,alpha,t,alphamag):
    original_nhh = NonHermitian_PeierlsSub_Hamiltonian(p,q,n,alpha,t,alphamag)
    tblock = inter_sublattice_block_PeierlsSub(p,q,n,t,alphamag)[0]
    zeroblock = np.zeros((np.size(tblock, 0), np.size(tblock, 1)))
    algebraic_equal = np.block([[zeroblock, (1 + alpha) * tblock], [(1 - alpha) * np.transpose(tblock), zeroblock]])
    print('=================================== \n')
    print('Are the two equal? {}'.format(np.all(np.around(original_nhh,10) == np.around(algebraic_equal,10)) == True))

