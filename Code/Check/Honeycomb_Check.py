import numpy as np
from Fundamental.NonHermitian_Hamiltonian import inter_sublattice_block_honeycomb_periodic_boundary
from Fundamental.Honeycomb_Lattice import honeycomb_lattice_periodic_boundary
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb_periodic_boundary
from Fundamental.Honeycomb_Lattice import honeycomb_points

def generate_honeycomb_check_doc(ham,nl):
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Honeycomb/nl{}_CheckDoc.txt'.format(nl)
    f = open(newfile, 'w')
    for n in range(1, np.size(ham, 0) + 1):
        checkindex = n
        checks = np.append(checks, '{} {} \n'.format(n, np.where(ham[:, checkindex - 1] != 0)[0] + 1))
        # print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
    f.writelines(checks)
    f.close()

def generate_honeycomb_periodicboundary_check_doc(ham,nl):
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Honeycomb/nl{}_PBC_CheckDoc.txt'.format(nl)
    f = open(newfile, 'w')
    for n in range(1, np.size(ham, 0) + 1):
        checkindex = n
        checks = np.append(checks, '{} {} \n'.format(n, np.where(ham[:, checkindex - 1] != 0)[0] + 1))
        # print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
    f.writelines(checks)
    f.close()

def NonHermitian_Honeycomb_PBC_algebraic_equivalence(nl, alpha, ham):
    tblock, foo = inter_sublattice_block_honeycomb_periodic_boundary(nl, 1)
    zeroblock = np.zeros((np.size(tblock,0),np.size(tblock,1)))
    equiv_mat = np.block([[zeroblock, (1+alpha)*tblock],[(1-alpha)*np.transpose(tblock), zeroblock]])
    print('Does NonHermitian Hamiltonian and algebraic equivalent construction match: {}'.format(np.all((ham==equiv_mat)==True)))

def NonHermitian_Honeycomb_PBC_from_Hermitian_basis_switch(nl):
    #Check where I check against a more elegant basis switch method of getting NonHermitian Matrix
    hermham = honeycomb_lattice_periodic_boundary(nl)
    a_sites, b_sites = site_assignment_honeycomb(nl, hermham)
    basis_key = np.concatenate((a_sites,b_sites)).astype(int)
    nonhermham = hermham[:,basis_key]
    nonhermham = nonhermham[basis_key,:]
    og_nonhermham = NonHermitian_Honeycomb_periodic_boundary(nl,1,0)
    print('Are both methods equivalent: {}'.format(np.all((nonhermham==og_nonhermham)==True)))

def site_assignment_Honeycomb_switch_check(nl):
    #It turns out that site assignment difference pairs should switch sign from generation to generation
    #I utilize this fact here to test site assignment code
    ham = honeycomb_lattice_periodic_boundary(nl)
    asites, bsites = site_assignment_honeycomb(nl, ham)
    difference_pairs = bsites-asites
    switch_locs = np.array([])
    for i in range(len(difference_pairs)-1):
        if difference_pairs[i] - difference_pairs[i+1] != 0:
            switch_locs = np.append(switch_locs, i)
    switch_locs = switch_locs.astype(int)
    print(np.unique(difference_pairs[switch_locs]-difference_pairs[switch_locs+1]))
    print(2*(switch_locs+1))
    points_perlevel, totalnum_points = honeycomb_points(nl)
    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_perlevel)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[0]))
        elif a == len(points_perlevel) - 1:
            point_cumulative = np.append(point_cumulative, int(totalnum_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[a] + point_cumulative[-1]))
    print(point_cumulative)

def generate_bilayer_honeycomb_check_doc(ham, nl):
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Bilayer_Honeycomb/nl{}Bilayer_CheckDoc.txt'.format(nl)
    f = open(newfile, 'w')
    for n in range(1, np.size(ham, 0) + 1):
        checkindex = n
        checks = np.append(checks, '{} {} \n'.format(n, np.where(ham[:, checkindex - 1] != 0)[0] + 1))
        # print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
    f.writelines(checks)
    f.close()

from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Honeycomb
from Fundamental.Honeycomb_Lattice import honeycomb_lattice
def bilayer_honeycomb_nonhermitian_check(nonhermham, ham, nl, alpha):
    onelayer = NonHermitian_Honeycomb(nl,1,alpha)
    reverseorder = np.concatenate((range(int(np.size(onelayer,0)/2), np.size(onelayer,0)), range(int(np.size(onelayer,0)/2))))
    onelayer_inv = onelayer[:,reverseorder][reverseorder,:]
    block1 = nonhermham[:int(np.size(nonhermham,0)/2), :int(np.size(nonhermham,0)/2)]
    block2 = nonhermham[:int(np.size(nonhermham,0)/2), int(np.size(nonhermham,0)/2):]
    block3 = nonhermham[int(np.size(nonhermham,0)/2):, :int(np.size(nonhermham,0)/2)]
    block4 = nonhermham[int(np.size(nonhermham,0)/2):, int(np.size(nonhermham,0)/2):]
    print('Is block1 good: {}'.format(np.all(block1 == onelayer)))
    print('Is block4 good: {}'.format(np.all(block4 == onelayer_inv)))

    block2block3check = np.zeros((int(np.size(ham,0)/2),int(np.size(ham,0)/2)))
    for row in range(int(np.size(ham,0)/2)):
        connections = np.where(ham[row,:]!=0)[0]
        otherlayer_connections = connections[np.where(connections>=int(np.size(ham,0)/2))[0]]
        block2block3check[row, otherlayer_connections-int(np.size(ham,0)/2)] = 1
    asites, bsites = site_assignment_honeycomb(nl, honeycomb_lattice(nl))
    changebasis = np.concatenate((asites, bsites))
    changebasis_otherlayer = np.concatenate((bsites, asites))
    changebasis = np.array([int(i) for i in changebasis])
    changebasis_otherlayer = np.array([int(i) for i in changebasis_otherlayer])
    block2block3check_block2 = block2block3check[changebasis,:][:,changebasis_otherlayer]
    block2block3check_block3 = np.transpose(block2block3check)[:, changebasis][changebasis_otherlayer, :]
    print('Is block2 good: {}'.format(np.all(block2 == (1+alpha)*block2block3check_block2)))
    print('Is block3 good: {}'.format(np.all(block3 == (1-alpha)*block2block3check_block3)))

def generate_bilayer_honeycomb_PBC_check_doc(ham, nl):
    checks = np.array([])
    newfile = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/Hamiltonian_Check_Docs/Bilayer_Honeycomb/nl{}BilayerPBC_CheckDoc.txt'.format(nl)
    f = open(newfile, 'w')
    for n in range(1, np.size(ham, 0) + 1):
        checkindex = n
        checks = np.append(checks, '{} {} \n'.format(n, np.where(ham[:, checkindex - 1] != 0)[0] + 1))
        # print('Connections are to:\n {}'.format(np.where(ham[:,checkindex-1]!=0)[0]+1))
    f.writelines(checks)
    f.close()