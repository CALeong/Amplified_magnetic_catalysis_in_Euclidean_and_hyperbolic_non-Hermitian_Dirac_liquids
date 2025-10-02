import numpy as np
from Fundamental.NonHermitian_Hamiltonian import site_assignment
from Fundamental.NonHermitian_Hamiltonian import inter_sublattice_block
from Fundamental.Hamiltonian import H0

def site_assignment_check_output(p,q,n,ham):
    a_sites, b_sites = site_assignment(p,q,n,ham)
    return(b_sites - a_sites)

def site_assignment_check_values(p,q,n,checklist):
    print('The unique values in check list are: {}'.format(np.unique(checklist)))
    switch_locations = np.array([])
    for i in range(len(checklist)-1):
        test_seq = np.array([checklist[i],checklist[i+1]])
        if np.all((test_seq == np.array([1,-1]))==True) or np.all((test_seq == np.array([-1,1]))==True):
            switch_locations = np.append(switch_locations,'pair {} into pair {}'.format(i+1,i+2))
    file_address = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/' \
                   'Hamiltonian_Check_Docs/NonHermitian_Specific_Checks/Site_Assignment_Switching_Locations/' +\
                   'p{}q{}n{}'.format(p,q,n)
    file = open(file_address,'w')
    print('Switches happen at:\n')
    switches_in_words = np.array([])
    for sl in switch_locations:
        print(sl+'\n')
        switches_in_words = np.append(switches_in_words, sl+'\n')
    file.writelines(switches_in_words)


def get_new_index(old_index,p,q,n,ham):
    a_sites, b_sites = site_assignment(p,q,n,ham)
    for a in range(len(a_sites)):
        if old_index == a_sites[a]:
            new_index = a
    for b in range(len(b_sites)):
        if old_index == b_sites[b]:
            new_index = b
    return(new_index)

def get_old_index_asite(new_index_asite,p,q,n,ham):
    a_sites, b_sites = site_assignment(p,q,n,ham)
    old_index_asite = a_sites[new_index_asite]
    return(old_index_asite)

def get_old_index_bsite(new_index_bsite,p,q,n,ham):
    a_sites, b_sites = site_assignment(p,q,n,ham)
    old_index_bsite = b_sites[new_index_bsite]
    return(old_index_bsite)
#
# def inter_sublattice_check(p,q,n):
#     ham = H0(p,q,n)
#     block, foo = inter_sublattice_block(p,q,n,1)
#     arow = np.array([])
#     bcol = np.array([])
#     for row in range(np.size(block,0)):
#         connections_newind = np.where(block[row,:]!=0)[0]
#         arow = np.append(arow,get_old_index_asite(row,p,q,n))
#         connection_oldind = np.array([])
#         for c in connections_newind:
#             connection_oldind = np.append(connection_oldind, get_old_index_bsite(c,p,q,n,ham))
#         bcol = np.vstack((bcol,connection_oldind))
#
# def inter_sublattice_checkp(p,q,n):
#     block, foo = inter_sublattice_block(p,q,n,1)

def inter_sublattice_block_honeycomb_PeierlsSub(nl, alphamag):

    from Fundamental.Honeycomb_Lattice import honeycomb_PeierlsSubstitution
    from Fundamental.Honeycomb_Lattice import honeycomb_points
    from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb
    points_perlevel, totalnum_points = honeycomb_points(nl)
    ham = honeycomb_PeierlsSubstitution(nl, alphamag)
    a_sites, b_sites = site_assignment_honeycomb(nl, ham)

    h0block1 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)), dtype=np.complex_)
    h0block2 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)), dtype=np.complex_)

    # b_site_connections = np.array([])
    for a in range(len(a_sites)):
        connections = np.where(ham[int(a_sites[a]),:] != 0)[0]
        for i in connections:
            h0block1[a][np.where(i == b_sites)[0]] = ham[int(a_sites[a]), int(i)]
    for b in range(len(b_sites)):
        connections = np.where(ham[int(b_sites[b]), :] != 0)[0]
        for i in connections:
            h0block2[b][np.where(i == a_sites)[0]] = ham[int(b_sites[b]), int(i)]
    return(h0block1, h0block2)

def NonHermitian_PeierlsSub_Honeycomb_Check(givenham, nl, alphamag, alpha):
    #In original code programmed it elegantly using chnage of basis
    #Here I copy and modify interblock code for hyperbolic and adpat it to honeycomb
    #And check if this clunkier construction matces with what I have in my actual code
    from Fundamental.Honeycomb_Lattice import honeycomb_points
    h0block1 = inter_sublattice_block_honeycomb_PeierlsSub(nl, alphamag)[0]
    h0block2 = np.zeros((int(honeycomb_points(nl)[1]/2),int(honeycomb_points(nl)[1]/2)))
    h0 = np.block([[h0block2,h0block1],[np.transpose(np.conjugate(h0block1)),h0block2]])
    hcdw = np.eye(int(honeycomb_points(nl)[1]))
    for row in range(np.size(hcdw,0)):
        if row >= honeycomb_points(nl)[1]/2:
            hcdw[row,row] = -1
    nonhermham = h0 + alpha*np.matmul(hcdw,h0)
    print('Do they match: {}'.format(np.all((givenham==nonhermham)==True)))
