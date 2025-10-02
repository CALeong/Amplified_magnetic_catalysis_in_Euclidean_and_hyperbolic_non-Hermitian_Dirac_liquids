import numpy as np
from Fundamental.NonHermitian_Hamiltonian import inter_sublattice_block_PeierlsSub
from Fundamental.NonHermitian_Hamiltonian import site_assignment

def Peierls_Sub_to_letters(p,q,n,alpha,ham):
    if q == 3:
        t = 1

        x1 = alpha / p
        tn1 = t * np.exp(1j * x1)
        x2 = (alpha + x1) / (p - 3)
        tn2 = t * np.exp(1j * x2)
        x3 = (alpha + x2) / (p - 3)
        x4 = (alpha + 2 * x2) / (p - 4)
        tn3 = t * np.exp(1j * x3)
        tn4 = t * np.exp(1j * x4)
        x5 = (alpha + x3) / (p - 3)
        x6 = (alpha + x4) / (p - 3)
        x7 = (alpha + 2 * x3) / (p - 4)
        x8 = (alpha + x3 + x4) / (p - 4)
        tn5 = t * np.exp(1j * x5)
        tn6 = t * np.exp(1j * x6)
        tn7 = t * np.exp(1j * x7)
        tn8 = t * np.exp(1j * x8)
        x9 = (alpha + x5) / (p - 3)
        x10 = (alpha + x7) / (p - 3)
        x11 = (alpha + 2 * x5) / (p - 4)
        x12 = (alpha + x5 + x7) / (p - 4)
        x13 = (alpha + x6 + x8) / (p - 4)
        x14 = (alpha + x8) / (p - 3)
        x15 = (alpha + x6) / (p - 3)
        x16 = (alpha + x5 + x8) / (p - 4)
        x17 = (alpha + 2 * x6) / (p - 4)
        tn9 = t * np.exp(1j * x9)
        tn10 = t * np.exp(1j * x10)
        tn11 = t * np.exp(1j * x11)
        tn12 = t * np.exp(1j * x12)
        tn13 = t * np.exp(1j * x13)
        tn14 = t * np.exp(1j * x14)
        tn15 = t * np.exp(1j * x15)
        tn16 = t * np.exp(1j * x16)
        tn17 = t * np.exp(1j * x17)

        val_letter = np.array([1+0j,tn1,tn2,tn3,tn4,tn5,tn6,tn7,tn8,tn9,tn10,tn11,tn12,tn13,tn14,
                               tn15,tn16,tn17])
        letter_val = np.array(['t','tn1','tn2','tn3','tn4','tn5','tn6','tn7','tn8','tn9','tn10','tn11','tn12','tn13','tn14',
                               'tn15','tn16','tn17'])

        connections = np.array([])
        for row in range(np.size(ham,0)):
            val_connects = np.array([])
            for val in ham[row,:]:
                if val != 0:
                    # print(val)
                    val_connects = np.append(val_connects, letter_val[np.where(val==val_letter)[0] or np.where(val==np.conj(val_letter))[0]])
            connections = np.append(connections, '{} {} \n'.format(row+1, val_connects))
            print(row+1, val_connects, '\n')
        file_address = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/' \
                       'Hamiltonian_Check_Docs/Hermitian_PeierlsSub/p{}q{}n{}'.format(p,q,n)
        file = open(file_address, 'w')
        file.writelines(connections)
        file.close()

def Peierls_Sub_to_letters_conjugate_specific(p,q,n,alpha,ham):
    if q == 3:
        t = 1

        x1 = alpha / p
        tn1 = t * np.exp(1j * x1)
        x2 = (alpha + x1) / (p - 3)
        tn2 = t * np.exp(1j * x2)
        x3 = (alpha + x2) / (p - 3)
        x4 = (alpha + 2 * x2) / (p - 4)
        tn3 = t * np.exp(1j * x3)
        tn4 = t * np.exp(1j * x4)
        x5 = (alpha + x3) / (p - 3)
        x6 = (alpha + x4) / (p - 3)
        x7 = (alpha + 2 * x3) / (p - 4)
        x8 = (alpha + x3 + x4) / (p - 4)
        tn5 = t * np.exp(1j * x5)
        tn6 = t * np.exp(1j * x6)
        tn7 = t * np.exp(1j * x7)
        tn8 = t * np.exp(1j * x8)
        x9 = (alpha + x5) / (p - 3)
        x10 = (alpha + x7) / (p - 3)
        x11 = (alpha + 2 * x5) / (p - 4)
        x12 = (alpha + x5 + x7) / (p - 4)
        x13 = (alpha + x6 + x8) / (p - 4)
        x14 = (alpha + x8) / (p - 3)
        x15 = (alpha + x6) / (p - 3)
        x16 = (alpha + x5 + x8) / (p - 4)
        x17 = (alpha + 2 * x6) / (p - 4)
        tn9 = t * np.exp(1j * x9)
        tn10 = t * np.exp(1j * x10)
        tn11 = t * np.exp(1j * x11)
        tn12 = t * np.exp(1j * x12)
        tn13 = t * np.exp(1j * x13)
        tn14 = t * np.exp(1j * x14)
        tn15 = t * np.exp(1j * x15)
        tn16 = t * np.exp(1j * x16)
        tn17 = t * np.exp(1j * x17)

        val_letter = np.array([1+0j,tn1,tn2,tn3,tn4,tn5,tn6,tn7,tn8,tn9,tn10,tn11,tn12,tn13,tn14,
                               tn15,tn16,tn17,np.conj(tn1),np.conj(tn2),np.conj(tn3),np.conj(tn4),np.conj(tn5),
                               np.conj(tn6),np.conj(tn7),np.conj(tn8),np.conj(tn9),np.conj(tn10),np.conj(tn11),
                               np.conj(tn12),np.conj(tn13),np.conj(tn14),np.conj(tn15),np.conj(tn16),np.conj(tn17)])
        letter_val = np.array(['t','tn1','tn2','tn3','tn4','tn5','tn6','tn7','tn8','tn9','tn10','tn11','tn12','tn13','tn14',
                               'tn15','tn16','tn17','tn1_conj','tn2_conj','tn3_conj','tn4_conj','tn5_conj','tn6_conj',
                               'tn7_conj','tn8_conj','tn9_conj','tn10_conj','tn11_conj','tn12_conj','tn13_conj','tn14_conj',
                               'tn15_conj','tn16_conj','tn17_conj'])

        connections = np.array([])
        for row in range(np.size(ham,0)):
            val_connects = np.array([])
            for val in ham[row,:]:
                if val != 0:
                    # print(val)
                    val_connects = np.append(val_connects, letter_val[np.where(val==val_letter)[0]])
            connections = np.append(connections, '{} {} \n'.format(row+1, val_connects))
            print(row+1, val_connects, '\n')
        file_address = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/' \
                       'Hamiltonian_Check_Docs/Hermitian_PeierlsSub/p{}q{}n{}_ConjugateSpecific'.format(p,q,n)
        file = open(file_address, 'w')
        file.writelines(connections)
        file.close()

def Peierls_Sub_Motif_Sequence_to_Letters(p,q,n,seq, alpha):
    t = 1
    if q == 3:
        x1 = alpha / p
        tn1 = t * np.exp(1j * x1)
        x2 = (alpha + x1) / (p - 3)
        tn2 = t * np.exp(1j * x2)
        x3 = (alpha + x2) / (p - 3)
        x4 = (alpha + 2 * x2) / (p - 4)
        tn3 = t * np.exp(1j * x3)
        tn4 = t * np.exp(1j * x4)
        x5 = (alpha + x3) / (p - 3)
        x6 = (alpha + x4) / (p - 3)
        x7 = (alpha + 2 * x3) / (p - 4)
        x8 = (alpha + x3 + x4) / (p - 4)
        tn5 = t * np.exp(1j * x5)
        tn6 = t * np.exp(1j * x6)
        tn7 = t * np.exp(1j * x7)
        tn8 = t * np.exp(1j * x8)
        x9 = (alpha + x5) / (p - 3)
        x10 = (alpha + x7) / (p - 3)
        x11 = (alpha + 2 * x5) / (p - 4)
        x12 = (alpha + x5 + x7) / (p - 4)
        x13 = (alpha + x6 + x8) / (p - 4)
        x14 = (alpha + x8) / (p - 3)
        x15 = (alpha + x6) / (p - 3)
        x16 = (alpha + x5 + x8) / (p - 4)
        x17 = (alpha + 2 * x6) / (p - 4)
        tn9 = t * np.exp(1j * x9)
        tn10 = t * np.exp(1j * x10)
        tn11 = t * np.exp(1j * x11)
        tn12 = t * np.exp(1j * x12)
        tn13 = t * np.exp(1j * x13)
        tn14 = t * np.exp(1j * x14)
        tn15 = t * np.exp(1j * x15)
        tn16 = t * np.exp(1j * x16)
        tn17 = t * np.exp(1j * x17)
        val_letter = np.array([1 + 0j, tn1, tn2, tn3, tn4, tn5, tn6, tn7, tn8, tn9, tn10, tn11, tn12, tn13, tn14,
                               tn15, tn16, tn17])
        letter_val = np.array(['t', 'tn1', 'tn2', 'tn3', 'tn4', 'tn5', 'tn6', 'tn7', 'tn8', 'tn9', 'tn10', 'tn11', 'tn12', 'tn13', 'tn14',
                               'tn15', 'tn16', 'tn17'])
        letters = np.array([])
        for s in seq:
            letters = np.append(letters, letter_val[np.where(val_letter==s)[0] or np.where(val_letter==np.conj(s))[0]])
        [ print(let,'\n') for let in letters ]
        return(letters)

def NonHermitian_PeierlsSub_Hamiltonian_Algebraic_Equivalence_Check(p,q,n,alphamag,alpha,ham):
    tblock = inter_sublattice_block_PeierlsSub(p,q,n,1,alphamag)[0]
    zeroblock = np.zeros((np.size(tblock,0),np.size(tblock,1)))
    equivalent_nhh = np.block([[zeroblock, tblock*(1+alpha)],[np.conj(np.transpose(tblock))*(1-alpha), zeroblock]])
    print(np.all(np.around(ham,10) == np.around(equivalent_nhh,10)) == True)

def NonHermitian_Hermitian_Comparison_PeierlsSub(p,q,n,ham,nhh):
    a_sites, b_sites = site_assignment(p,q,n,ham)
    basis_switch_pre = np.concatenate((a_sites,b_sites)).astype(int)
    basis_switch = np.linspace(0,np.size(ham,0)-1,np.size(ham,0),dtype=int)
    basis_switch = basis_switch[np.argsort(basis_switch_pre)]
    ham_reconstructed = nhh[:,basis_switch]
    ham_reconstructed = ham_reconstructed[basis_switch,:]
    print(np.all((ham==ham_reconstructed)==True))

def near_diagonal_conjugate_check(ham):
    upperdiag = np.diag(ham, k=1)
    lowerdiag = np.diag(ham, k=-1)
    print('Is all upper diag positive imaginary: {}'.format(np.all((np.imag(upperdiag) >= 0)==True)))
    print('Is all lower diag negative imaginary: {}'.format(np.all((np.imag(lowerdiag) <= 0)==True)))
    print('Is upper diag and lower diag conjuagtes of one another: {}'.format(np.all((upperdiag==np.conj(lowerdiag))==True)))

def Peierls_Sub_to_letters_conjugate_specific_Honeycomb_nl8(a,ham):
    t = 1

    x1 = a/6
    x2 = (a + x1)/3
    x3 = (a + x2)/3
    x4 = (a + 2*x2)/2
    x5 = (a + x3)/3
    x6 = (a + x3 + x4)/2
    x7 = (a + x5)/3
    x8 = (a + x5 + x6)/2
    x9 = (a + 2*x6)/2
    x10 = (a + x7)/3
    x11 = (a + x7 + x8)/2
    x12 = (a + x8 + x9)/2
    x13 = (a + x10)/3
    x14 = (a + x10 + x11)/2
    x15 = (a + x11 + x12)/2
    x16 = (a + 2*x12)/2
    x17 = (a + x13)/3
    x18 = (a + x13 + x14)/2
    x19 = (a + x14 + x15)/2
    x20 = (a + x15 + x16)/2

    tn1 = t * np.exp(1j*x1)
    tn2 = t * np.exp(1j*x2)
    tn3 = t * np.exp(1j*x3)
    tn4 = t * np.exp(1j*x4)
    tn5 = t * np.exp(1j*x5)
    tn6 = t * np.exp(1j*x6)
    tn7 = t * np.exp(1j*x7)
    tn8 = t * np.exp(1j*x8)
    tn9 = t * np.exp(1j*x9)
    tn10 = t * np.exp(1j*x10)
    tn11 = t * np.exp(1j*x11)
    tn12 = t * np.exp(1j*x12)
    tn13 = t * np.exp(1j*x13)
    tn14 = t * np.exp(1j*x14)
    tn15 = t * np.exp(1j*x15)
    tn16 = t * np.exp(1j*x16)
    tn17 = t * np.exp(1j*x17)
    tn18 = t * np.exp(1j*x18)
    tn19 = t * np.exp(1j*x19)
    tn20 = t * np.exp(1j*x20)

    val_letter = np.array([1 + 0j, tn1, tn2, tn3, tn4, tn5, tn6, tn7, tn8, tn9, tn10, tn11, tn12, tn13, tn14,
                           tn15, tn16, tn17, tn18, tn19, tn20,
                           np.conj(tn1), np.conj(tn2), np.conj(tn3), np.conj(tn4), np.conj(tn5),
                           np.conj(tn6), np.conj(tn7), np.conj(tn8), np.conj(tn9), np.conj(tn10), np.conj(tn11),
                           np.conj(tn12), np.conj(tn13), np.conj(tn14), np.conj(tn15), np.conj(tn16), np.conj(tn17),
                           np.conj(tn18), np.conj(tn19), np.conj(tn20)])
    letter_val = np.array(
        ['t', 'tn1', 'tn2', 'tn3', 'tn4', 'tn5', 'tn6', 'tn7', 'tn8', 'tn9', 'tn10', 'tn11', 'tn12', 'tn13', 'tn14',
         'tn15', 'tn16', 'tn17', 'tn18', 'tn19', 'tn20',
         'tn1_conj', 'tn2_conj', 'tn3_conj', 'tn4_conj', 'tn5_conj', 'tn6_conj',
         'tn7_conj', 'tn8_conj', 'tn9_conj', 'tn10_conj', 'tn11_conj', 'tn12_conj', 'tn13_conj', 'tn14_conj',
         'tn15_conj', 'tn16_conj', 'tn17_conj', 'tn18_conj', 'tn19_conj', 'tn20_conj'])

    connections = np.array([])
    for row in range(np.size(ham, 0)):
        val_connects = np.array([])
        for val in ham[row, :]:
            if val != 0:
                # print(val)
                val_connects = np.append(val_connects, letter_val[np.where(np.around(val,13) == np.around(val_letter,13))[0]])
        connections = np.append(connections, '{} {} \n'.format(row + 1, val_connects))
        print(row + 1, val_connects, '\n')
    file_address = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/' \
                   'Hamiltonian_Check_Docs/Hermitian_PeierlsSub/Honeycomb_nl8_ConjugateSpecific'
    file = open(file_address, 'w')
    file.writelines(connections)
    file.close()

def Peierls_Sub_to_letters_conjugate_specific_NoLineNumbers(p, q, n, alpha, ham):
    if q == 3:
        t = 1

        x1 = alpha / p
        tn1 = t * np.exp(1j * x1)
        x2 = (alpha + x1) / (p - 3)
        tn2 = t * np.exp(1j * x2)
        x3 = (alpha + x2) / (p - 3)
        x4 = (alpha + 2 * x2) / (p - 4)
        tn3 = t * np.exp(1j * x3)
        tn4 = t * np.exp(1j * x4)
        x5 = (alpha + x3) / (p - 3)
        x6 = (alpha + x4) / (p - 3)
        x7 = (alpha + 2 * x3) / (p - 4)
        x8 = (alpha + x3 + x4) / (p - 4)
        tn5 = t * np.exp(1j * x5)
        tn6 = t * np.exp(1j * x6)
        tn7 = t * np.exp(1j * x7)
        tn8 = t * np.exp(1j * x8)
        x9 = (alpha + x5) / (p - 3)
        x10 = (alpha + x7) / (p - 3)
        x11 = (alpha + 2 * x5) / (p - 4)
        x12 = (alpha + x5 + x7) / (p - 4)
        x13 = (alpha + x6 + x8) / (p - 4)
        x14 = (alpha + x8) / (p - 3)
        x15 = (alpha + x6) / (p - 3)
        x16 = (alpha + x5 + x8) / (p - 4)
        x17 = (alpha + 2 * x6) / (p - 4)
        tn9 = t * np.exp(1j * x9)
        tn10 = t * np.exp(1j * x10)
        tn11 = t * np.exp(1j * x11)
        tn12 = t * np.exp(1j * x12)
        tn13 = t * np.exp(1j * x13)
        tn14 = t * np.exp(1j * x14)
        tn15 = t * np.exp(1j * x15)
        tn16 = t * np.exp(1j * x16)
        tn17 = t * np.exp(1j * x17)

        val_letter = np.array([1 + 0j, tn1, tn2, tn3, tn4, tn5, tn6, tn7, tn8, tn9, tn10, tn11, tn12, tn13, tn14,
                               tn15, tn16, tn17, np.conj(tn1), np.conj(tn2), np.conj(tn3), np.conj(tn4),
                               np.conj(tn5),
                               np.conj(tn6), np.conj(tn7), np.conj(tn8), np.conj(tn9), np.conj(tn10), np.conj(tn11),
                               np.conj(tn12), np.conj(tn13), np.conj(tn14), np.conj(tn15), np.conj(tn16),
                               np.conj(tn17)])
        letter_val = np.array(
            ['t', 'tn1', 'tn2', 'tn3', 'tn4', 'tn5', 'tn6', 'tn7', 'tn8', 'tn9', 'tn10', 'tn11', 'tn12', 'tn13',
             'tn14',
             'tn15', 'tn16', 'tn17', 'tn1_conj', 'tn2_conj', 'tn3_conj', 'tn4_conj', 'tn5_conj', 'tn6_conj',
             'tn7_conj', 'tn8_conj', 'tn9_conj', 'tn10_conj', 'tn11_conj', 'tn12_conj', 'tn13_conj', 'tn14_conj',
             'tn15_conj', 'tn16_conj', 'tn17_conj'])

        connections = np.array([])
        for row in range(np.size(ham, 0)):
            val_connects = np.array([])
            for val in ham[row, :]:
                if val != 0:
                    # print(val)
                    val_connects = np.append(val_connects, letter_val[np.where(val == val_letter)[0]])
            connections = np.append(connections, '{} \n'.format(val_connects))
            print(row + 1, val_connects, '\n')
        file_address = '/home/cal422/PycharmProjects/Hyperbolic_Lattice_Self_Consistent_Hartree_Fock_Data_Acquisition/' \
                       'Hamiltonian_Check_Docs/Hermitian_PeierlsSub/p{}q{}n{}_ConjugateSpecific_NoLineNumber'.format(p, q, n)
        file = open(file_address, 'w')
        file.writelines(connections)
        file.close()

