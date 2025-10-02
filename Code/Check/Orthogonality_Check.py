import numpy as np
import scipy
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian as nhh
# from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian_Dagger as nhh_dagger
def orthogonality_matrix(eners,states_matrix):
    ortho_results = np.empty((len(eners), len(eners)))
    for e1 in range(len(eners)):
        for e2 in range(len(eners)):
            ortho_results[e1, e2] = np.dot(states_matrix[:, e1], states_matrix[:, e2])
    return(ortho_results)

def biorthogonality_matrix(eners,left_states_matrix,right_states_matrix):
    biortho_results = np.empty((len(eners), len(eners)), dtype=np.clongdouble)
    for e1 in range(len(eners)):
        for e2 in range(len(eners)):
            biortho_results[e1, e2] = np.sum(left_states_matrix[:, e1]*right_states_matrix[:, e2])
    return(biortho_results)

def ortho_matrix_check(ortho_mat):
    print('Are all the self inner products 1? {}'.format(np.all((np.around(np.diag(ortho_mat),10) == 1))))
    off_diag_only = np.around(ortho_mat - np.diag(np.diag(ortho_mat)), 10)
    print('Are there any non orthogonal vectors? {}'.format(np.any((off_diag_only != 0))))

def ortho_matrix_diag_check(ortho_mat):
    print('Are all the self inner products 1? {}'.format(np.all((np.diag(ortho_mat) == 1))))
# def alpha0_check(p,q,nl):
#     ham = nhh(p,q,nl,0,1)
#     ham_dag = nhh_dagger(p,q,nl,0,1)
#     r_eners, rs = scipy.linalg.eig(ham)
#     l_eners, ls_pre = scipy.linalg.eig(ham_dag)
#     rs = rs[:,np.argsort(r_eners)]
#     ls_pre = ls_pre[:,np.argsort(l_eners)]
#     r_eners = np.sort(r_eners)
#     l_eners = np.sort(l_eners)
#     print('Do energy spectrums match: {}'.format(np.all(np.around(r_eners,8)==np.around(l_eners,8))==True))
#     rl_match = np.all((np.around(rs,8) == np.around(ls_pre,8)) == True)
#     print('Do the right and left eigvecs match: {}'.format(rl_match))
#
