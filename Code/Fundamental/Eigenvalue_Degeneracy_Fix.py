import scipy
import numpy as np
from Fundamental.Hamiltonian import H0
from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian as nhh
from Check.Orthogonality_Check import biorthogonality_matrix

# def num_degen_spectrum(eners):
#     return(len(eners)-len(np.unique(eners)))

# def num_degen(p,q,n,alpha,t):
#     ham = nhh(p,q,n,alpha,t)
#     eners, states = scipy.linalg.eig(ham)
#     return(len(eners)-len(np.unique(eners)))

# def num_degen_round(p,q,n,alpha,t,deci_places):
#     ham = nhh(p,q,n,alpha,t)
#     eners, states = scipy.linalg.eig(ham)
#     return(len(eners)-len(np.unique(np.around(eners, deci_places))))

# def ortho_sign_error_fix(ls_mat_orth, rs_mat_orth):
#     for i in range(np.size(ls_mat_orth,1)):
#         if np.sum(ls_mat_orth[:,i]*rs_mat_orth[:,i]) < 0:
#             rs_mat_orth[:,i] = -1*rs_mat_orth[:,i]
#     return(ls_mat_orth, rs_mat_orth)

# def biorthogonal_degen_sort_error_fix(ls_mat, rs_mat):
#     dim = np.size(ls_mat,0)
#     associated_eigvec_pairs_list1 = np.array([])
#     associated_eigvec_pairs_list2 = np.array([])
#     for d1 in range(dim):
#         ortho_results = np.array([])
#         for d2 in range(dim):
#             ortho_results = np.append(ortho_results,np.abs(np.sum(ls_mat[:,d2]*rs_mat[:,d1]))) #ADD ABSOLUTE VAL
#         max_ortho_ind = np.where(ortho_results == np.max(ortho_results))
#         associated_eigvec_pairs_list1 = np.append(associated_eigvec_pairs_list1,d1)
#         associated_eigvec_pairs_list2 = np.append(associated_eigvec_pairs_list2,max_ortho_ind)
#     print('Is pairing one-to-one? {}'.format(len(np.unique(associated_eigvec_pairs_list2))==dim))
#     if len(np.unique(associated_eigvec_pairs_list2))!=dim:
#         print('Neglected pairs: {}'.format(np.setxor1d(associated_eigvec_pairs_list1,associated_eigvec_pairs_list2)))
#     pair_errors = np.array([])
#     for v in range(len(associated_eigvec_pairs_list1)):
#         if np.abs(associated_eigvec_pairs_list1[v]-associated_eigvec_pairs_list2[v]) > 1:
#             pair_errors = np.append(pair_errors,v)
#         else:
#             continue
#     if len(pair_errors) != 0:
#         print('pair errors at: {}'.format(pair_errors))
#     else:
#         print('No pair errors')
#     rs_mat_sorted = rs_mat[:,[ np.int(a1) for a1 in associated_eigvec_pairs_list1 ]]
#     ls_mat_sorted = ls_mat[:,[ np.int(a2) for a2 in associated_eigvec_pairs_list2 ]]
#     return(ls_mat_sorted,rs_mat_sorted)

# def matrix_eigenvector_swap_fix(mat,ls,rs):
#     associated_eigvec_pairs_list1 = np.array([])
#     associated_eigvec_pairs_list2 = np.array([])
#     for r in range(np.size(mat,0)-1):
#         for c in range(np.size(mat,1)-1):
#             if mat[r+1,c] == mat[r,c+1] and mat[r+1,c] == np.max(mat[:,c]):
#                 associated_eigvec_pairs_list1 = np.append(associated_eigvec_pairs_list1, r)
#                 associated_eigvec_pairs_list2 = np.append(associated_eigvec_pairs_list2, c+1)
#                 associated_eigvec_pairs_list1 = np.append(associated_eigvec_pairs_list1, r+1)
#                 associated_eigvec_pairs_list2 = np.append(associated_eigvec_pairs_list2, c)
#             elif mat[r-1,c] == mat[r,c-1] and mat[r-1,c] == np.max(mat[:,c]):
#                 continue
#             else:
#                 associated_eigvec_pairs_list1 = np.append(associated_eigvec_pairs_list1, r)
#                 associated_eigvec_pairs_list2 = np.append(associated_eigvec_pairs_list2, c)
#     if np.any(associated_eigvec_pairs_list1 != np.size(mat,0)-1) == False:
#         associated_eigvec_pairs_list1 = np.append(associated_eigvec_pairs_list1, np.size(mat,0)-1)
#         associated_eigvec_pairs_list2 = np.append(associated_eigvec_pairs_list2, np.size(mat,0)-1)
#     rs = rs[:,[ np.int(a2) for a2 in associated_eigvec_pairs_list2 ]]
#     return(ls,rs)

# def biortho_fix_degen(eners, ls_mat, rs_mat):
#     if num_degen_spectrum(eners) != 0:
#         uniq_eners, eners_degen_count = np.unique(np.around(eners,8), return_counts=True)
#         degen_eners = uniq_eners[np.where(eners_degen_count > 1)]
#         for de in degen_eners:
#             ener_spectrum_where_degen = np.where(np.around(eners,8) == de)
#             ls_prob_eigvecs = np.empty((len(eners),1))
#             rs_prob_eigvecs = np.empty((len(eners),1))
#             for w in ener_spectrum_where_degen:
#                 ls_prob_eigvecs = np.concatenate((ls_prob_eigvecs,ls_mat[:,w]),axis=1)
#                 rs_prob_eigvecs = np.concatenate((rs_prob_eigvecs,rs_mat[:,w]),axis=1)
#             ls_prob_eigvecs = ls_prob_eigvecs[:,1:]
#             rs_prob_eigvecs = rs_prob_eigvecs[:,1:]
#             ls_new_ortho_eigvecs = scipy.linalg.orth(ls_prob_eigvecs)
#             rs_new_ortho_eigvecs = scipy.linalg.orth(rs_prob_eigvecs)
#             for i in range(len(ener_spectrum_where_degen)):
#                 ls_mat[:,ener_spectrum_where_degen[i]] = ls_new_ortho_eigvecs[i]
#                 rs_mat[:,ener_spectrum_where_degen[i]] = rs_new_ortho_eigvecs[i]
#     return(ls_mat,rs_mat)

# def fix_degen(eners, states_mat):
#     if num_degen_spectrum(eners) != 0:
#         uniq_eners, eners_degen_count = np.unique(eners, return_counts=True)
#         degen_eners = uniq_eners[np.where(eners_degen_count > 1)]
#         for de in degen_eners:
#             ener_spectrum_where_degen = np.where(eners == de)
#             prob_eigvecs= np.empty((len(eners),1))
#             for w in ener_spectrum_where_degen:
#                 prob_eigvecs = np.concatenate((prob_eigvecs,states_mat[:,w]),axis=1)
#             prob_eigvecs = prob_eigvecs[:,1:]
#             new_ortho_eigvecs = scipy.linalg.orth(prob_eigvecs)
#             for i in range(len(ener_spectrum_where_degen)):
#                 states_mat[:,ener_spectrum_where_degen[i]] = new_ortho_eigvecs[i]
#     return(states_mat)

def small_chaos(chaos_order, num_sites):
    chaos = np.array([])
    for n in range(int(num_sites)):
        chaos = np.append(chaos, np.random.uniform(-chaos_order/2,chaos_order/2))
    chaos_mat = np.diag(chaos)
    chaos_correction_val = np.trace(chaos_mat)/num_sites
    chaos_correction_mat = np.eye(int(num_sites))*chaos_correction_val
    return(chaos_mat-chaos_correction_mat)

# def small_chaos_check(p,q,nl,alpha_list,t,chaos_order_list):
#     from Check.Orthogonality_Check import biorthogonality_matrix
#     from Check.Orthogonality_Check import orthogonality_matrix
#     from Fundamental.Hamiltonian import H0
#     from Fundamental.NonHermitian_Hamiltonian import NonHermitian_Hamiltonian as nhh
#     import scipy
#     import numpy as np
#     from Fundamental.Biorthogonal import biortogonal_normalize
#     from Fundamental.Biorthogonal import left_eigvecs
#     from Fundamental.Number_Points import points
#     problem_alpha_list = np.array([])
#     problem_chaosorder_list = np.array([])
#     for a in range(len(alpha_list)):
#         for c in range(len(chaos_order_list)):
#             counter = 0
#             while counter <= 4:
#                 chaos_ham = small_chaos(chaos_order_list[c], points(p, q, nl)[1])
#                 og_ham = nhh(p,q,nl,alpha_list[a],t)
#                 ham = og_ham + chaos_ham
#                 eners, ls, rs = scipy.linalg.eig(ham, left=True, right=True)
#                 ls = np.conj(ls)
#                 ls = ls[:, np.argsort(eners)]
#                 rs = rs[:, np.argsort(eners)]
#                 eners = np.sort(eners)
#                 ls_norm, rs_norm = biortogonal_normalize(ls, rs)
#                 biortho_mat = biorthogonality_matrix(eners, ls_norm, rs_norm)
#                 biortho_mat = np.around(biortho_mat, 8)
#                 biortho_mat_offdiag = biortho_mat - np.diag(np.diag(biortho_mat))
#                 if np.count_nonzero(biortho_mat_offdiag) != 0:
#                     problem_alpha_list = np.append(problem_alpha_list,alpha_list[a])
#                     problem_chaosorder_list = np.append(problem_chaosorder_list,chaos_order_list[c])
#                 counter += 1
#     return(problem_alpha_list,problem_chaosorder_list)