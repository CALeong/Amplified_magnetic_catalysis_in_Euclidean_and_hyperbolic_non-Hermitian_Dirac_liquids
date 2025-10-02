import numpy as np

#All code based on The Kernel Polynomial Method by Weibe, Wellein, Alvermann, and Fehske

def jackson_kernel(n,N): #Based on Eq. 71
    return((1/(N+1))*(((N-n+1)*np.cos((np.pi*n)/(N+1)))+((np.sin((np.pi*n)/(N+1)))*((np.cos((np.pi)/(N+1)))/(np.sin((np.pi)/(N+1)))))))

def rescale(og_operator, spec_min, spec_max, epsilon): #Based on Eq. 24 and 25
    a = (spec_max - spec_min) / (2 - epsilon)
    b = (spec_max + spec_min) / 2
    a_mat = np.eye(np.size(og_operator,0))*((spec_max - spec_min) / (2 - epsilon))
    b_mat = np.eye(np.size(og_operator,0))*((spec_max + spec_min) / 2)
    return((og_operator/a) - b_mat/a)

def chebyshev_recursion_matrix_input(n_max,x): #Based on Eq. 10
    chebyshev_polys = np.array([])
    for n in range(1,n_max+1):
        if n==0:
            chebyshev_polys = np.append(chebyshev_polys, np.eye(np.size(x,0)))
        elif n==1:
            chebyshev_polys = np.append(chebyshev_polys, x)
        else:
            chebyshev_polys = np.append(chebyshev_polys, 2*np.matmul(x,chebyshev_polys[n-2])-chebyshev_polys[n-3])
    return(chebyshev_polys)

def chebyshev_state_expansion(initial_state_ket, n_max, operator): #Based on Eq. 30, 31, 32
    chebyshev_polys = chebyshev_recursion_matrix_input(n_max, operator)
    expanded_states = np.array([])
    for n in range(n_max):
        expanded_states = np.append(expanded_states, np.matmul(chebyshev_polys[n],initial_state_ket))
    return(expanded_states)

def expectation_value_moments_same_braket(initial_state_ket, n_max, operator): #Based on Eq. 34 and 35
    expanded_states = chebyshev_state_expansion(initial_state_ket, n_max, operator)
    expanded_states_bra = [np.conjugate(np.transpose(i)) for i in expanded_states]
    moments = np.array([])
    initial_state_bra = np.conjugate(np.transpose(initial_state_ket))
    m0 = np.matmul(initial_state_bra, expanded_states[0])
    m1 = np.matmul(initial_state_bra, expanded_states[1])
    for n in range(n_max):
        if n == 0:
            moments = np.append(moments, m0)
        elif n == 1:
            moments = np.append(moments, m1)
        elif n % 2 == 0:
            moments = np.append(moments, 2*np.matmul(expanded_states_bra[int(n/2)], expanded_states[int(n/2)]) - m0)
        elif n % 2 == 1:
            moments = np.append(moments, 2*np.matmul(expanded_states_bra[int((n-1)/2 + 1)], expanded_states[int((n-1)/2)]) - m1)



