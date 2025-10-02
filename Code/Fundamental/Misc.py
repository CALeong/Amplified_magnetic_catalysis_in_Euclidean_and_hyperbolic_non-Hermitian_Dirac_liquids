import numpy as np

def flatband_fraction(energy_file_addr,distfromzeroener):
    energies = np.load(energy_file_addr)
    flatband_states_num = len(np.where(np.abs(energies)<=distfromzeroener)[0])
    return(flatband_states_num/len(energies))