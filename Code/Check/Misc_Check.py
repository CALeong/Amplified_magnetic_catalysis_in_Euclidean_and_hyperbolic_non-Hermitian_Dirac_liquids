import numpy as np

def center_vs_raw_data_check(p, center_data_address, raw_data_address):
    center_data = np.load(center_data_address)
    raw_data = np.load(raw_data_address)
    tot_num_sites = np.size(raw_data,1) - 1
    a_center_sites = np.arange(0,int(p/2))
    b_center_sites = np.arange(0,int(p/2)) + tot_num_sites/2
    a_center_sites = np.array([int(i) for i in a_center_sites])
    b_center_sites = np.array([int(i) for i in b_center_sites])
    a_center_raw_data = raw_data[:,a_center_sites+1]
    b_center_raw_data = raw_data[:,b_center_sites+1]
    a_center_delta = np.array([])
    b_center_delta = np.array([])
    for row in range(np.size(a_center_raw_data,0)):
        a_center_delta = np.append(a_center_delta, np.abs(np.average(a_center_raw_data[row,:])))
    for row in range(np.size(b_center_raw_data,0)):
        b_center_delta = np.append(b_center_delta, np.abs(np.average(b_center_raw_data[row,:])))
    center_results_from_raw = (a_center_delta + b_center_delta) / 2
    print('Do Raw and Center match: {}'.format(np.all((center_results_from_raw-center_data)==0)))
    print(np.max(np.abs(center_results_from_raw-center_data)))