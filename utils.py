import os
import gc
import mne
import joblib
import matplotlib
import numpy as np
import os.path as op
import matplotlib.pyplot as plt

from mne import read_source_estimate
from mne.datasets import sample, eegbci, fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, write_inverse_operator, read_inverse_operator, apply_inverse_epochs, apply_inverse, read_inverse_operator

matplotlib.use('tkagg')
        
# --------------------------------------------------------------------------

def get_cluster_data(cluster_id, class_label, domain="X_time", dataset="KUL", cluster_data_folder="results/cluster_data"):
    
    AA_channel_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 
                        'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 
                        'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 
                        'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 
                        'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 
                        'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2',
                        'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 
                        'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

    cluster_data = joblib.load(cluster_data_folder)

    X = cluster_data[dataset][class_label][str(cluster_id)][domain]
    sampling_rate = 128

    info = mne.create_info(AA_channel_names, sampling_rate, ch_types='eeg')
    
    # if domain.startswith("R_"):
    #     print(">>>>>>>>", X.shape)
        
    #     # X[X < 0] = 0  
    #     X = np.abs(X)      
    
    epochs = mne.EpochsArray(X, info)
    epochs.set_montage('standard_1020')  
    
    return epochs
    
# --------------------------------------------------------------------------
    
def create_forward_solution(epochs, plot=False):

    fs_dir = fetch_fsaverage(verbose=True)
    
    trans = "fsaverage"  
    src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    if plot:
    
        # Check that the locations of EEG electrodes is correct with respect to MRI
        mne.viz.plot_alignment(
            epochs.info,
            src=src,
            eeg=["original", "projected"],
            trans=trans,
            show_axes=True,
            mri_fiducials=True,
            dig="fiducials",
        ) 
        
    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
    )

    return fwd

# --------------------------------------------------------------------------

def load_inverse_operator(epochs, inverse_operator_path, overwrite=False, plot=False):
    
    if not os.path.exists(inverse_operator_path) or overwrite:

        # Lead field matrix
        fwd = create_forward_solution(epochs, plot=plot)

        noise_cov = mne.compute_covariance(
            epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
        )

        inverse_operator = make_inverse_operator(
            epochs.info, fwd, noise_cov, loose=0.2, depth=0.8
        )
        
        write_inverse_operator(inverse_operator_path, inverse_operator, overwrite=True)

    else:
        inverse_operator = read_inverse_operator(inverse_operator_path)
    
    return inverse_operator
    
# --------------------------------------------------------------------------


