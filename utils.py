import pandas as pd
import logging
import zipfile
import argparse
import re
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import base64
import zlib
import struct
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import numpy as np
import scipy
from scipy.signal import find_peaks, peak_prominences, peak_widths
# #
#Utility Functions
# #


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



def peaks_find(x):
    peaks, _= scipy.signal.find_peaks(x, prominence= 1)
    return peaks


def peak_find(x):
    peaks= scipy.signal.find_peaks_cwt(x, np.arange(1,30), min_snr= 1)
    return peaks


def smoothing(data):
    for ii in range(2):
        (data, coeff_d) = pywt.dwt(data, 'sym3')
    return data


#

def peak_range(x,y,z):
    dd=[]
    for e,r in zip(x,y):
        kk= z[e:r]
        dd.append(kk)
    return dd
#

# def peak_area(scan_array, intensity_array, start, stop):
#     area = 0
#
#     for i in range(start + 1, stop):
#         x1 = scan_array[i - 1]
#         y1 = intensity_array[i - 1]
#         x2 = scan_array[i]
#         y2 = intensity_array[i]
#         area += (y1 * (x2 - x1)) + ((y2 - y1) * (x2 - x1) / 2.)
#     return area

def width(x, peak):

    prominences, left_bases, right_bases = peak_prominences(x, peak)
    offset = np.ones_like(prominences)
    widths, h_eval, left_ips, right_ips = peak_widths(
    x, peak,
    rel_height=1,
    prominence_data=(offset, left_bases, right_bases))
    print(left_ips)
    print(right_ips)

    return widths, left_ips, right_ips


##function to integrate utility Functions

def final(df):
    try:
        x= np.array(df['intensity'].tolist())
        y= np.array(df['mz'].tolist())
        z= np.array(df['scan'].tolist())
        w= np.array(df['rt'].tolist())
        er= np.array(df['filename'].tolist())
        peak= peak_find(x)

        if len(peak) >= 1:
            from scipy.signal import chirp, find_peaks, peak_widths
            peak_width, left_ips, right_ips = width(x, peak)
            left_ips = left_ips
            right_ips= right_ips
            peak_mz= y[peak]
            peak_intensity= x[peak]
            peak_rt= w[peak]
            filename= er[peak]
            master_dict= {'peak_mz': peak_mz, 'peak_intensity': peak_intensity, 'peak_rt': peak_rt, 'peak_width': peak_width, 'left_ips': left_ips, 'right_ips': right_ips, 'filename': filename}
            df9= pd.DataFrame(master_dict)

            return df9
    except Exception as e:
        print(e)
