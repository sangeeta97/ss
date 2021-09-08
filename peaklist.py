import pandas as pd
import logging
import zipfile
import argparse
import re
import sys
import warnings
import pywt
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
# #
#Utility Functions
# #
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

def peak_loc(xx):

    peak_indexes = scipy.signal.argrelextrema(xx, np.greater)
    peak_indexes = peak_indexes[0]
    return peak_indexes

def peaks_property_left(x, y):
    _, left_bases, _ = scipy.signal.peak_prominences(x, y)
    return left_bases
#
#
def peaks_property_right(x, y):
    _, _, right = scipy.signal.peak_prominences(x, y)
    return right
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



def peak_area(x, y):
    areas = []
    for peak, t in zip(x, y):
        area1 = np.trapz(peak, t)
        areas.append(area1)
    return areas




##function to integrate utility Functions

def final(df):
    try:
        x= np.array(df['intensity'].tolist())
        y= np.array(df['mz'].tolist())
        z= np.array(df['scan'].tolist())
        w= np.array(df['rt'].tolist())
        er= np.array(df['filename'].tolist())
        rz= np.array(df['initial'].tolist())
        peak= peak_find(x)
        print(peak)
        if len(peak) >= 1:
            from scipy.signal import chirp, find_peaks, peak_widths
            result_half = peak_widths(x, peak, rel_height=0.5)
            result_half= result_half[0]
            result_half= np.array(result_half)
            result_bol= (result_half > 2) & (result_half < 30)
            peak= peak[result_bol]
            peak_mz= y[peak]
            peak_intensity= x[peak]
            peak_rt= w[peak]
            filename= er[peak]
            initial= rz[peak]
            number= peak
            plot_intense= [x] * len(peak)
            # left_index= peaks_property_left(x, peak)
            # right_index= peaks_property_right(x, peak)
            # scan_range= peak_range(left_index, right_index, z)
            # mz_range= peak_range(left_index, right_index, y)
            # intensity_range= peak_range(left_index, right_index, x)
            # rt_range= peak_range(left_index, right_index, w)
            # peakarea= peak_area(peak, scan_range)

            # ft = np.fft.fft(y)
            # peakarea= peak_area(peak_rt, peak_intensity)

            # mz_range= list(mz_range)
            # rt_range= list(rt_range)
            # scan_range= list(scan_range)
            # intensity_range= list(intensity_range)
            # mz_range= ''.join([x for x in mz_range])
            # rt_range= ''.join([x for x in rt_range])
            # scan_range= ''.join([x for x in scan_range])
            # intensity_range= ''.join([x for x in intensity_range])



            master_dict= {'peak_mz': peak_mz, 'peak_intensity': peak_intensity, 'peak_rt': peak_rt, 'filename': filename, 'initial': initial, 'number': number, 'plot': plot_intense}
            df9= pd.DataFrame(master_dict)

            return df9
    except Exception as e:
        print(e)
