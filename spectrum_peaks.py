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
import xml.etree.ElementTree as ET
plt.style.use('seaborn-pastel')
import os
from numpy import arange
from utils import *





class masslist_data():
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.filename= str(os.path.split(self.path)[-1])





    def __repr__(self):
        return "name of file is " +"  "+ str(os.path.split(self.path)[-1])


    def add_objects(self):
        xx= self.path
        workers= []
        if xx.endswith(".mzXML"):
            print('yes')
            tree = ET.parse(self.path)
            root = tree.getroot()
            first= root.findall('.//{http://sashimi.sourceforge.net/schema_revision/mzXML_3.2}scan')
            second= root.findall('.//{http://sashimi.sourceforge.net/schema_revision/mzXML_3.2}peaks')
            for x, y in zip(first, second):
                x= x.attrib
                k= y.text
                z= y.attrib
                obj= Worker(x, k, z)
                workers.append(obj)
            return workers





    def find_data(self):
        workers= self.add_objects()
        mlist= []
        scanlist= []
        rtlist= []
        intensitylist= []
        filename=[]

        for x in workers:
            a, b = x.tag_text()
            scan, _, _, retentiontime= x.tag_dict()
            mlist.extend(a)
            scanlist.extend([scan]* len(a))
            filename.extend([self.filename]* len(a))

            rtlist.extend([retentiontime] * len(a))
            intensitylist.extend(b)
        return mlist, rtlist, scanlist, intensitylist, filename



    def make_dataframe(self):
        a, b, c, d, e = self.find_data()
        df1= pd.DataFrame({'mz': a, 'intensity': d, 'scan': c, 'rt': b, 'filename': e})
        return df1


    def peaklist_full(self):
        df1= self.make_dataframe()
        df1.to_csv(f'{self.filename}.csv')
        df1['rt']= df1['rt'].astype(float)
        print(df1.head())
        return df1


class Worker():
    def __init__(self, x, y, z):
        super().__init__()
        self.tag = x
        self.text = y
        self.peaktag= z

    def tag_dict(self):
        data= self.tag
        scan= int(data['num'])
        basePeakMz= data['basePeakMz']
        basePeakIntensity= data['basePeakIntensity']
        basePeakIntensity= basePeakIntensity.strip()
        basePeakIntensity= re.sub('e0', 'e+0', basePeakIntensity)
        basePeakIntensity= round(float(basePeakIntensity), 4)
        retentionTime= data['retentionTime']
        retentionTime= re.sub('[^0-9.]+', '', retentionTime)
        retentionTime= float(retentionTime)
        retentionTime= retentionTime/60
        return scan, basePeakMz, basePeakIntensity, retentionTime



    def tag_text(self):
        mz_list, intensity_list= [], []
        dd= self.peaktag
        coded= self.text
        mz_list= []
        precision = 'f'
        if dd['precision'] == 64:
            precision = 'd'

        # get endian
        endian = '!'
        if dd['byteOrder'] == 'little':
            endian = '<'
        elif dd['byteOrder'] == 'big':
            endian = '>'
        compression=None


        data = coded.encode("ascii")
        data = base64.decodebytes(data)

        mz_list= []

        if dd['compressionType'] == 'zlib':
            data = zlib.decompress(data)


        # convert from binary
        count = len(data) // struct.calcsize(endian + precision)
        data = struct.unpack(endian + precision * count, data[0:len(data)])
        points = map(list, zip(data[::2], data[1::2]))

        for x in points:

            mz_list.append(round(x[0], 4))
            intensity_list.append(round(x[-1], 4))

        mz_list= np.array(mz_list)
        intensity_list= np.array(intensity_list)
        peak_intensity = intensity_list
        peak_mz= mz_list

        return peak_mz, peak_intensity




class Peaklist_data():

    def __init__(self, obj, filename):
        super().__init__()
        self.mass = obj
        self.filename= os.path.split(filename)[-1]




    def find_peak(self):
        df1= self.mass
        df1= df1[df1['intensity'] > 1000]
        print('peak_picking step')
        df1= df1.dropna()
        df1['rt']= df1['rt'].astype(float)
        minimum= df1['rt'].min()
        print(minimum)
        maximum= df1['rt'].max()
        thresold= 0.8
        ranges= arange(minimum, maximum, thresold)
        bins = pd.cut(df1['rt'], ranges)
        print(bins)
        dfm= pd.DataFrame()
        for c, d in df1.groupby(bins):
            if not d.empty:
                minimum1= d['mz'].min()
                maximum1= d['mz'].max()
                thresold1= 0.005
                ranges1= arange(minimum1, maximum1, thresold1)
                bins1 = pd.cut(d['mz'], ranges1)
                grouped= d.groupby(bins1)
                for y, w in grouped:

                    if not w.empty:
                        # w['intensity']= moving_average(w['intensity'].values, 60)
                        dfx= final(w)
                        print('done')
                        dfm= dfm.append(dfx, ignore_index= True)
        dfm.to_csv(f"{self.filename}_peaks.csv")
        return dfm

        #
        # except Exception as e:
        #     pass


######Multiprocessing of many files




# def create_slaves(input_list):
#     try:
#         slaves = []
#         for input_data in input_list:
#             mass= masslist_data(input_data)
#             peaks_full= mass.peaklist_full()
#             peaks= Peaklist_data(peaks_full, input_data)
#
#             slaves.append(peaks)
#         return slaves
#
#     except Exception as e:
#         print(e)
#         warnings.warn(f"The error {e} has been found")
#
#
# from threading import Thread
#
# def execute(workers):
#     try:
#         threads = [Thread(target=w.find_peak) for w in workers]
#         for thread in threads: thread.start()
#         for thread in threads: thread.join()
#
#     except Exception as e:
#         print(e)
#         warnings.warn(f"The error {e} has been found")


def process_all(data_dir):
    filepath= []

    for dirname, _, filenames in os.walk(data_dir):
        print(dirname)
        print(filenames)
        for filename in filenames:
            filepath.append(os.path.join(dirname, filename))

    for f in filepath:

        mass= masslist_data(f)
        peaks_full= mass.peaklist_full()
        peaks= Peaklist_data(peaks_full, f)
        peaks_final= peaks.find_peak()


        #
        # workers = create_slaves(filepath)
        # return execute(workers)







if __name__ == "__main__":
    process_all('./data')
