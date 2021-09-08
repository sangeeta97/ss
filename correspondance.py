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
from spectrum_peaks import *
from sklearn.cluster import DBSCAN
from sklearn import mixture
from scipy.optimize import linear_sum_assignment
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
import plotly.graph_objects as go


def peak_range(x,y,z):
    dd=[]
    for e,r in zip(x,y):
        kk= z[e:r]
        dd.append(kk)
    return dd



def width(x, peak, xc):
    peak= np.array([peak])
    prominences, left_bases, right_bases = peak_prominences(x, peak)
    offset = np.ones_like(prominences)
    widths, h_eval, left_ips, right_ips = peak_widths(
    x, peak,
    rel_height=1,
    prominence_data=(offset, left_bases, right_bases))
    # np.testing.assert_equal(x[peak] - h_eval, 1)
    # xx= peak_range(left_ips, right_ips, x)
    # import matplotlib.pyplot as plt
    # plt.plot(xx)
    # plt.plot(peak, xx[peak], "x")
    #
    # plt.savefig(f'{xc}_{widths}.png')
    return widths[0]



def make_xic(x, peak, xc):
    peak= np.array([peak])
    prominences, left_bases, right_bases = peak_prominences(x, peak)
    offset = np.ones_like(prominences)
    widths, h_eval, left_ips, right_ips = peak_widths(
    x, peak,
    rel_height=1,
    prominence_data=(offset, left_bases, right_bases))
    np.testing.assert_equal(x[peak] - h_eval, 1)
    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.savefig(f'{xc}_{widths}.png')






class correspondance():
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.mz_tolerance= 0.001
        self.rt_tolerance= 0.05


    def result(self):
        obj1= masslist_data(self.path)
        data= obj1.peaklist_full()
        return data


class Combine():
    def __init__(self, dir):
        self.dir= dir
        print(self.dir)
        self.mz_tolerance= 0.001
        self.rt_tolerance= 0.05


    def read_values(self):
        listdataframe= pd.DataFrame()
        filepath= list()
        for dirname, _, filenames in os.walk(self.dir):
            print(dirname)
            print(filenames)
            for filename in filenames:
                filepath.append(os.path.join(dirname, filename))
        for x in filepath:
            xc= os.path.split(x)[-1]
            xc= xc.split('.')[0]
            obj= correspondance(x)
            dd= obj.result()

            kk= dd
            kk['width']= kk.apply(lambda row: width(row['plot'], row['number'], xc), axis= 1)
            kk= kk[kk['width'] != '0.']
            kk.info()
            textdata= kk['peak_mz'].tolist()
            kk.to_csv(f'{xc}.csv')
            kk['width']= pd.to_numeric(kk['width'])
            size= kk['width'].tolist()
            fig = go.Figure(data=[go.Scatter(
                x= kk['peak_rt'],
                y= kk['peak_intensity'],
                mode= 'markers',

                text= textdata,
                marker=dict(
                    size=size,
                    sizemode='area',
                    sizeref=2.*max(size)/(40.**2),
                    sizemin=4
                )

            )])
            # fig = px.scatter(kk, x="peak_rt", y="peak_intensity", size='width')
            fig.write_html(f"{xc}_peaks.html")

            kk= kk.sort_values(by= 'peak_intensity', ascending = False)
            pl= kk[0:10]
            pl.apply(lambda row: make_xic(row['plot'], row['number'], xc), axis= 1)

            listdataframe= pd.concat([listdataframe, dd])

        return listdataframe



    def clusters(self):
        dataframes= self.read_values()
        print('yeswell')
        print(dataframes.head())
        # gmm = mixture.GaussianMixture(n_components= 7451,
        #                           covariance_type="diag")
        # gmm.fit(dataframes.loc[:, ["peak_mz", "peak_rt"]])

        # minimum= dataframes['peak_rt'].min()
        #
        #
        # maximum= dataframes['peak_rt'].max()
        #
        # thresold= 0.3
        # thresold1= 0.005
        # #
        # # ranges= arange(minimum, maximum, thresold)
        # # bins = pd.cut(dataframes['peak_rt'], ranges)
        # grouped= dataframes.groupby('initial')
        # listd= []
        #
        # for x, y in grouped:
        #     listd.append(y)
        #
        # dfx1= listd[0].sort_values(by= 'peak_mz')
        # dfx2= listd[1].sort_values(by= 'peak_mz')
        # dfx1= dfx1[['peak_mz', 'peak_rt', 'peak_intensity', 'filename', 'initial']]
        # dfx2= dfx2[['peak_mz', 'peak_rt', 'peak_intensity', 'filename', 'initial']]
        #
        # df10= pd.merge_asof(dfx1, dfx2, on='peak_mz', tolerance=thresold1, direction='nearest')
        # df10= df10.dropna()
        # print(df10.index.size)
        # df10['difference']= df10['peak_rt_x'] - df10['peak_rt_y']
        # df10['difference']= df10['difference'].map(lambda x: abs(x))
        # df10= df10[df10['difference'] < 0.3]
        # print(df10.head(19))
        #
        # df10.to_csv('full_analysis.csv')


        # proba = gmm.predict_proba(dataframes.loc[:, ["peak_mz", "peak_rt"]].values)
        # print(proba)
        # rows, cols = proba.shape
        # print(rows)
        # print(cols)
        # if rows != cols:
        #     fill = np.zeros(shape=(rows- cols, cols))
        #     proba = np.vstack((proba, fill))
        # _, best_cluster = linear_sum_assignment(proba)
        # best_cluster = best_cluster[:rows]
        # cluster = pd.Series(data=best_cluster, index=dataframes.index)
        # dataframes= dataframes.merge(cluster.rename('new'), how= 'inner', left_index= True, right_index= True)
        # dataframes= dataframes.sort_values(by= 'new')
        # print(dataframes.tail(30))
        # dataframes.to_csv('peaks_correspondance.csv')
        #



        # min_samples= dataframes['initial'].nunique()
        # minimum= dataframes['peak_mz'].min()
        # maximum= dataframes['peak_mz'].max()
        # thresold= 0.005
        # dataframes= dataframes.sort_values(by= 'peak_mz')
        # ranges= arange(minimum, maximum, thresold)
        # bins = pd.cut(dataframes['peak_mz'], ranges)
        # grouped= dataframes.groupby(bins)
        # df11= pd.DataFrame()
        # for u, w in grouped:
        #     try:
        #         ft_points = w.loc[:, ["peak_mz", "peak_rt"]].copy()
        #         dbscan = DBSCAN(eps=0.05, min_samples=min_samples, n_jobs= -1)
        #         dbscan.fit(ft_points)
        #         cluster = pd.Series(data=dbscan.labels_, index=w.index)
        #         cluster= cluster.map(lambda x: str(x))
        #         cluster= cluster.map(lambda x: ''.join([x, str(u)]))
        #         w= w.merge(cluster.rename('new'), how= 'inner', left_index= True, right_index= True)
        #         w= w.sort_values(by= 'new')
        #         df11= pd.concat([df11, w])
        #
        #     except Exception as e:
        #         pass
        # df11= df11[['peak_mz', 'peak_rt', 'peak_intensity', 'initial', 'new', 'filename']]
        # df11= df11[~df11['new'].str.contains('-1', regex=False)]
        # grouped2= df11.groupby('new')['peak_mz'].agg('mean')
        # grouped3= df11.groupby('new')['peak_rt'].agg('mean')
        # grouped4= df11.groupby('new')['peak_intensity'].apply(np.copy)
        # grouped5= df11.groupby('new')['filename'].apply(np.copy)
        #
        # df18= pd.concat([grouped2, grouped3, grouped4, grouped5], axis= 1)
        # df18['size']= df18['filename'].map(lambda x: len(x))
        # df18= df18[df18['size']< 4]
        # df18['size']= df18['filename'].map(lambda x: set(x))
        # df18['size']= df18['size'].map(lambda x: len(x))
        # df18= df18[df18['size'] > 1]

        # df18['mapping']= df18.apply(lambda row: dict(zip(row['peak_intensity'], row['filename'])))
        # df18['700_PH1_WH3.mzXML']= df18['mapping'].map(lambda x: x.get('700_PH1_WH3.mzXML'))
        # df18['699_PH1_WH2.mzXML']= df18['mapping'].map(lambda x: x.get('699_PH1_WH2.mzXML'))

        #
        #
        #
        # df18.to_csv('peaks_correspondance.csv')


if __name__ == "__main__":

    obj123= Combine('./data')
    obj123.clusters()
