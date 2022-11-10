# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    Cross-validate the highest 4 biggest annual values
Purpose: Data quality

Created on: 2020-06-18

Parameters
----------
DWD daily or sub-hourly resolutions

Returns
-------
Interpolated data highest extremes

"""

__author__ = "Abbas El Hachem"
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
__version__ = 0.1
__last_update__ = '02.03.2020'
# =============================================================

import os

# import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skew

from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
# ################### import modules as  ##############################


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'axes.labelsize': 14})

# Data Path
# =============================================================================
out_dir = (r"X:\staff\elhachem\GitHub\qcpcp\example_results")
# main_dir = Path(r"/home/IWS/hachem/ClimXtreme")
os.chdir(out_dir)


path_to_coords = pd.read_csv(
    r"X:\staff\elhachem\GitHub\qcpcp\example_data\DWD_coords_utm32.csv",
    index_col=0,
    sep=',')
dwd_ids = path_to_coords.index.to_list()

#"X:\staff\elhachem\GitHub\qcpcp\"
#==========================================================================
# read data and get station ids

# HDF5_dwd_ppt = HDF5(path_to_dwd_hdf5_orig)
# stn_df_all = HDF5_dwd_ppt.get_pandas_dataframe(dwd_ids)
#
# stn_df_all = stn_df_all.loc['2005-01-01 01:00:00':'2015-12-31 23:00:00', :]
# stn_df_all.to_csv(r"X:\staff\elhachem\GitHub\qcpcp\example_data\dwd_60min_pcp.csv",
#               sep=';',
#               float_format='%0.2f')

stn_df_all = pd.read_csv(r"X:\staff\elhachem\GitHub\qcpcp"
                         r"\example_data\dwd_60min_pcp.csv",
                         sep=';',
                         index_col=0,
                         parse_dates=True,
                         infer_datetime_format=True)

aggs_list = ['60min', '120min', '180min',
             '240min', '360min', '720min', '1440min']


df_lambda = pd.DataFrame(index=aggs_list,
                         columns=stn_df_all.columns)
df_skews_orig = pd.DataFrame(index=aggs_list,
                             columns=stn_df_all.columns)
df_skews_transf = pd.DataFrame(index=aggs_list,
                               columns=stn_df_all.columns)


for stn in tqdm(stn_df_all.columns):
    print(stn)
    # break
    stn_df = stn_df_all.loc[:, stn]
    stn_df_nonan = stn_df.dropna(how='any')
    stn_df_nonan = stn_df_nonan[stn_df_nonan >= 0].dropna()
    for agg in aggs_list:
        vals = []
        # break
        # agg = '1min'
        dwd_pcp_res = stn_df_nonan.resample(agg).sum()
        orig_data_skew = skew(dwd_pcp_res)
        #print(agg, 'orig_data_skew: ', orig_data_skew)

        if dwd_pcp_res.values.size > 0:
            try:
                cdf = ECDF(dwd_pcp_res.values.ravel())
                x0 = cdf.x[1:]
                y0 = cdf.y[1:]
                p0 = y0[np.where(x0 == 0)][-1]

                # generate 1000 samples
                sample_a = []

                for i in (range(10)):

                    # s = norm.pdf(std_norm__range, 0, 1)
                    s = np.random.standard_normal(dwd_pcp_res.size)

                    idx_below = np.where(s < p0)[0]
                    idx_abv = np.where(s >= p0)[0]
                    freq_below = idx_below.size / dwd_pcp_res.size
                    freq_abv = idx_abv.size / dwd_pcp_res.size
                    sample_a.append(freq_below)

                a_mean = np.mean(sample_a)
                s[s < a_mean] = 0
                # plt.hist(s)
                skew_stand_norm = skew(s)
                orig_data_skew = skew(dwd_pcp_res)

                df_skews_orig.loc[agg, stn] = orig_data_skew

                range_max = 20

                # min_dif = 5

                skewness_diff_list = [orig_data_skew]
                for i in range(1, range_max):
                    # break
                    lamnda_test = 1 / i
                    cube_root = (dwd_pcp_res.values.ravel())**(lamnda_test)
                    # cube_root2 = cube_root/lamnda_test
                    skewness_diff = np.sqrt(
                        (skew(cube_root) - skew_stand_norm)**2)
                    skewness_diff_list.append(skewness_diff)
                    #print(i, skewness_diff)
                    # if :
                    # print(i)
                    if (skewness_diff_list[i - 1] - skewness_diff_list[i] < 0.01):

                        try:
                            # print('final i', i - 1,
                            #       skewness_diff_list[i - 1])

                            if i == 0:
                                i = 2
                            lambda_final = 1 / (i - 1)
                        except Exception as msg:
                            lambda_final = 1
                        cube_root = (dwd_pcp_res.values.ravel()
                                     )**(lambda_final)
                        break

                df_lambda.loc[agg, stn] = lambda_final
                df_skews_transf.loc[agg, stn] = skew(cube_root)

                #print('y/3', skew(cube_root))
            except Exception as msg:
                print(msg, stn)
                continue


#=======================================================================
#
#=======================================================================

df_lambda = df_lambda[df_lambda < 1]
df_lambda = df_lambda[df_lambda > 0]

df_lambda.mean(axis=1).to_csv(r'lambda_box_cox_trans_func.csv',
                              sep=';',
                              float_format='%0.3f')

df_skews_orig = df_skews_orig[df_skews_orig > 0]
df_skews_transf = df_skews_transf[df_skews_transf > 0]

plt.ioff()
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_subplot(111)

for stn in df_lambda.columns:
    if df_lambda.loc[:, stn].dropna().size > 0:
        plt.plot(df_lambda.loc[:, stn],
                 alpha=0.5, c='grey',
                 linestyle='-', linewidth=1)

plt.plot(df_lambda.mean(axis=1), c='r',
         marker='*', markersize=4,
         label='Mean Transformation Factor',
         linewidth=4)


plt.ylabel(r'Tranformation factor [$\lambda$]')
plt.xlabel('Temporal resolution')
# plt.title('Fitted transformation factor for %d stations '
#           % df_lambda.columns.size)
plt.grid(alpha=0.25)
plt.legend(loc='lower right')
plt.savefig(r'Transf_factor_lambda.png',
            bbox_inches='tight')
plt.close()


plt.ioff()
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_subplot(111)

for stn in df_skews_transf.columns:
    if df_skews_transf.loc[:, stn].dropna().size > 0:
        plt.plot(df_skews_transf.loc[:, stn],
                 marker='+', c='k', alpha=.7)
        plt.plot(df_skews_orig.loc[:, stn],
                 marker='.', c='grey', alpha=0.7)

plt.plot(df_skews_orig.mean(axis=1), marker='.',
         c='b', alpha=0.95,
         label='Average Original Skewness',
         linewidth=4)

plt.plot(df_skews_orig.median(axis=1), marker='.',
         c='g', alpha=0.95,
         label='Median Original Skewness',
         linewidth=4)

plt.plot(df_skews_transf.mean(axis=1), marker='+',
         c='r', alpha=0.95,
         label='Average Transformed Skewness',
         linewidth=4)
plt.ylim([0, 200])
plt.ylabel(r'Skewness')
plt.xlabel('Temporal resolution')
# plt.title('Skewness of original and transformed data for %d stations ' %
#           df_lambda.columns.size)
plt.grid(alpha=0.25)
plt.legend(loc='upper right')
plt.savefig(r'Skew_before_after.png',
            bbox_inches='tight')
