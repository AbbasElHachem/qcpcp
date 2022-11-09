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

import sys
import time
import timeit
# import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import spatial
import matplotlib.dates as mdates

import multiprocessing as mp
from adjustText import adjust_text
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.dates import DateFormatter
from pykrige.ok import OrdinaryKriging as OKpy


plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'axes.labelsize': 24})

#=======================================================================
#
#=======================================================================

out_dir = r"X:\staff\elhachem\GitHub\qcpcp\example_results"
path_to_tranf_factor = (
    r"X:\staff\elhachem\GitHub\qcpcp\example_results"
    r"\lambda_box_cox_trans_func.csv")

dwd_in_coords_df = pd.read_csv(
    r"X:\staff\elhachem\GitHub\qcpcp\example_data"
    r"\DWD_coords_utm32.csv",
    index_col=0,
    sep=',')
dwd_ids = dwd_in_coords_df.index.to_list()


stn_df_all = pd.read_csv(r"X:\staff\elhachem\GitHub\qcpcp"
                         r"\example_data\dwd_60min_pcp.csv",
                         sep=';',
                         index_col=0,
                         parse_dates=True,
                         infer_datetime_format=True)
# stn_id_to_plot = 'P00399'

# number of values to interpolate
nbr_biggest_vals = 4
temp_freq_to_use = '60min'
# maximum radius for station selection
neigbhrs_radius_dwd = 10e4
nbr_neighbours = 30

vg_range = 4e4
vg_model_str = 'spherical'

date_form = DateFormatter("%Y-%m-%d %H:%M")

df_transf_factor = pd.read_csv(
    path_to_tranf_factor, sep=';',
    index_col=0)
factor_temp_agg = df_transf_factor.loc[temp_freq_to_use, :]
print(factor_temp_agg)

target_stn = 'P03231'
#==========================================================================
# read data and get station ids


def plot_config_event(event_date,
                      stn_one_id, stn_one_xcoords,
                      stn_one_ycoords,
                      ppt_stn_one, interp_ppt,
                      stns_ngbrs, ppt_ngbrs,
                      x_ngbrs, y_ngbrs,
                      out_dir, save_acc='',
                      transf_data=False,
                      show_axis=False):

    scalebar = ScaleBar(1,
                        label_loc='bottom',
                        location='lower right',
                        border_pad=1,
                        box_alpha=0.5,
                        pad=0.5
                        )
    # 1 pixel = 0.2 meter
    # plt.gca().add_artist(scalebar)
    plt.ioff()
    texts = []
    fig = plt.figure(figsize=(16, 16), dpi=300)
    ax = fig.add_subplot(111)

    ax.scatter(stn_one_xcoords,
               stn_one_ycoords,
               c='orange',
               marker='X', alpha=0.9,
               s=80, label='CR=%0.2f' % save_acc)

    # plot first station
    ax.scatter(stn_one_xcoords,
               stn_one_ycoords,
               c='red',
               marker='X',
               s=80, label='Observed')

    ppt_stn_one_float_format = '% 0.2f' % ppt_stn_one
    texts.append(ax.text(stn_one_xcoords,
                         stn_one_ycoords + 1000,
                         ppt_stn_one_float_format,
                         color='r'))
    # interp
    ax.scatter(stn_one_xcoords,
               stn_one_ycoords,
               c='m',
               marker='X', alpha=0.99,
               s=80, label='Interpolated')

    ppt_interp = '% 0.2f' % interp_ppt

    texts.append(ax.text(stn_one_xcoords + 1000,
                         stn_one_ycoords,
                         ppt_interp,
                         color='m'))

    texts.append(ax.text(stn_one_xcoords,
                         stn_one_ycoords,
                         save_acc,
                         color='orange'))

    for ix, _ in enumerate(stns_ngbrs):

        val_float_format = '% 0.1f ' % (ppt_ngbrs[ix])
#                     val_float_format = '% 0.1f mm' % (time_idx_val[0])

        if ppt_ngbrs[ix] > 1:
            ax.scatter(x_ngbrs[ix],
                       y_ngbrs[ix],
                       c='g',
                       marker='o',
                       s=45)
            texts.append(ax.text(x_ngbrs[ix],
                                 y_ngbrs[ix],
                                 val_float_format,
                                 color='g'))
        else:
            ax.scatter(x_ngbrs[ix],
                       y_ngbrs[ix],
                       c='b',
                       marker='o',
                       s=45)
            texts.append(ax.text(x_ngbrs[ix],
                                 y_ngbrs[ix],
                                 val_float_format,
                                 color='b'))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='->',
                                color='red', lw=0.25),
                fontsize=20)

#     ax.set_title('%s %s n=%d' %
#                  (stn_one_id, event_date, len(stns_ngbrs)))
    if not show_axis:
        ax.set_xticks(ticks=[])
        ax.set_yticks(ticks=[])

    ax.grid(alpha=0.75)
    # ax.set_aspect(1.0)
    ax.legend(loc='lower left', frameon=False)
    ax.add_artist(scalebar)
    # plt.show()
    # plt.tight_layout()

    if transf_data:
        add_acc = 'transf'
    else:
        add_acc = ''
    plt.savefig(os.path.join(out_dir,
                             'stn_%s_space_config_%s2_%s.png' % (
                                 stn_one_id,
                                 str(event_date).replace(
                                     ':', '_').replace('-', '_'),
                                 add_acc)),
                bbox_inches='tight', pad_inches=.2)

    plt.close()
    return


#==============================================================================


# all_dwd_stns_ids = [stn_id_to_plot]
# for ix, stn_id in enumerate(dwd_ids):
#     print(ix, len(dwd_ids))
stn_id = target_stn
# read df and get biggest values and time stamp
stn_df = stn_df_all.loc[:, stn_id]
stn_df_biggest = pd.DataFrame(
    index=stn_df.index, columns=[stn_id])

for year_ in np.unique(stn_df.index.year):
    # print(year_)
    idx_yearly = np.where(stn_df.index.year == year_)[0]
    stn_df_year = stn_df.iloc[idx_yearly].nlargest(
        n=nbr_biggest_vals).sort_index()
    stn_df_biggest.loc[stn_df_year.index,
                       stn_id] = stn_df_year.values.ravel()

stn_df_biggest.dropna(how='all', inplace=True)
stn_df_biggest = stn_df_biggest[stn_df_biggest.values > 0]
# stn_df_biggest.sort_values(by=stn_id)[:-15].sort_index()
# create empty for saving results
stn_df_biggest_interp = np.round(
    stn_df_biggest.copy(deep=True), 2)
print('Events to cross validate', stn_df_biggest_interp)

# find neighbours within radius
x_dwd_interpolate = dwd_in_coords_df.loc[stn_id, 'X']
y_dwd_interpolate = dwd_in_coords_df.loc[stn_id, 'Y']

all_dwd_stns_except_interp_loc = [
    stn for stn in dwd_ids if stn != stn_id]

for ex, event_date in enumerate(stn_df_biggest_interp.index):
    try:
        print(event_date, ex, ' / ', len(stn_df_biggest_interp.index))

        df_ngbrs = stn_df_all.loc[event_date,
                                  all_dwd_stns_except_interp_loc
                                  ].dropna(axis=0, how='all')
        #df_ngbrs = resampleDf(df_ngbrs, '1440min')
        ids_with_data = df_ngbrs.index.to_list()
        # df_ngbrs.values.max()
        # get ppt at all neighbours
        x_dwd_all = dwd_in_coords_df.loc[ids_with_data, 'X'].dropna(
        ).values
        y_dwd_all = dwd_in_coords_df.loc[ids_with_data, 'Y'].dropna(
        ).values
        # dwd_neighbors_coords = np.c_[x_dwd_all.ravel(), y_dwd_all.ravel()]
        # points_tree = spatial.KDTree(dwd_neighbors_coords)
        # _, indices = points_tree.query(
        #     np.array([x_dwd_interpolate, y_dwd_interpolate]),
        #     k=30 + 1)

        # ids_ngbrs = [ids_with_data[ix_ngbr] for ix_ngbr in indices]
        # df_ngbrs = df_ngbrs.loc[event_date, ids_ngbrs].dropna(axis=0, how='all')

        # stns_ngbrs = df_ngbrs.columns.to_list()
        # x_ngbrs = dwd_in_coords_df.loc[stns_ngbrs, 'X']
        # y_ngbrs = dwd_in_coords_df.loc[stns_ngbrs, 'Y']

        ppt_ngbrs = df_ngbrs.values.ravel()

        ppt_ngbrs_tranf = ppt_ngbrs**factor_temp_agg.values

        # Interpolation
        ppt_var = np.var(ppt_ngbrs_tranf)
        if ppt_var == 0:
            ppt_var = 0.1

        # using DWD data
        OK_dwd_netatmo_crt = OKpy(
            x_dwd_all, y_dwd_all,
            ppt_ngbrs_tranf,
            variogram_model=vg_model_str,
            variogram_parameters={
                'sill': ppt_var,
                'range': vg_range,
                'nugget': 0},
            exact_values=True)

        # sigma = _
        interpolated_vals_O0, est_var = OK_dwd_netatmo_crt.execute(
            'points', np.array([x_dwd_interpolate]),
            np.array([y_dwd_interpolate]))

        interpolated_vals_O0 = interpolated_vals_O0.data
        interpolated_vals_O0[interpolated_vals_O0 < 0] = 0

        obsv_ppt_0 = stn_df_biggest_interp.loc[event_date, stn_id]
        obsv_ppt_0_r2 = obsv_ppt_0**factor_temp_agg.values
        #==========================================================
        # # calcualte standard deviation of estimated values
        #==========================================================
        std_est_vals_O0 = np.sqrt(est_var.data)
        # calculate difference observed and estimated  # values
        try:
            diff_obsv_interp_O0 = np.round(np.abs(
                (interpolated_vals_O0 - obsv_ppt_0_r2
                 ) / std_est_vals_O0), 2)
        except Exception:
            print('ERROR STD')

        #===========================================================
        # interpolating original data
        #===========================================================
        OK_dwd_netatmo_crt = OKpy(
            x_dwd_all, y_dwd_all,
            ppt_ngbrs,
            variogram_model=vg_model_str,
            variogram_parameters={
                'sill': ppt_var,
                'range': vg_range,
                'nugget': 0},
            exact_values=True)

        # sigma = _
        interpolated_vals_O0_orig, est_var_orig = OK_dwd_netatmo_crt.execute(
            'points', np.array([x_dwd_interpolate]),
            np.array([y_dwd_interpolate]))

        interpolated_vals_O0_orig = interpolated_vals_O0_orig.data
        interpolated_vals_O0_orig[interpolated_vals_O0_orig < 0] = 0
        std_est_vals_O0_orig = np.sqrt(est_var_orig.data)

        diff_obsv_interp_O0_orig = np.round(np.abs(
            (interpolated_vals_O0_orig - obsv_ppt_0
             ) / std_est_vals_O0_orig), 2)

        stn_df_biggest_interp.loc[
            event_date, 'Interp'] = interpolated_vals_O0_orig

        if diff_obsv_interp_O0 < 3:
            print('not outlier')
            stn_df_biggest_interp.loc[event_date,
                                      'Outlier O0'] = diff_obsv_interp_O0
        if diff_obsv_interp_O0 >= 3:
            print(obsv_ppt_0)
            if diff_obsv_interp_O0 >= 3:
                stn_df_biggest_interp.loc[
                    event_date,
                    'Outlier O0'] = diff_obsv_interp_O0

            # orig data
            event_start = event_date + pd.Timedelta(hours=-24)
            event_end = event_date + pd.Timedelta(hours=24)
            orig_hourly = stn_df_all.loc[
                event_start:event_end, stn_id]
            # np.argmax(orig_hourly.values)
            # orig_hourly.iloc[1111, :]
            orig_ngbrs_hourly = stn_df_all.loc[
                event_start:event_end, ids_with_data]

            max_ppt = max(np.nanmax(
                orig_hourly.values.ravel()), np.nanmax(
                orig_ngbrs_hourly.values.ravel()))

            plt.ioff()
            fig = plt.figure(figsize=(16, 16), dpi=200)
            # orig_hourly.plot()
            ax = fig.add_subplot(111)

            for _nbr in orig_ngbrs_hourly.columns:

                ax.plot(orig_ngbrs_hourly.index,
                        orig_ngbrs_hourly.loc[
                            :,
                            _nbr].values.ravel(),

                        c='lightblue')

            ax.bar(
                x=orig_ngbrs_hourly.index,
                height=orig_ngbrs_hourly.loc[:, _nbr].values.ravel(),
                color='b',
                edgecolor='lightblue',
                width=0.005,
                label='Neighbors')

            ax.bar(
                x=orig_hourly.index,
                height=orig_hourly.values.ravel(),
                color='red',
                edgecolor='darkred',
                width=0.0051,
                label='Center station')
            ax.grid(alpha=0.75, linestyle='--')
            yvals = np.arange(0, np.nanmax(
                orig_hourly.values.ravel()) + 1, 2)
            ax.set_yticks(ticks=yvals)
            ax.set_yticklabels(labels=yvals)

            # ax.set_xticks(ticks=orig_ngbrs_hourly.index[::10])
            # ax.set_xticklabels(orig_ngbrs_hourly.index[::10],
            # rotation=45)
            ax.xaxis.set_major_locator(
                mdates.HourLocator(interval=24))
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_minor_locator(
                mdates.MinuteLocator(interval=10))

            ax.set_ylabel('Precipitation %s' % r' $[mm.min^{-1}]$')
            plt.legend(loc=0, fontsize=18)
            # plt.show()
            plt.savefig(os.path.join(out_dir,
                                     'stn_%s_ngbrs_%s.png' % (
                                         stn_id,
                                         str(event_date).replace(
                                             ':', '_').replace('-', '_'),
                                     )),
                        bbox_inches='tight', pad_inches=.2)
            plt.close()
            print('plotting spacial config')

            plot_config_event(event_date=event_date,
                              stn_one_id=stn_id, stn_one_xcoords=x_dwd_interpolate,
                              stn_one_ycoords=y_dwd_interpolate,
                              ppt_stn_one=obsv_ppt_0_r2,
                              interp_ppt=interpolated_vals_O0,
                              stns_ngbrs=ids_with_data,
                              ppt_ngbrs=ppt_ngbrs_tranf,
                              x_ngbrs=x_dwd_all,
                              y_ngbrs=y_dwd_all,
                              out_dir=out_dir,
                              save_acc=diff_obsv_interp_O0,
                              transf_data=False,
                              show_axis=True)

            # plot_config_event(event_date=event_date,
            #                   stn_one_id=stn_id, stn_one_xcoords=x_dwd_interpolate,
            #                   stn_one_ycoords=y_dwd_interpolate,
            #                   ppt_stn_one=obsv_ppt_0,
            #                   interp_ppt=interpolated_vals_O0_orig,
            #                   stns_ngbrs=stns_ngbrs,
            #                   ppt_ngbrs=ppt_ngbrs,
            #                   x_ngbrs=x_ngbrs,
            #                   y_ngbrs=y_ngbrs,
            #                   out_dir=out_dir,
            #                   save_acc=diff_obsv_interp_O0_orig,
            #                   transf_data=False,
            #                   show_axis=True)

    except Exception as msg:
        print('Error at the end', msg)
        continue

    plt.close()

# =========================================================================
stn_df_biggest_interp.dropna(how='any', inplace=True)
