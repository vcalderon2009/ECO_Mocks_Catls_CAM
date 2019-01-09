#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-12-08
# Last Modified: 2018-12-08
# Vanderbilt University
from __future__ import absolute_import, division, print_function 
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""

"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import web_utils       as cweb
from cosmo_utils.utils import stats_funcs     as cstats
from cosmo_utils.utils import geometry        as cgeom
from cosmo_utils.mock_catalogues import catls_utils as cmcu
from cosmo_utils.mock_catalogues import mags_calculations as cmags

import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('text', usetex=True)
import seaborn as sns
#sns.set()
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)
from tqdm import tqdm
from datetime import datetime

# Project packages
from src.survey_utils import ReadSurvey
import hmf
import astropy.cosmology as astrocosmo
import astropy.constants as ac
import astropy.units     as u
import astropy.table     as astro_table
import requests
from collections import Counter
import subprocess
from tqdm import tqdm
from scipy.io.idl import readsav
from astropy.table import Table
from astropy.io import fits
import copy
from multiprocessing import Pool, Process, cpu_count
from scipy.interpolate import interp1d
import tarfile
from glob import glob

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm

## --------- General functions ------------##

class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        """
        Modifier for `argparse` help parameters, that sorts them alphabetically
        """
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _check_pos_val(val, val_min=0):
    """
    Checks if value is larger than `val_min`

    Parameters
    ----------
    val : `int` or `float`
        Value to be evaluated by `val_min`

    val_min: `float` or `int`, optional
        minimum value that `val` can be. This value is set to `0` by default.

    Returns
    -------
    ival : `float`
        Value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError : Raised if `val` is NOT larger than `val_min`
    """
    ival = float(val)
    if ival <= val_min:
        msg  = '`{0}` is an invalid input!'.format(ival)
        msg += '`val` must be larger than `{0}`!!'.format(val_min)
        raise argparse.ArgumentTypeError(msg)

    return ival

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Description of Script'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    ## Variables
    # Size of the cube
    parser.add_argument('-sizecube',
                        dest='size_cube',
                        help='Length of simulation cube in Mpc/h',
                        type=float,
                        default=130.)
    ## Type of Abundance matching
    parser.add_argument('-abopt',
                        dest='catl_type',
                        help='Type of Abund. Matching used in catalogue',
                        type=str,
                        choices=['mr'],
                        default='mr')
    # Median Redshift
    parser.add_argument('-zmed',
                        dest='zmedian',
                        help='Median Redshift of the survey',
                        type=float,
                        default=0.)
    # Type of survey
    parser.add_argument('-survey',
                        dest='survey',
                        help='Type of survey to produce. Choices: A, B, ECO',
                        type=str,
                        choices=['A','B','ECO'],
                        default='ECO')
    # Halo definition
    parser.add_argument('-halotype',
                        dest='halotype',
                        help='Type of halo definition.',
                        type=str,
                        choices=['mvir','m200b'],
                        default='mvir')
    # Cosmology used for the project
    parser.add_argument('-cosmo',
                        dest='cosmo_choice',
                        help='Cosmology to use. Options: 1) Planck, 2) LasDamas',
                        type=str,
                        default='Planck',
                        choices=['Planck','LasDamas'])
    # Halomass function
    parser.add_argument('-hmf',
                        dest='hmf_model',
                        help='Halo Mass Function choice',
                        type=str,
                        default='warren',
                        choices=['warren','tinker08'])
    ## Redshift-space distortions
    parser.add_argument('-zspace',
                        dest='zspace',
                        help="""
                        Option for adding redshift-space distortions (RSD).
                        Options: (1) = No RSD, (2) With RSD""",
                        type=int,
                        choices=[1,2],
                        default=2)
    ## Minimum of galaxies in a group
    parser.add_argument('-nmin',
                        dest='nmin',
                        help='Minimum number of galaxies in a galaxy group',
                        type=int,
                        choices=range(1,1000),
                        metavar='[1-1000]',
                        default=1)
    ## Perpendicular Linking Length
    parser.add_argument('-l_perp',
                        dest='l_perp',
                        help='Perpendicular linking length',
                        type=_check_pos_val,
                        default=0.07)
    ## Parallel Linking Length
    parser.add_argument('-l_para',
                        dest='l_para',
                        help='Parallel linking length',
                        type=_check_pos_val,
                        default=1.1)
    ## Random Seed
    parser.add_argument('-seed',
                        dest='seed',
                        help='Random seed to be used for the analysis',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help="""
                        Delete files created by the script, in case the exist 
                        already""",
                        type=_str2bool,
                        default=False)
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cfutils.Program_Msg(__file__))
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        type=_str2bool,
                        default=False)
    ## Parsing Objects
    args = parser.parse_args()

    return args

def param_vals_test(param_dict):
    """
    Checks if values are consistent with each other.

    Parameters
    -----------
    param_dict : `dict`
        Dictionary with `project` variables

    Raises
    -----------
    ValueError : Error
        This function raises a `ValueError` error if one or more of the 
        required criteria are not met
    """
    ## Size of the cube
    assert(param_dict['size_cube'] == 130.)

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

def add_to_dict(param_dict):
    """
    Aggregates extra variables to dictionary

    Parameters
    ----------
    param_dict : `dict`
        dictionary with input parameters and values

    Returns
    ----------
    param_dict : `dict`
        dictionary with old and new values added
    """
    ## Central/Satellite designations
    cens = int(1)
    sats = int(0)
    ##
    ## ECO-related files
    url_catl = 'http://lss.phy.vanderbilt.edu/groups/data_eco_vc/'
    cweb.url_checker(url_catl)
    ##
    ## Mock cubes - Path
    url_mock_cubes = 'http://lss.phy.vanderbilt.edu/groups/data_eco_vc/ECO_CAM/ECO/'
    cweb.url_checker(url_mock_cubes)
    ## Survey name
    if param_dict['survey'] == 'ECO':
        survey_name = 'ECO'
    else:
        survey_name = 'RESOLVE_{0}'.format(param_dict['survey'])
    ##
    ## Plotting constants
    plot_dict = plot_const()
    ##
    ## Variable constants
    const_dict = val_consts()
    # Dictionary of Halobias files
    hb_files_dict = param_dict['survey_args'].halobias_files_dict()
    n_hb_files    = len(hb_files_dict.keys())
    # Cosmological model and Halo Mass function
    cosmo_model = param_dict['survey_args'].cosmo_create()
    # Redshift and Comoving Distances
    z_dc_pd = param_dict['survey_args'].comoving_z_distance()
    # Mass Function
    mf_pd = param_dict['survey_args'].hmf_calc()
    # Survey Coordinate dictionary
    param_dict = survey_specs(param_dict)
    ##
    ## Saving to dictionary
    param_dict['cens'          ] = cens
    param_dict['sats'          ] = sats
    param_dict['url_catl'      ] = url_catl
    param_dict['url_mock_cubes'] = url_mock_cubes
    param_dict['plot_dict'     ] = plot_dict
    param_dict['const_dict'    ] = const_dict
    param_dict['survey_name'   ] = survey_name
    param_dict['hb_files_dict' ] = hb_files_dict
    param_dict['n_hb_files'    ] = n_hb_files
    param_dict['cosmo_model'   ] = cosmo_model
    param_dict['z_dc_pd'       ] = z_dc_pd
    param_dict['mf_pd'         ] = mf_pd

    return param_dict

def plot_const():
    """
    Returns constants for plotting

    Returns
    -------
    plot_dict: python dictionary
        dictionary with text labels, fontsizes, etc.
    """
    # Size labels
    size_label = 20
    size_title = 25
    # Markers
    markersize = 3.
    # Dictionary
    plot_dict = {}
    plot_dict['size_label'] = size_label
    plot_dict['title'     ] = size_title
    plot_dict['markersize'] = markersize

    return plot_dict

def val_consts():
    """
    Dictionary with variable constants

    Returns
    --------
    val_dict: python dictionary
        python dictionary with values of variables used throughout the script
    """
    ## Speed of light - Units km/s
    c = ac.c.to(u.km/u.s).value

    const_dict = {}
    const_dict['c'] = c

    return const_dict

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict : `dict`
        Dictionary with current and new paths to project directories
    """
    # Directory of Cosmological files
    cosmo_dir = param_dict['survey_args'].cosmo_outdir(create_dir=True)
    # Halo Mass Function - Directory
    mf_dir = param_dict['survey_args'].mass_func_output_dir(create_dir=True)
    ##
    ## Saving to dictionary
    proj_dict['cosmo_dir'] = cosmo_dir
    proj_dict['mf_dir'   ] = mf_dir
    
    return proj_dict

def tarball_create(hb_ii_name, param_dict, proj_dict, catl_ext='hdf5'):
    """
    Creates TAR object with mock catalogues, figures and README file

    Parameters
    -----------
    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    catl_ext: string, optional (default = 'hdf5')
        file extension of the `mock` catalogues created.

    """
    Prog_msg   = param_dict['Prog_msg' ]
    ## List of Mock catalogues
    catl_path_arr = param_dict['survey_args'].hb_gal_catl_files_list(
        hb_ii_name, catl_kind='memb', perf=False, file_ext=catl_ext)
    ## README file
    # Downloading working README file
    fig_outdir = param_dict['survey_args'].fig_outdir(hb_ii_name,
        create_dir=True)
    fig_file = glob('{0}/*xyz*.pdf'.format(fig_outdir))[0]
    # README file
    readme_dir  = os.path.join( param_dict['survey_args'].proj_dict['base_dir'],
                                'references')
    readme_file = glob('{0}/ECO_Mocks_VC.md'.format(readme_dir))[0]
    # README file
    # readme_file   = os.path.join(   proj_dict['base_dir'],
    #                                 'references',
    #                                 'README_RTD.pdf')
    # cfutils.File_Download_needed(readme_file, param_dict['readme_url'])
    # cfutils.File_Exists(readme_file)
    ## Saving to TAR file
    tar_file_path = param_dict['survey_args'].tar_output_file(hb_ii_name)
    # Opening file
    with tarfile.open(tar_file_path, mode='w:gz') as tf:
        tf.add(readme_file, arcname=os.path.basename(readme_file))
        tf.add(fig_file, arcname=os.path.basename(fig_file))
        for file_kk in catl_path_arr:
            ## Reading in DataFrame
            gal_pd_kk = cfreaders.read_hdf5_file_to_pandas_DF(file_kk)
            ## DataFrame `without` certain columns
            gal_pd_mod = catl_drop_cols(gal_pd_kk)
            ## Saving modified DataFrame to file
            file_mod_kk = file_kk+'.mod'
            cfreaders.pandas_df_to_hdf5_file(gal_pd_mod, file_mod_kk,
                key='gal_catl')
            cfutils.File_Exists(file_mod_kk)
            # Saving to Tar-file
            tf.add(file_mod_kk, arcname=os.path.basename(file_kk))
            # Deleting extra file
            os.remove(file_mod_kk)
    tf.close()
    cfutils.File_Exists(tar_file_path)
    if param_dict['verbose']:
        print('{0} TAR file saved as: {1}'.format(Prog_msg, tar_file_path))

def catl_drop_cols(mockgal_pd):
    """
    Drops certain columns from the galaxy DataFrame

    Parameters
    -----------
    mockgal_pd: pandas DataFrame
        DataFrame containing information for each mock galaxy.
        Includes galaxy properties + group ID

    Returns
    -----------
    gal_pd_mod: pandas DataFrame
        Updated version of the DataFrame containing information for each 
        mock galaxy.

    """
    ## Copies of DataFrames
    gal_pd   = mockgal_pd.copy()
    ## Columns
    gal_cols = ['x','y','z','vx','vy','vz','galid','x_orig','y_orig','z_orig',
                'vel_pec','ra_orig']
    # New object `without` these columns
    gal_pd_mod = gal_pd.loc[:,~gal_pd.columns.isin(gal_cols)].copy()

    return gal_pd_mod



## --------- Halobias File - Analysis ------------##

## Main analysis of the Halobias File
def hb_analysis(ii, hb_ii_name, param_dict, proj_dict):
    """
    Main function that analyzes the Halobias file and constructs a set of
    mock catalogues.

    Parameters
    ------------
    ii : `int`
        Integer of the halobias file being analyzed, after having ordered
        the list of files alphabetically.

    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    param_dict : `dict`
        Dictionary with the `project` variables.

    proj_dict : `dict`
        Dictionary with current and new paths to project directories.

    ext : `str`
        Extension to use for the resulting catalogues.
    """
    ## Extract data from Halobias File
    hb_ii_pd = hb_file_extract_data(hb_ii_name, param_dict)
    ## Carving out geometry of Survey and carrying out the analysis
    if (param_dict['survey'] == 'ECO'):
        eco_geometry_mocks(hb_ii_pd, hb_ii_name, param_dict, proj_dict)
    ## Plotting different catalogues in simulation box
    mockcatls_simbox_plot(hb_ii_name, param_dict, proj_dict)
    ## Luminosity function for each catalogue
    mocks_lum_function(hb_ii_name, param_dict, proj_dict)
    ##
    ## Saving everything to TARBALL
    tarball_create(hb_ii_name, param_dict, proj_dict)


## Reading and extracting data From Halobias file
def hb_file_extract_data(hb_ii_name, param_dict):
    """
    Extracts the data from the Halobias file being analyzed.

    Parameters
    ------------
    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    param_dict : `dict`
        Dictionary with the `project` variables.

    Returns
    ------------
    hb_ii_pd : `pandas.DataFrame`
        DataFrame containing main info from Halobias being analyzed.
    """
    # Halobias filename
    hb_ii_file = param_dict['hb_files_dict'][hb_ii_name]
    # Reading in file
    hb_ii_pd = cfreaders.read_hdf5_file_to_pandas_DF(hb_ii_file)
    # Adding extra halo properties
    hb_ii_pd = hb_extras(hb_ii_pd, param_dict)
    # Distance between central and satellites
    hb_ii_pd = cen_sat_dist_calc(hb_ii_pd, param_dict)

    return hb_ii_pd

## Extra Halo properties
def hb_extras(hb_ii_pd, param_dict):
    """
    Determines the Central/Satellite designation for each galaxy in the
    halobias file.

    Parameters
    ------------
    hb_ii_pd : `pandas.DataFrame`
        DataFrame containing main info from Halobias being analyzed.

    param_dict : `dict`
        Dictionary with the `project` variables.

    Returns
    ------------
    hb_ii_pd : `pandas.DataFrame`
        DataFrame containing main info from Halobias being analyzed +
        new galaxy Central/Satellite designations.
    """
    ## Constants
    failval = num.nan
    ngals   = len(hb_ii_pd)
    cens    = param_dict['cens']
    sats    = param_dict['sats']
    # Initializing new columns of galaxy type and Halo Mass
    hb_ii_pd.loc[:, 'cs_flag'         ] = 0
    ##
    ## Central/Satellite - Indices
    cen_idx = hb_ii_pd.loc[hb_ii_pd['halo_upid'] == -1].index.values
    sat_idx = hb_ii_pd.loc[hb_ii_pd['halo_upid'] != -1].index.values
    ## Cen/Sat Designations
    hb_ii_pd.loc[cen_idx, 'cs_flag'] = cens
    hb_ii_pd.loc[sat_idx, 'cs_flag'] = sats
    ##
    ## Total number of galaxies per halo
    haloid_ngal_counter = Counter(hb_ii_pd['halo_hostid'])
    haloid_arr          = hb_ii_pd['halo_hostid'].values
    haloid_counts       = [[] for x in range(ngals)]
    for gal in tqdm(range(ngals)):
        haloid_counts[gal] = haloid_ngal_counter[haloid_arr[gal]]
    # Assigning to DataFrame
    hb_ii_pd.loc[:, 'haloid_host_ngal'] = num.array(haloid_counts).astype(int)
    ##
    hb_ii_pd.loc[:, 'log_host_mvir'] = num.log10(hb_ii_pd['halo_mvir_host_halo'])

    return hb_ii_pd

## Distance between centrals and satellites
def cen_sat_dist_calc(hb_ii_pd, param_dict):
    """
    Computes the distance between the central galaxy and its corresponding
    satellite galaxies in a given DM halo.

    Parameters
    -----------
    hb_ii_pd : `pandas.DataFrame`
        DataFrame containing main info from Halobias being analyzed.

    param_dict : `dict`
        Dictionary with the `project` variables.

    Returns
    -----------
    hb_ii_pd : `pandas.DataFrame`
        DataFrame containing main info from Halobias being analyzed + info
        on the cen-sat distances.
    """
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Distance-Central Assignment ...'.format(Prog_msg))
    ##
    ## Centrals and Satellites
    cens = param_dict['cens']
    sats = param_dict['sats']
    dist_c_label = 'dist_c'
    dist_sq_c_label = 'dist_c_sq'
    ## Galaxy coordinates
    coords  = ['x'   ,'y'   , 'z'  ]
    coords2 = ['x_sq','y_sq','z_sq']
    ## Unique HaloIDs
    haloid_unq = num.unique(hb_ii_pd.loc[hb_ii_pd['haloid_host_ngal'] > 1,
                            'halo_hostid'])
    n_halo_unq = len(haloid_unq)
    # Desired columns
    hb_cols_select = ['x','y','z', 'cs_flag', 'halo_hostid']
    # Copy of `hb_ii_pd`
    hb_ii_pd_mod = hb_ii_pd[hb_cols_select].copy()
    # Initializing new column in `hb_ii_pd_mod`
    hb_ii_pd_mod.loc[:, dist_sq_c_label] = num.zeros(hb_ii_pd_mod.shape[0])
    # Positions squared
    hb_ii_pd_mod.loc[:, 'x_sq'] = hb_ii_pd_mod['x']**2
    hb_ii_pd_mod.loc[:, 'y_sq'] = hb_ii_pd_mod['y']**2
    hb_ii_pd_mod.loc[:, 'z_sq'] = hb_ii_pd_mod['z']**2
    # Looping over number of haloes
    tqdm_desc = 'Cen-Sat Distances'
    for ii, halo_ii in enumerate(tqdm(haloid_unq, desc=tqdm_desc)):
        # Halo ID subsample
        halo_ii_pd = hb_ii_pd_mod.loc[hb_ii_pd['halo_hostid'] == halo_ii]
        # Cens and Sats DataFrames
        cens_coords = halo_ii_pd.loc[halo_ii_pd['cs_flag'] == cens, coords]
        sats_coords = halo_ii_pd.loc[halo_ii_pd['cs_flag'] == sats, coords]
        sats_idx    = sats_coords.index.values
        # Distance from central galaxy
        cens_coords_mean = cens_coords.mean(axis=0).values
        # Difference in coordinates
        dist_sq_arr = num.sum(
            sats_coords.subtract(cens_coords_mean, axis=1).values**2, axis=1)
        # Assigning distances to each satellite
        hb_ii_pd_mod.loc[sats_idx, dist_sq_c_label] = dist_sq_arr
    ##
    ## Taking the square root of distances
    hb_ii_pd_mod.loc[:, dist_c_label] = (hb_ii_pd_mod[dist_sq_c_label].values)**.5
    # Assigning it to 'hb_ii_pd'
    hb_ii_pd.loc[:, dist_c_label] = hb_ii_pd_mod[dist_c_label].values
    if param_dict['verbose']:
        print('{0} Distance-Central Assignment ... Done'.format(Prog_msg))

    return hb_ii_pd

## --------- Makemock-related - Analysis ------------##

## Geometry of ECO catalogues
def eco_geometry_mocks(hb_ii_pd, hb_ii_name, param_dict, proj_dict):
    """
    Carves out the geometry of the `ECO` survey and produces set 
    of mock catalogues
    
    Parameters
    -------------
    hb_ii_pd : `pandas.DataFrame`
        DataFrame containing information from Halobias + other Halo-related
        information.

    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.
    
    param_dict : `dict`
        Dictionary with the `project` variables.

    proj_dict : `dict`
        Dictionary with info of the paths and directories used throughout
        this project.
    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Creating Mock Catalogues ....'.format(Prog_msg))
    ## Coordinates dictionary
    coord_dict    = param_dict['coord_dict'].copy()
    ## Coordinate and Dataframe lists
    pos_coords_mocks = []
    ##############################################
    ###### ----- X-Y Upper Left Mocks  -----######
    ##############################################
    hb_ul_pd      = copy.deepcopy(hb_ii_pd)
    coord_dict_ul = coord_dict.copy()
    # Coordinates
    coord_dict_ul['ra_min']  = 0.
    coord_dict_ul['ra_max']  = coord_dict_ul['ra_range']
    coord_dict_ul['ra_diff'] = coord_dict_ul['ra_max_real'] - coord_dict_ul['ra_max']
    gap_ul     = 1.
    x_init_ul  = 20.
    y_init_ul  = 0.
    z_init_ul  = 5.
    z_delta_ul = gap_ul + coord_dict_ul['d_th']
    if coord_dict_ul['dec_min'] < 0.:
        z_init_ul += num.abs(coord_dict_ul['dec_min'])
    z_mocks_n_ul = int(num.floor(param_dict['size_cube']/z_delta_ul))
    ## Determining positions
    for kk in range(z_mocks_n_ul):
        pos_coords_mocks.append([   x_init_ul, y_init_ul, z_init_ul,
                                    hb_ul_pd.copy(), coord_dict_ul])
        z_init_ul += z_delta_ul
    ##############################################
    ###### ----- X-Y Upper Right Mocks -----######
    ##############################################
    hb_ur_pd      = copy.deepcopy(hb_ii_pd)
    coord_dict_ur = copy.deepcopy(coord_dict_ul)
    # Coordinates
    coord_dict_ur['ra_min' ] = 180.
    coord_dict_ur['ra_max' ] = 180. + coord_dict_ur['ra_range']
    coord_dict_ur['ra_diff'] = coord_dict_ur['ra_max_real'] - coord_dict_ur['ra_max']
    gap_ur     = 1.
    x_init_ur  = param_dict['size_cube'] - 20.
    y_init_ur  = param_dict['size_cube'] - 3.
    z_init_ur  = 5.
    z_delta_ur = gap_ur + coord_dict_ur['d_th']
    if coord_dict_ur['dec_min'] < 0.:
        z_init_ur += num.abs(coord_dict_ur['dec_min'])
    z_mocks_n_ur = int(num.floor(param_dict['size_cube']/z_delta_ur))
    ## Determining positions
    for kk in range(z_mocks_n_ur):
        pos_coords_mocks.append([   x_init_ur, y_init_ur, z_init_ur,
                                    hb_ur_pd.copy(), coord_dict_ur])
        z_init_ur += z_delta_ur
    ##############################################
    ## Creating mock catalogues
    ##############################################
    ##
    ## ----| Multiprocessing |---- ##
    ##
    ## Number of catalogues
    n_catls = len(pos_coords_mocks)
    # Creating individual catalogues
    for zz_mock, pos_coords_mocks_zz in enumerate(pos_coords_mocks):
        # Making z'th catalogue
        catl_create_main(zz_mock, hb_ii_name, pos_coords_mocks_zz,
            param_dict, proj_dict)
    ##
    ## Reinitializing `param_dict` to None
    if param_dict['verbose']:
        print('{0} Creating Mock Catalogues .... Done'.format(Prog_msg))

## Main function for creating the mock catalogues
def catl_create_main(zz_mock, hb_ii_name, pos_coords_mocks_zz, param_dict,
    proj_dict):
    """
    Distributes the analyis of the creation of mock catalogues into 
    more than 1 processor

    Parameters
    -----------
    zz_mock : `int`
        number of the mock catalogue being analyzed

    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    pos_coords_mocks: tuples, shape (4,)
        tuple with the positons coordinates, coordinate dictionary, 
        and DataFrame to be used

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    -----------

    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    ## Deciding which catalogues to read
    ## Reading in input parameters
    # Copy of 'pos_coords_mocks_zz'
    pos_coords_mocks_zz_copy = copy.deepcopy(pos_coords_mocks_zz)
    # Paramters
    (   x_ii         ,
        y_ii         ,
        z_ii         ,
        hb_ii       ,
        coord_dict_ii) = pos_coords_mocks_zz_copy
    ## Size of cube
    size_cube = float(param_dict['size_cube'])
    ## Cartesian coordinates
    pos_zz = num.asarray([x_ii, y_ii, z_ii])
    ## Formatting new positions
    ## Placing the observer at `pos_zz` and centering coordinates to center 
    ## of box
    for kk, coord_kk in enumerate(['x','y','z']):
        # Keeping original Coordinates
        hb_ii.loc[:, coord_kk + '_orig'] = hb_ii[coord_kk].values
        ## Moving observer
        hb_ii.loc[:,coord_kk] = hb_ii[coord_kk] - pos_zz[kk]
        ## Periodic boundaries
        clf_ii_neg = hb_ii.loc[hb_ii[coord_kk] <= -(size_cube/2.)].index
        clf_ii_pos = hb_ii.loc[hb_ii[coord_kk] >=  (size_cube/2.)].index
        ## Fixing negative values
        if len(clf_ii_neg) != 0:
            hb_ii.loc[clf_ii_neg, coord_kk] += size_cube
        if len(clf_ii_pos) != 0:
            hb_ii.loc[clf_ii_pos, coord_kk] -= size_cube
    ##
    ## Interpolating values for redshift and comoving distance
    ## and adding redshift-space distortions
    (   mock_pd     ,
        mock_zz_file) = makemock_catl(  hb_ii, hb_ii_name, coord_dict_ii,
                                        zz_mock, param_dict, proj_dict)
    ##
    ## Group-finding
    (   mockgal_pd  ,
        mockgroup_pd) = group_finding(  mock_pd, mock_zz_file, 
                                        param_dict, proj_dict)
    ##
    ## Group mass, group galaxy type, and total Mr/Mstar for groups
    (   mockgal_pd  ,
        mockgroup_pd) = group_mass_assignment(mockgal_pd, mockgroup_pd, 
                            param_dict, proj_dict)
    ##
    ## Halo Rvir
    mockgal_pd = halos_rvir_calc(mockgal_pd, param_dict)
    ##
    ## Dropping columns from `mockgal_pd` and `mockgroup_pd`
    ##
    ## Writing output files - `Normal Catalogues`
    writing_to_output_file(mockgal_pd, mockgroup_pd, zz_mock, hb_ii_name,
        param_dict, proj_dict, perf_catl=False)

def makemock_catl(hb_ii, hb_ii_name, coord_dict_ii, zz_mock, param_dict,
    proj_dict):
    """
    Function that calculates distances and redshift-space distortions 
    for the galaxies that make it into the catalogues

    Parameters
    -----------
    hb_ii: pandas DataFrame
        DataFrame with the information on galaxies, along with position coords,
        velocities, etc.

    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    coord_dict_ii: python dictionary
        dictionary with RA, DEC, and other geometrical variables used 
        throughout this script.

    zz_mock: int
        number of the mock catalogue being analyzed

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    -----------
    gal_idx: pandas DataFrame
        Updated Dataframe with new positions, coordinates, etc.

    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Creating Mock Catalogue [{1}] ....'.format(Prog_msg, 
            zz_mock))
    ## Galaxy Directory
    mock_catl_ii_dir = param_dict['survey_args'].catl_output_dir(hb_ii_name,
        catl_kind='gal', perf=False, create_dir=True)
    ## Galaxy File
    mock_catl_pd_file = os.path.join(   mock_catl_ii_dir,
                                        '{0}_{1}_galcatl_cat_{2}.hdf5'.format(
                                            param_dict['survey'],
                                            hb_ii_name,
                                            zz_mock))
    ## Number of galaies
    hb_ngal    = len(hb_ii)
    speed_c = param_dict['const_dict']['c']
    ## Distances from observer to galaxies
    z_dc_pd   = param_dict['z_dc_pd']
    dc_max      = z_dc_pd['dc'].max()
    dc_z_interp = interp1d(z_dc_pd['dc'], z_dc_pd['z'])
    ## Redshift-space distortions
    # Cartesina Coordinates
    cart_gals   = hb_ii[['x' ,'y' ,'z' ]].values
    vel_gals    = hb_ii[['vx','vy','vz']].values
    ## Initializing arrays
    r_dist_arr    = num.zeros(hb_ngal)
    ra_arr        = num.zeros(hb_ngal)
    dec_arr       = num.zeros(hb_ngal)
    cz_arr        = num.zeros(hb_ngal)
    cz_nodist_arr = num.zeros(hb_ngal)
    vel_tan_arr   = num.zeros(hb_ngal)
    vel_tot_arr   = num.zeros(hb_ngal)
    vel_pec_arr   = num.zeros(hb_ngal)
    # Looping over all galaxies
    for kk in tqdm(range(hb_ngal)):
        cz_local = -1.
        ## Distance From observer
        r_dist = (num.sum(cart_gals[kk]**2))**.5
        assert(r_dist <= dc_max)
        ## Velocity in km/s
        cz_local = speed_c * dc_z_interp(r_dist)
        cz_val   = cz_local
        ## Right Ascension and declination
        (   ra_kk,
            dec_kk) = mock_cart_to_spherical_coords(cart_gals[kk], r_dist)
        ## Whether or not to add redshift-space distortions
        if param_dict['zspace'] == 1:
            vel_tot = 0.
            vel_tan = 0.
            vel_pec = 0.
        elif param_dict['zspace'] == 2:
            vr       = num.dot(cart_gals[kk], vel_gals[kk])/r_dist
            cz_val  += vr * (1. + param_dict['zmedian'])
            vel_tot  = (num.sum(vel_gals[kk]**2))**.5
            vel_tan  = (vel_tot**2 - vr**2)**.5
            vel_pec  = (cz_val - cz_local)/(1. + param_dict['zmedian'])
        ##
        ## Saving to arrays
        r_dist_arr   [kk] = r_dist
        ra_arr       [kk] = ra_kk
        dec_arr      [kk] = dec_kk
        cz_arr       [kk] = cz_val
        cz_nodist_arr[kk] = cz_local
        vel_tot_arr  [kk] = vel_tot
        vel_tan_arr  [kk] = vel_tan
        vel_pec_arr  [kk] = vel_pec
    ##
    ## Assigning to DataFrame
    hb_ii.loc[:,'r_dist'   ] = r_dist_arr
    hb_ii.loc[:,'ra'       ] = ra_arr
    hb_ii.loc[:,'dec'      ] = dec_arr
    hb_ii.loc[:,'cz'       ] = cz_arr
    hb_ii.loc[:,'cz_nodist'] = cz_nodist_arr
    hb_ii.loc[:,'vel_tot'  ] = vel_tot_arr
    hb_ii.loc[:,'vel_tan'  ] = vel_tan_arr
    hb_ii.loc[:,'vel_pec'  ] = vel_pec_arr
    ##
    ## Selecting galaxies with `czmin` and `czmax` criteria
    #  Right Ascension
    if coord_dict_ii['ra_min'] < 0.:
        ra_min_mod = coord_dict_ii['ra_min'] + 360.
        mock_pd    = hb_ii.loc[(hb_ii['dec'] >= coord_dict_ii['dec_min']) &
                                (hb_ii['dec'] <= coord_dict_ii['dec_max']) &
                                (hb_ii['abs_rmag'] != 0.) &
                                (hb_ii['abs_rmag'] <= param_dict['mr_limit'])].copy()
        mock_pd    = mock_pd.loc[~( (mock_pd['ra'] < ra_min_mod) &
                                    (mock_pd['ra'] > coord_dict_ii['ra_max']))]
        # ra_idx1 = hb_ii.loc[(hb_ii['ra'] < (coord_dict_ii['ra_min'] + 360))&
        #                      (hb_ii['ra'] >  coord_dict_ii['ra_max'])].index
        # ra_idx1 = ra_idx1.values
        # idx_arr = num.arange(0, hb_ngal)
        # ra_idx  = num.delete(idx_arr, ra_idx1).astype(int)
    elif coord_dict_ii['ra_min'] >= 0.:
        mock_pd = hb_ii.loc[(hb_ii['ra'] >= coord_dict_ii['ra_min']) &
                             (hb_ii['ra'] <= coord_dict_ii['ra_max']) &
                             (hb_ii['dec'] >= coord_dict_ii['dec_min']) &
                             (hb_ii['dec'] <= coord_dict_ii['dec_max']) &
                             (hb_ii['abs_rmag'] != 0.) &
                             (hb_ii['abs_rmag'] <= param_dict['mr_limit'])].copy()
        # ra_idx = hb_ii.loc[(hb_ii['ra'] >= coord_dict_ii['ra_min']) &
        #                     (hb_ii['ra'] <= coord_dict_ii['ra_max'])].index
        # ra_idx = ra_idx.values
    # Declination
    # dec_idx = hb_ii.loc[   (hb_ii['dec'] >= coord_dict_ii['dec_min']) &
    #                         (hb_ii['dec'] <= coord_dict_ii['dec_max'])].index.values
    # mr_idx = hb_ii.loc[hb_ii['abs_rmag'] != 0.].index.values
    # ra_dec_mr_idx = num.intersect1d(num.intersect1d(ra_idx, dec_idx), mr_idx)
    ##
    ## Velocity limits
    mock_pd = mock_pd.loc[  (mock_pd['cz'] >= param_dict['czmin']) & 
                            (mock_pd['cz'] <= param_dict['czmax'])]
    ##
    ## New Catalogue
    if len(mock_pd) != 0:
        ## Chaning RA values
        if coord_dict_ii['ra_min'] < 0.:
            ra_min_limit  = coord_dict_ii['ra_min'] + 360.
            ra_new_arr    = mock_pd['ra'].values
            ra_except_idx = num.where(   (ra_new_arr >= ra_min_limit) &
                                        (ra_new_arr <= 360.))[0]
            ra_new_arr[ra_except_idx] += (-360.) + coord_dict_ii['ra_diff']
            ra_normal_idx = num.where(  (ra_new_arr >= 0.) &
                                        (ra_new_arr <= coord_dict_ii['ra_max']))[0]
            ra_new_arr[ra_normal_idx] += coord_dict_ii['ra_diff']
            ra_neg_idx = num.where(ra_new_arr < 0.)[0]
            if len(ra_neg_idx) != 0.:
                ra_new_arr[ra_neg_idx] += 360.
        elif coord_dict_ii['ra_min'] >= 0.:
            ra_new_arr  = mock_pd['ra'].values
            ra_new_arr += coord_dict_ii['ra_diff']
            ra_neg_idx  = num.where(ra_new_arr < 0.)[0]
            if len(ra_neg_idx) != 0:
                ra_new_arr[ra_neg_idx] += 360.
    ##
    ## Saving new array to DataFrame
    ra_orig_arr = mock_pd['ra'].values
    # Assigning new values for RA
    mock_pd.loc[:,'ra'     ] = ra_new_arr
    mock_pd.loc[:,'ra_orig'] = ra_orig_arr
    ##
    ## Resetting indices
    mock_pd.reset_index(inplace=True, drop=True)
    ##
    ## Assert that coordinates fall within Survey limits
    assert( (mock_pd['ra' ].min() >= coord_dict_ii['ra_min_real']) &
            (mock_pd['ra' ].max() <= coord_dict_ii['ra_max_real']) &
            (mock_pd['dec'].min() >= coord_dict_ii['dec_min'    ]) &
            (mock_pd['dec'].max() <= coord_dict_ii['dec_max'    ]))
    ##
    ## Saving file to Pandas DataFrame
    cfreaders.pandas_df_to_hdf5_file(mock_pd, mock_catl_pd_file, key='galcatl')
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Creating Mock Catalogues [{1}]....Done'.format(Prog_msg,
            zz_mock))

    return mock_pd, mock_catl_pd_file

def group_finding(mock_pd, mock_zz_file, param_dict, proj_dict,
    file_ext='csv'):
    """
    Runs the group finder `FoF` on the file, and assigns galaxies to 
    galaxy groups

    Parameters
    -----------
    mock_pd: pandas DataFrame
        DataFrame with positions, velocities, and more for the 
        galaxies that made it into the catalogue

    mock_zz_file: string
        path to the galaxy catalogue

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    file_ext: string, optional (default = 'csv')
        file extension for the FoF file products

    Returns
    -----------
    mockgal_pd_merged: pandas DataFrame
        DataFrame with the info on each mock galaxy + their group properties

    mockgroup_pd: pandas DataFrame
        DataFrame with the info on each mock galaxy group
    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Group Finding ....'.format(Prog_msg))
    # Speed of light - in km/s
    speed_c = param_dict['const_dict']['c']
    ##
    ## Running FoF
    # File prefix

    # Defining files for FoF output and Mock coordinates
    fof_file        = '{0}.galcatl_fof.{1}'.format(mock_zz_file, file_ext)
    grep_file       = '{0}.galcatl_grep.{1}'.format(mock_zz_file, file_ext)
    grep_g_file     = '{0}.galcatl_grep_g.{1}'.format(mock_zz_file, file_ext)
    mock_coord_path = '{0}.galcatl_radeccz.{1}'.format(mock_zz_file, file_ext)
    ## RA-DEC-CZ file
    mock_coord_pd = mock_pd[['ra','dec','cz']].to_csv(mock_coord_path,
                        sep=' ', header=None, index=False)
    cfutils.File_Exists(mock_coord_path)
    ## Creating `FoF` command and executing it
    fof_exe = os.path.join( cwpaths.get_code_c(), 'bin', 'fof9_ascii')
    cfutils.File_Exists(fof_exe)
    # FoF command
    fof_str = '{0} {1} {2} {3} {4} {5} {6} {7} > {8}'
    fof_arr = [ fof_exe,
                param_dict['survey_vol'],
                param_dict['zmin'],
                param_dict['zmax'],
                param_dict['l_perp'],
                param_dict['l_para'],
                param_dict['nmin'],
                mock_coord_path,
                fof_file]
    fof_cmd = fof_str.format(*fof_arr)
    # Executing command
    if param_dict['verbose']:
        print(fof_cmd)
    subprocess.call(fof_cmd, shell=True)
    ##
    ## Parsing `fof_file` - Galaxy and Group files
    gal_cmd   = 'grep G -v {0} > {1}'.format(fof_file, grep_file)
    group_cmd = 'grep G    {0} > {1}'.format(fof_file, grep_g_file)
    # Running commands
    if param_dict['verbose']:
        print(gal_cmd  )
        print(group_cmd)
    subprocess.call(gal_cmd  , shell=True)
    subprocess.call(group_cmd, shell=True)
    ##
    ## Extracting galaxy and group information
    # Column names
    gal_names   = ['groupid', 'galid', 'ra', 'dec', 'z']
    group_names = [ 'G', 'groupid', 'cen_ra', 'cen_dec', 'cen_z', 'ngals',\
                    'sigma_v', 'rproj']
    # Pandas DataFrames
    # Galaxies
    grep_pd = pd.read_csv(grep_file, sep='\s+', header=None, names=gal_names,
                index_col='galid').sort_index()
    grep_pd.index.name = None
    # Converting redshift to velocity
    grep_pd.loc[:,'cz'] = grep_pd['z'] * speed_c
    grep_pd = grep_pd.drop('z', axis=1)
    # Galaxy groups
    mockgroup_pd = pd.read_csv(grep_g_file, sep='\s+', header=None, 
                names=group_names)
    # Group centroid velocity
    mockgroup_pd.loc[:,'cen_cz'] = mockgroup_pd['cen_z'] * speed_c
    mockgroup_pd = mockgroup_pd.drop('cen_z', axis=1)
    mockgroup_pd = mockgroup_pd.drop('G', axis=1)
    ## Joining the 2 datasets for galaxies
    mockgal_pd_merged = pd.concat([mock_pd, grep_pd['groupid']], axis=1)
    # Removing `1` from `groupid`
    mockgroup_pd.loc     [:,'groupid'] -= 1
    mockgal_pd_merged.loc[:,'groupid'] -= 1
    ## Removing FoF files
    if param_dict['verbose']:
        print('{0} Removing group-finding related files'.format(
            param_dict['Prog_msg']))
    os.remove(fof_file)
    os.remove(grep_file)
    os.remove(grep_g_file)
    os.remove(mock_coord_path)
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Group Finding ....Done'.format(Prog_msg))

    return mockgal_pd_merged, mockgroup_pd

def group_mass_assignment(mockgal_pd, mockgroup_pd, param_dict, proj_dict):
    """
    Assigns a theoretical halo mass to the group based on a group property

    Parameters
    -----------
    mockgal_pd: pandas DataFrame
        DataFrame containing information for each mock galaxy.
        Includes galaxy properties + group ID

    mockgroup_pd: pandas DataFrame
        DataFame containing information for each galaxy group

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    Returns
    -----------
    mockgal_pd_new: pandas DataFrame
        Original info + abundance matched mass of the group, M_group

    mockgroup_pd_new: pandas DataFrame
        Original info of `mockgroup_pd' + abundance matched mass, M_group
    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Group Mass Assign. ....'.format(Prog_msg))
    ## Copies of DataFrames
    gal_pd   = mockgal_pd.copy()
    group_pd = mockgroup_pd.copy()
    ## Constants
    Cens     = int(1)
    Sats     = int(0)
    n_gals   = len(gal_pd  )
    n_groups = len(group_pd)
    ## Type of abundance matching
    if param_dict['catl_type'] == 'mr':
        prop_gal    = 'abs_rmag'
        reverse_opt = True
    elif param_dict['catl_type'] == 'mstar':
        prop_gal    = 'logmstar'
        reverse_opt = False
    # Absolute value of `prop_gal`
    prop_gal_abs = prop_gal + '_abs'
    ##
    ## Selecting only a `few` columns
    # Galaxies
    gal_pd = gal_pd.loc[:,[prop_gal, 'groupid']]
    # Groups
    group_pd = group_pd[['ngals']]
    ##
    ## Total `prop_gal` for groups
    group_prop_arr = [[] for x in range(n_groups)]
    ## Looping over galaxy groups
    # Mstar-based
    # if param_dict['catl_type'] == 'mstar':
    #     for group_zz in tqdm(range(n_groups)):
    #         ## Stellar mass
    #         group_prop = gal_pd.loc[gal_pd['groupid']==group, prop_gal].values
    #         group_log_prop_tot = num.log10(num.sum(10**group_prop))
    #         ## Saving to array
    #         group_prop_arr[group_zz] = group_log_prop_tot
    # Luminosity-based
    if (param_dict['catl_type'] == 'mr'):
        for group_zz in tqdm(range(n_groups)):
            ## Total abs. magnitude of the group
            group_prop = gal_pd.loc[gal_pd['groupid']==group_zz, prop_gal].values
            group_prop_tot = Mr_group_calc(group_prop)
            ## Saving to array
            group_prop_arr[group_zz] = group_prop_tot
    ##
    ## Saving to DataFrame
    group_prop_arr            = num.asarray(group_prop_arr)
    group_pd.loc[:, prop_gal] = group_prop_arr
    if param_dict['verbose']:
        print('{0} Calculating group masses...Done'.format(
            param_dict['Prog_msg']))
    ##
    ## --- Halo Abundance Matching --- ##
    ## Mass function for given cosmology
    mf_pd   = param_dict['mf_pd']
    mf_dict = dict({    'var' : mf_pd['logM'].values,
                        'dens': mf_pd['ngtm'].values})
    ## Halo mass
    Mh_ab = cm.abundance_matching.abundance_matching_f(
                group_prop_arr,
                mf_dict,
                volume1=param_dict['survey_vol'],
                dens1_opt=False,
                reverse=reverse_opt)
    # Assigning to DataFrame
    group_pd.loc[:, 'M_group'] = Mh_ab
    ###
    ### ---- Galaxies ---- ###
    # Adding `M_group` to galaxy catalogue
    gal_pd = pd.merge(gal_pd, group_pd[['M_group', 'ngals']],
                        how='left', left_on='groupid', right_index=True)
    # Remaining `ngals` column
    gal_pd = gal_pd.rename(columns={'ngals':'g_ngal'})
    #
    # Selecting `central` and `satellite` galaxies
    gal_pd.loc[:, prop_gal_abs] = num.abs(gal_pd[prop_gal])
    gal_pd.loc[:, 'g_galtype']  = num.ones(n_gals).astype(int)*Sats
    g_galtype_groups            = num.ones(n_groups)*Sats
    ##
    ## Looping over galaxy groups
    for zz in tqdm(range(n_groups)):
        gals_g = gal_pd.loc[gal_pd['groupid']==zz]
        ## Determining group galaxy type
        gals_g_max = gals_g.loc[gals_g[prop_gal_abs]==gals_g[prop_gal_abs].max()]
        g_galtype_groups[zz] = int(num.random.choice(gals_g_max.index.values))
    g_galtype_groups = num.asarray(g_galtype_groups).astype(int)
    ## Assigning group galaxy type
    gal_pd.loc[g_galtype_groups, 'g_galtype'] = Cens
    ##
    ## Dropping columns
    # Galaxies
    gal_col_arr = [prop_gal, prop_gal_abs, 'groupid']
    gal_pd      = gal_pd.drop(gal_col_arr, axis=1)
    # Groups
    group_col_arr = ['ngals']
    group_pd      = group_pd.drop(group_col_arr, axis=1)
    ##
    ## Merging to original DataFrames
    # Galaxies
    mockgal_pd_new = pd.merge(mockgal_pd, gal_pd, how='left', left_index=True,
        right_index=True)
    # Groups
    mockgroup_pd_new = pd.merge(mockgroup_pd, group_pd, how='left',
        left_index=True, right_index=True)
    if param_dict['verbose']:
        print('{0} Group Mass Assign. ....Done'.format(Prog_msg))

    return mockgal_pd_new, mockgroup_pd_new

def mock_cart_to_spherical_coords(cart_arr, dist):
    """
    Computes the right ascension and declination for the given 
    point in (x,y,z) position

    Parameters
    -----------
    cart_arr: numpy.ndarray, shape (3,)
        array with (x,y,z) positions

    dist: float
        dist to the point from observer's position

    Returns
    -----------
    ra_val: float
        right ascension of the point on the sky

    dec_val: float
        declination of the point on the sky
    """
    ## Reformatting coordinates
    # Cartesian coordinates
    (   x_val,
        y_val,
        z_val) = cart_arr/float(dist)
    # Distance to object
    dist = float(dist)
    ## Declination
    dec_val = 90. - num.degrees(num.arccos(z_val))
    ## Right ascension
    if x_val == 0:
        if y_val > 0.:
            ra_val = 90.
        elif y_val < 0.:
            ra_val = -90.
    else:
        ra_val = num.degrees(num.arctan(y_val/x_val))
    ##
    ## Seeing on which quadrant the point is at
    if x_val < 0.:
        ra_val += 180.
    elif (x_val >= 0.) and (y_val < 0.):
        ra_val += 360.

    return ra_val, dec_val

def Mr_group_calc(gal_mr_arr):
    """
    Calculated total r-band absolute magnitude of the group

    Parameters
    ----------
    gal_mr_arr: array_like
        array of r-band absolute magnitudes of member galaxies of the group

    Returns
    -------
    group_mr: float
        total r-band absolute magnitude of the group
    """
    group_lum = num.sum(10.**cmags.absolute_magnitude_to_luminosity(
                            gal_mr_arr, 'r'))
    group_mr  = cmags.luminosity_to_absolute_mag(group_lum, 'r')

    return group_mr

## ---------| Halo Rvir calculation |------------##

def halos_rvir_calc(mockgal_pd, param_dict, catl_sim_eq=False):
    """
    Calculates the virial radius of dark matter halos for each Halo in the 
    catalogue
    Taken from:
        http://home.strw.leidenuniv.nl/~franx/college/galaxies10/handout4.pdf

    Parameters:
    ------------
    mockgal_pd: pandas DataFrame
        DataFrame containing information for each mock galaxy.
        Includes galaxy properties + group ID + Ab. Match. Mass

    param_dict: python dictionary
        dictionary with `project` variables

    catl_sim_eq: boolean, optional (default = False)
        option to replace the `rvir` of all halos with zeros 
        when the number of galaxies from a distinct halo DO NOT MATCH the 
        total number of galaxies from a distinct halo,
        i.e. n_catl(halo) == n_sim(halo)

    Returns
    ------------
    mockgal_pd_new: pandas DataFrame
        Original info + Halo rvir
    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Halo Rvir Calc. ....'.format(Prog_msg))
    ## Copies of DataFrames
    gal_pd      = mockgal_pd.copy()
    ## Cosmological model parameters
    cosmo_model = param_dict['cosmo_model']
    H0          = cosmo_model.H0.to(u.km/(u.s * u.Mpc))
    Om0         = cosmo_model.Om0
    Ode0        = cosmo_model.Ode0
    ## Other constants
    G           = ac.G
    speed_c     = ac.c.to(u.km/u.s)
    ##
    ## Halo IDs
    haloid_counts = Counter(gal_pd['halo_hostid'])
    haloid_arr    = num.unique(gal_pd['halo_hostid'])
    ## Mean cz's
    haloid_z = num.array([gal_pd.loc[gal_pd['halo_hostid']==xx,'cz'].mean() for \
                        xx in haloid_arr])/speed_c.value
    ## Halo masses
    haloid_mass = num.array([gal_pd.loc[gal_pd['halo_hostid']==xx,'log_host_mvir'].mean() for \
                        xx in haloid_arr])
    ## Halo rvir - in Mpc/h
    rvir_num = (10**(haloid_mass)*u.Msun) * G
    rvir_den = 100 * H0**2 * (Om0 * (1.+haloid_z)**3 + Ode0)
    rvir_q   = ((rvir_num / rvir_den)**(1./3)).to(u.Mpc)
    rvir     = rvir_q.value
    ## Replacing with zero if necessary
    if catl_sim_eq:
        ## Replacing value
        repl_val = 0.
        ## Halo ngals - in catalogue
        haloid_ngal_cat = num.array([haloid_counts[xx] for xx in haloid_arr])
        ## Halo ngals - in simulation
        haloid_ngal_sim = num.array([gal_pd.loc[gal_pd['halo_hostid']==xx, 'halo_ngal'].values[0]\
                            for xx in haloid_arr])
        ## Chaning `rvir` values to zeros if halo is not complete
        rvir_bool = [1 if haloid_ngal_cat[xx]==haloid_ngal_sim[xx] else 0 \
                        for xx in range(len(haloid_arr))]
        rvir[rvir_bool] = repl_val
    ## Saving to DataFrame
    rvir_pd = pd.DataFrame({'halo_hostid':haloid_arr, 'halo_rvir':rvir})
    # Dropping 'rvir' column for subhalo
    gal_pd.drop('halo_rvir', axis=1, inplace=True)
    ## Merging DataFrames
    # Galaxies
    mockgal_pd_new = pd.merge(  left=gal_pd      ,
                                right=rvir_pd    ,
                                how='left'       ,
                                left_on='halo_hostid' ,
                                right_on='halo_hostid')
    if param_dict['verbose']:
        print('{0} Halo Rvir Calc. ....'.format(Prog_msg))

    return mockgal_pd_new

## ---------| Writing to Files |------------##

def writing_to_output_file(mockgal_pd, mockgroup_pd, zz_mock, hb_ii_name,
    param_dict, proj_dict, output_fmt = 'hdf5', perf_catl=False):
    """
    Writes the galaxy and group information to ascii files + astropy LaTeX
    tables

    Parameters
    -----------
    mockgal_pd: pandas DataFrame
        DataFrame containing information for each mock galaxy.
        Includes galaxy properties + group ID + Ab. Match. Mass

    mockgroup_pd: pandas DataFrame
        DataFame containing information for each galaxy group

    zz_mock: float
        number of group/galaxy catalogue being analyzed

    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    perf_catl: boolean, optional (default = False)
        if 'true', it saves the `perfect` version of the galaxy / group 
        catalogue.

    """
    ## Keys
    gal_key   = 'gal_catl'
    group_key = 'group_catl'
    ## Filenames
    if perf_catl:
        ## Perfect Galaxy catalogue
        gal_outdir = param_dict['survey_args'].catl_output_dir(hb_ii_name,
            catl_kind='memb', perf=True, create_dir=True)
        gal_file = os.path.join(gal_outdir,
                                '{0}_{1}_cat_{2}_{3}_memb_cat_perf.{4}'.format(
                                    hb_ii_name,
                                    param_dict['survey'],
                                    zz_mock, 
                                    param_dict['cosmo_choice'],
                                    output_fmt))
        ## Perfect Group catalogue
        group_outdir = param_dict['survey_args'].catl_output_dir(hb_ii_name,
            catl_kind='group', perf=True, create_dir=True)
        group_file = os.path.join(group_outdir,
                                '{0}_{1}_cat_{2}_{3}_group_cat_perf.{4}'.format(
                                    hb_ii_name,
                                    param_dict['survey'],
                                    zz_mock,
                                    param_dict['cosmo_choice'],
                                    output_fmt))
    else:
        ## Normal galaxy catalogue
        gal_outdir = param_dict['survey_args'].catl_output_dir(hb_ii_name,
            catl_kind='memb', perf=False, create_dir=True)
        gal_file = os.path.join(gal_outdir,
                                '{0}_{1}_cat_{2}_{3}_memb_cat.{4}'.format(
                                    hb_ii_name,
                                    param_dict['survey'],
                                    zz_mock, 
                                    param_dict['cosmo_choice'],
                                    output_fmt))
        ## Normal group catalogue
        group_outdir = param_dict['survey_args'].catl_output_dir(hb_ii_name,
            catl_kind='group', perf=False, create_dir=True)
        group_file = os.path.join(group_outdir,
                                '{0}_{1}_cat_{2}_{3}_group_cat.{4}'.format(
                                    hb_ii_name,
                                    param_dict['survey'],
                                    zz_mock,
                                    param_dict['cosmo_choice'],
                                    output_fmt))
    ##
    ## Saving DataFrames to files
    # Member catalogue
    cfreaders.pandas_df_to_hdf5_file(mockgal_pd, gal_file, key=gal_key)
    # Group catalogue
    cfreaders.pandas_df_to_hdf5_file(mockgroup_pd, group_file, key=group_key)
    ##
    ## Checking for file's existence
    cfutils.File_Exists(gal_file)
    cfutils.File_Exists(group_file)
    print('{0} gal_file  : {1}'.format(param_dict['Prog_msg'], gal_file))
    print('{0} group_file: {1}'.format(param_dict['Prog_msg'], group_file))

## -----------| Plotting-related functions |----------- ##

def mockcatls_simbox_plot(hb_ii_name, param_dict, proj_dict, catl_ext='.hdf5',
    fig_fmt='pdf', figsize=(9,9)):
    """
    Plots the distribution of the mock catalogues in the simulation box

    Parameters
    ------------
    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    catl_ext: string, optional (default = '.hdf5')
        file extension of the mock catalogues

    fig_fmt: string, optional (default = 'pdf')
        file format of the output figure
        Options: `pdf`, or `png`

    figsize: tuple, optional (default = (9,9))
        figure size of the output figure, in units of inches
    """
    ## Constants and variables
    Prog_msg   = param_dict['Prog_msg' ]
    plot_dict  = param_dict['plot_dict']
    markersize = plot_dict['markersize']
    ## List of catalogues
    catl_path_arr = param_dict['survey_args'].hb_gal_catl_files_list(
        hb_ii_name, catl_kind='memb', perf=False, file_ext=catl_ext)
    n_catls       = len(catl_path_arr)
    ## Filename
    fig_outdir = param_dict['survey_args'].fig_outdir(hb_ii_name,
        create_dir=True)
    fname = os.path.join(   fig_outdir,
                            '{0}_{1}_{2}_{3}_xyz_mocks.{4}'.format(
                                param_dict['survey'],
                                hb_ii_name,
                                param_dict['halotype'],
                                param_dict['cosmo_choice'],
                                fig_fmt))
    ## Setting up figure
    x_label = r'\boldmath X [Mpc $\mathrm{h^{-1}}$]'
    y_label = r'\boldmath Y [Mpc $\mathrm{h^{-1}}$]'
    z_label = r'\boldmath Z [Mpc $\mathrm{h^{-1}}$]'
    xlim    = (0, param_dict['size_cube'])
    ylim    = (0, param_dict['size_cube'])
    # Figure title
    if param_dict['survey'] == 'ECO':
        fig_title = 'ECO Survey'
    else:
        fig_title = 'RESOLVE {0}'.format(param_dict['survey'])
    # Figure and axes
    plt.close()
    plt.clf()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(221, facecolor='white', aspect='equal')
    ax2 = fig.add_subplot(222, facecolor='white', aspect='equal')
    ax3 = fig.add_subplot(223, facecolor='white', aspect='equal')
    # Limits
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    # Labels
    ax1.set_xlabel(x_label, fontsize=plot_dict['size_label'])
    ax1.set_ylabel(y_label, fontsize=plot_dict['size_label'])
    ax2.set_xlabel(x_label, fontsize=plot_dict['size_label'])
    ax2.set_ylabel(z_label, fontsize=plot_dict['size_label'])
    ax3.set_xlabel(y_label, fontsize=plot_dict['size_label'])
    ax3.set_ylabel(z_label, fontsize=plot_dict['size_label'])
    # Grid
    ax1.grid(True, color='gray', which='major', linestyle='--')
    ax2.grid(True, color='gray', which='major', linestyle='--')
    ax3.grid(True, color='gray', which='major', linestyle='--')
    # Major ticks
    major_ticks = num.arange(0,param_dict['size_cube']+1, 20)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax2.set_xticks(major_ticks)
    ax2.set_yticks(major_ticks)
    ax3.set_xticks(major_ticks)
    ax3.set_yticks(major_ticks)
    # Colormap
    cm      = plt.get_cmap('gist_rainbow')
    col_arr = [cm(ii/float(n_catls)) for ii in range(n_catls)]
    # Title
    title_obj = fig.suptitle(fig_title, fontsize=plot_dict['title'])
    title_obj.set_y(1.04)
    ##
    ## Looping over different catalogues
    for kk, catl_kk in enumerate(tqdm(catl_path_arr)):
        # Reading parameters
        catl_kk_pd = cfreaders.read_hdf5_file_to_pandas_DF(catl_kk)
        # Color
        color_kk = col_arr[kk]
        # Galaxy indices
        (   x_kk_arr,
            y_kk_arr,
            z_kk_arr) = catl_kk_pd[['x_orig','y_orig','z_orig']].values.T
        ## Plotting points (galaxies)
        ax1.plot(x_kk_arr, y_kk_arr, marker='o', color=color_kk,
            markersize=markersize, linestyle='None', rasterized=True)
        ax2.plot(x_kk_arr, z_kk_arr, marker='o', color=color_kk,
            markersize=markersize, linestyle='None', rasterized=True)
        ax3.plot(y_kk_arr, z_kk_arr, marker='o', color=color_kk,
            markersize=markersize, linestyle='None', rasterized=True)
    # Adjusting space
    plt.subplots_adjust(top=0.86)
    plt.tight_layout()
    # Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()

def mocks_lum_function(hb_ii_name, param_dict, proj_dict, catl_ext='.hdf5',
    fig_fmt='pdf', figsize=(9,9)):
    """
    Computes the luminosity function of the mock catalogues

    Parameters
    ------------
    hb_ii_name : `str`
        Name of key corresponding to the Halobias file being analyzed.

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    catl_ext: string, optional (default = '.hdf5')
        file extension of the mock catalogues

    fig_fmt: string, optional (default = 'pdf')
        file format of the output figure
        Options: `pdf`, or `png`

    figsize: tuple, optional (default = (9,9))
        figure size of the output figure, in units of inches
    """
    matplotlib.rcParams['axes.linewidth'] = 2.5
    ## Constants and variables
    Prog_msg    = param_dict['Prog_msg' ]
    plot_dict   = param_dict['plot_dict']
    markersize  = plot_dict['markersize']
    ## Separation for the `M_r` bins, in units of magnitudes
    mr_bins_sep = 0.2
    ## List of catalogues
    catl_path_arr = param_dict['survey_args'].hb_gal_catl_files_list(
        hb_ii_name, catl_kind='memb', perf=False, file_ext=catl_ext)
    n_catls       = len(catl_path_arr)
    ## Filename
    fig_outdir = param_dict['survey_args'].fig_outdir(hb_ii_name,
        create_dir=True)
    fname = os.path.join(   fig_outdir,
                            '{0}_{1}_{2}_{3}_lum_function_mocks.{4}'.format(
                                param_dict['survey'],
                                hb_ii_name,
                                param_dict['halotype'],
                                param_dict['cosmo_choice'],
                                fig_fmt))
    # Colormap
    cm      = plt.get_cmap('gist_rainbow')
    col_arr = [cm(ii/float(n_catls)) for ii in range(n_catls)]
    ## Setting up figure
    x_label = r'\boldmath $M_{r}$'
    y_label = r'\boldmath $n(< M_{r}) \left[h^{3}\ \textrm{Mpc}^{-3}\right]$'
    # Figure
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    # Labels
    ax1.set_xlabel(x_label, fontsize=plot_dict['size_label'])
    ax1.set_ylabel(y_label, fontsize=plot_dict['size_label'])
    ## Looping over mock catalogues
    ## Looping over different catalogues
    for kk, catl_kk in enumerate(tqdm(catl_path_arr)):
        # Reading parameters
        catl_kk_pd = cfreaders.read_hdf5_file_to_pandas_DF(catl_kk)
        # Color
        color_kk = col_arr[kk]
        ## Calculating luminosity function
        mr_bins = cstats.Bins_array_create(catl_kk_pd['abs_rmag'], base=mr_bins_sep)
        N_lum   = [num.where(catl_kk_pd['abs_rmag'] < xx)[0].size+1 for xx in mr_bins]
        n_lum   = num.asarray(N_lum)/param_dict['survey_vol']
        ## Plotting
        ax1.plot(mr_bins, n_lum, color=color_kk, marker='o', linestyle='-',
            markersize=markersize)
    # Log-axis
    ax1.set_yscale('log')
    # Reverse axis
    ax1.invert_xaxis()
    # Adjusting space
    plt.subplots_adjust(top=0.86)
    plt.tight_layout()
    # Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()



## -----------| Survey-related functions |----------- ##

def survey_specs(param_dict):
    """
    Provides the specifications of the survey being created

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    ----------
    param_dict: python dictionary
        dictionary with the 'updated' project variables
    """
    ## Cosmological model
    cosmo_model = param_dict['survey_args'].cosmo_create()
    ## Redshift, volumen and r-mag limit for each survey
    if param_dict['survey'] == 'A':
        czmin      = 2532.
        czmax      = 7470.
        # survey_vol = 20957.7789388
        mr_limit   = -17.33
    elif param_dict['survey'] == 'B':
        czmin      = 4250.
        czmax      = 7250.
        # survey_vol = 15908.063125
        mr_limit   = -17.00
    elif param_dict['survey'] == 'ECO':
        czmin      = 2532.
        czmax      = 7470.
        # survey_vol = 192294.221932
        mr_limit   = -17.33
    ##
    ## Right Ascension and Declination coordinates for each survey
    if param_dict['survey'] == 'A':
        ra_min_real = 131.25
        ra_max_real = 236.25
        dec_min     = 0.
        dec_max     = 5.
        # Extras
        dec_range   = dec_max - dec_min
        ra_range    = ra_max_real - ra_min_real
        ra_min      = (180. - ra_range)/2.
        ra_max      = ra_min + ra_range
        ra_diff     = ra_max_real - ra_max
        # Assert statements
        assert(dec_min < dec_max)
        assert(ra_range >= 0)
        assert(ra_min < ra_max)
        assert(ra_min_real < ra_max_real)
    elif param_dict['survey'] == 'B':
        ra_min_real = 330.
        ra_max_real = 45.
        dec_min     = -1.25
        dec_max     = 1.25
        # Extras
        dec_range   = dec_max - dec_min
        ra_range    = ra_max_real - (ra_min_real - 360.)
        ra_min      = (180. - ra_range)/2.
        ra_max      = ra_min + ra_range
        ra_diff     = ra_max_real - ra_max
        # Assert statements
        assert(dec_min < dec_max)
        assert(ra_range >= 0)
        assert(ra_min < ra_max)
    elif param_dict['survey'] == 'ECO':
        ra_min_real = 130.05
        ra_max_real = 237.45
        dec_min     = -1
        dec_max     = 49.85
        # Extras
        dec_range   = dec_max - dec_min
        ra_range    = ra_max_real - ra_min_real
        ra_min      = (180. - ra_range)/2.
        ra_max      = ra_min + ra_range
        ra_diff     = ra_max_real - ra_max
        # Assert statements
        assert(dec_min < dec_max)
        assert(ra_range >= 0)
        assert(ra_min < ra_max)
        assert(ra_min_real < ra_max_real)
    ## Survey volume
    km_s       = u.km/u.s
    z_arr      = (num.array([czmin, czmax])*km_s/(ac.c.to(km_s))).value
    z_arr      = (num.array([czmin, czmax])*km_s/(3e5*km_s)).value
    r_arr      = cosmo_model.comoving_distance(z_arr).to(u.Mpc).value
    survey_vol = param_dict['survey_args'].survey_vol_calc( [0, ra_range],
                                                            [dec_min, dec_max],
                                                            r_arr)
    ##
    ## Survey height, and other geometrical factors
    (   h_total,
        h1     ,
        s1_top ,
        s2     ) = param_dict['survey_args'].geometry_calc( r_arr[0],
                                                            r_arr[1],
                                                            ra_range)
    (   h_side ,
        h2     ,
        s1_side,
        d_th   ) = param_dict['survey_args'].geometry_calc( r_arr[0],
                                                            r_arr[1],
                                                            dec_range)
    ##
    # ra_dec dictionary
    coord_dict = {}
    coord_dict['ra_min_real'] = ra_min_real
    coord_dict['ra_max_real'] = ra_max_real
    coord_dict['dec_min'    ] = dec_min
    coord_dict['dec_max'    ] = dec_max
    coord_dict['dec_range'  ] = dec_range
    coord_dict['ra_range'   ] = ra_range
    coord_dict['ra_min'     ] = ra_min
    coord_dict['ra_max'     ] = ra_max
    coord_dict['ra_diff'    ] = ra_diff
    # Height and other geometrical objects
    coord_dict['h_total'    ] = h_total
    coord_dict['s1_top'     ] = s1_top
    coord_dict['s2'         ] = s2
    coord_dict['h1'         ] = h1
    coord_dict['h_side'     ] = h_side
    coord_dict['s1_side'    ] = s1_side
    coord_dict['d_th'       ] = d_th
    coord_dict['h2'         ] = h2
    coord_dict['r_arr'      ] = r_arr
    ##
    ## Resolve-B Mr limit
    mr_eco   = -17.33
    mr_res_b = -17.00
    ## Saving to `param_dict`
    param_dict['czmin'     ] = czmin
    param_dict['czmax'     ] = czmax
    param_dict['zmin'      ] = z_arr[0]
    param_dict['zmax'      ] = z_arr[1]
    param_dict['survey_vol'] = survey_vol
    param_dict['mr_limit'  ] = mr_limit
    param_dict['mr_eco'    ] = mr_eco
    param_dict['mr_res_b'  ] = mr_res_b
    param_dict['coord_dict'] = coord_dict

    return param_dict




## --------- Multiprocessing ------------##

def multiprocessing_catls(hb_keys, param_dict, proj_dict, memb_tuples_ii):
    """
    Distributes the analysis of the catalogues into more than 1 processor

    Parameters:
    -----------
    hb_keys : `numpy.ndarray`
        List of Halobias filenames keys.        

    param_dict : `dict`
        Dictionary with the `project` variables.

    proj_dict : `dict`
        Dictionary with current and new paths to project directories

    memb_tuples_ii : `tuple`
        Tuple of halobias file indices to be analyzed
    """
    ## Program Message
    Prog_msg = param_dict['Prog_msg']
    ## Reading in Catalogue IDs
    start_ii, end_ii = memb_tuples_ii
    ## Index value
    idx_arr  = num.array(range(start_ii, end_ii), dtype=int)
    ## Catalogue array
    hb_keys_ii = hb_keys[start_ii : end_ii]
    ##
    ## Looping the desired catalogues
    for (ii, hb_key_ii) in zip(idx_arr, hb_keys_ii):
        ## Converting index to main `int`
        ii = int(ii)
        ## Choosing 1st catalogue
        if param_dict['verbose']:
            print('{0} Analyzing `{1}`\n'.format(Prog_msg, hb_key_ii))
        ## Extracting `name` of the catalogue
        hb_ii_name = os.path.splitext(os.path.split(hb_key_ii)[1])[0]
        ## Analaysis for the Halobias file
        hb_analysis(ii, hb_ii_name, param_dict, proj_dict)

## --------- Main Function ------------##

def main(args):
    """
    Main function to create CAM mock group galaxy catalogues.
    """
    ## Starting time
    start_time = datetime.now()
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    #
    # Creating instance of `ReadML` with the input parameters
    param_dict['survey_args'] = ReadSurvey(**param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    # Adding additional parameters
    param_dict = add_to_dict(param_dict)
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths(__file__))
    proj_dict = param_dict['survey_args'].proj_dict
    proj_dict  = directory_skeleton(param_dict, proj_dict)
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        key_arr = ['Prog_msg', 'hb_files_dict', 'z_dc_pd', 'mf_pd']
        if key not in key_arr:
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ### ---- Analyzing Catalogues ---- ###
    ##
    # Number of Halobias Files
    n_hb_files = param_dict['n_hb_files']
    # Halobias Keys
    hb_keys = num.sort(list(param_dict['hb_files_dict'].keys()))
    ## Using `multiprocessing` to analyze merged catalogues files
    ## Number of CPU's to use
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ## Defining step-size for each CPU
    if cpu_number <= n_hb_files:
        catl_step = int(n_hb_files / cpu_number)
        memb_arr = num.arange(0, n_hb_files+1, catl_step)
    else:
        catl_step = int((n_hb_files / cpu_number)**-1)
        memb_arr = num.arange(0, n_hb_files+1)
    ## Array with designated catalogue numbers for each CPU
    memb_arr[-1] = n_hb_files
    ## Tuples of the ID of each catalogue
    memb_tuples  = num.asarray([(memb_arr[xx], memb_arr[xx+1])
                            for xx in range(memb_arr.size-1)])
    ## Assigning `memb_tuples` to function `multiprocessing_catls`
    procs = []
    for ii in range(len(memb_tuples)):
        # Defining `proc` element
        proc = Process(target=multiprocessing_catls, 
                        args=(hb_keys, param_dict, 
                            proj_dict, memb_tuples[ii]))
        # Appending to main `procs` list
        procs.append(proc)
        proc.start()
    ##
    ## Joining `procs`
    for proc in procs:
        proc.join()
    ## End time for running the catalogues
    end_time   = datetime.now()
    total_time = end_time - start_time
    print('{0} Total Time taken (Create): {1}'.format(Prog_msg, total_time))

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
