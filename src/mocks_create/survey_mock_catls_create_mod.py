#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : DATE
# Last Modified: DATE
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
                        choices=['mr', 'mstar'],
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
    # FoF linking lengths
    l_perp = 0.07
    l_para = 1.1
    # Dictionary of Halobias files
    hb_files_dict = param_dict['survey_args'].halobias_files_dict()
    n_hb_files    = len(hb_files_dict.keys())
    # Cosmological model and Halo Mass function
    cosmo_model = param_dict['survey_args'].cosmo_create()
    # Redshift and Comoving Distances
    z_dc_pd = param_dict['survey_args'].comoving_z_distance()
    # Mass Function
    mf_pd = param_dict['survey_args'].hmf_calc()
    ##
    ## Saving to dictionary
    param_dict['cens'          ] = cens
    param_dict['sats'          ] = sats
    param_dict['url_catl'      ] = url_catl
    param_dict['url_mock_cubes'] = url_mock_cubes
    param_dict['plot_dict'     ] = plot_dict
    param_dict['const_dict'    ] = const_dict
    param_dict['l_perp'        ] = l_perp
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
    ## Central/Satellite Designation
    hb_ii_pd = cen_sat_designation(hb_ii_pd)


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
    hb_ii_pd = hb_extras(hb_ii_pd)
    # Distance between central and satellites
    hb_ii_pd = cen_sat_dist_calc(hb_ii_pd, param_dict)

    return hb_ii_pd

## Extra Halo properties
def hb_extras(hb_ii_pd):
    """
    Determines the Central/Satellite designation for each galaxy in the
    halobias file.

    Parameters
    ------------
    hb_ii_pd : `pandas.DataFrame`
        DataFrame containing main info from Halobias being analyzed.

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
    for ii, halo_ii in enumerate(tqdm(haloid_unq)):
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
            sats_coords.substract(cens_coords_mean, axis=1).values**2, axis=1)
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



# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
