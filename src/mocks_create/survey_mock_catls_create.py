#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 02/20/2018
# Last Modified: 02/20/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Creates the catalogues for (1) ECO, (2) RESOLVE-A, and (3) RESOLVE-B surveys.
"""
# Path to Custom Utilities folder
import os
import sys
import git

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

# Importing Modules
import numpy as num
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
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

# Source module
from src.survey_utils import ReadSurvey


## Functions

## ---------| Reading input arguments and other main functions |--------- ##

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
    val: int or float
        value to be evaluated by `val_min`

    val_min: float or int, optional (default = 0)
        minimum value that `val` can be

    Returns
    -------
    ival: float
        value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError: Raised if `val` is NOT larger than `val_min`
    """
    ival = float(val)
    if ival <= val_min:
        msg  = '`{0}` is an invalid input!'.format(ival)
        msg += '`val` must be larger than `{0}`!!'.format(val_min)
        raise argparse.ArgumentTypeError(msg)

    return ival

def is_valid_file(parser, arg):
    """
    Check if arg is a valid file that already exists on the file system.

    Parameters
    ----------
    parser : argparse object
    arg : str

    Returns
    -------
    arg
    """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!\n" % arg)
    else:
        return arg

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
    ##  Version
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
    ## CLF Choice
    parser.add_argument('-clf',
                        dest='clf_type',
                        help='Type of CLF to choose. 1) Cacciato, 2) LasDamas Best-fit',
                        type=int,
                        choices=[1,2],
                        default=2)
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
                        default=True)
    ## Parsing Objects
    args = parser.parse_args()

    return args

def param_vals_test(param_dict):
    """
    Checks if values are consistent with each other.

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with `project` variables

    Raises
    -----------
    ValueError: Error
        This function raises a `ValueError` error if one or more of the 
        required criteria are not met
    """
    ##
    ## This is where the tests for `param_dict` input parameters go.
    ##
    ## Size of the cube
    assert(param_dict['size_cube'] == 130.)

def add_to_dict(param_dict):
    """
    Aggregates extra variables to dictionary

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with input parameters and values

    Returns
    ----------
    param_dict: python dictionary
        dictionary with old and new values added
    """
    ## Central/Satellite designations
    cens = int(1)
    sats = int(0)
    ## HOD Parameters
    hod_dict             = {}
    hod_dict['logMmin' ] = 11.4
    hod_dict['sigLogM' ] = 0.2
    hod_dict['logM0'   ] = 10.8
    hod_dict['logM1'   ] = 12.8
    hod_dict['alpha'   ] = 1.05
    hod_dict['zmed_val'] = 'z0p000'
    hod_dict['znow'    ] = 0
    ## Choice of Survey
    choice_survey = 2
    ###
    ### URL to download catalogues
    url_catl = 'http://lss.phy.vanderbilt.edu/groups/data_eco_vc/'
    url_cam_catl = 'http://lss.phy.vanderbilt.edu/groups/data_eco_vc/ECO_CAM/ECO/'
    cweb.url_checker(url_catl)
    cweb.url_checker(url_cam_catl)
    ##
    ## Plotting constants
    plot_dict = plot_const()
    ##
    ## Variable constants
    const_dict = val_consts()
    # FoF linking lengths
    l_perp = 0.07
    l_para = 1.1
    ## Survey name
    if param_dict['survey'] == 'ECO':
        survey_name = 'ECO'
    else:
        survey_name = 'RESOLVE_{0}'.format(param_dict['survey'])
    ## README url
    readme_url = 'https://goo.gl/Xo317R'
    ##
    ## Adding to `param_dict`
    param_dict['cens'         ] = cens
    param_dict['sats'         ] = sats
    param_dict['url_catl'     ] = url_catl
    param_dict['url_cam_catl' ] = url_cam_catl
    param_dict['hod_dict'     ] = hod_dict
    param_dict['choice_survey'] = choice_survey
    param_dict['plot_dict'    ] = plot_dict
    param_dict['const_dict'   ] = const_dict
    param_dict['l_perp'       ] = l_perp
    param_dict['l_para'       ] = l_para
    param_dict['survey_name'  ] = survey_name
    param_dict['readme_url'   ] = readme_url

    return param_dict

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    # Raw directory
    raw_dir      = os.path.join(proj_dict['data_dir'],
                                'raw')
    # Halobias Files
    hb_files_dir = os.path.join(raw_dir,
                                'hb_files',
                                param_dict['halotype'],
                                param_dict['survey'])
    # Cosmological files
    cosmo_dir = os.path.join(   raw_dir,
                                param_dict['cosmo_choice'],
                                'cosmo_files')
    # Interim directory
    int_dir      = os.path.join( proj_dict['data_dir'],
                                'interim')
    # Mass Function
    mf_dir       = os.path.join(int_dir,
                                'MF',
                                param_dict['cosmo_choice'],
                                param_dict['survey'])
    # Conditional Luminosity Function (CLF)
    clf_dir      = os.path.join(int_dir,
                                'CLF_HB',
                                param_dict['cosmo_choice'],
                                param_dict['halotype'],
                                param_dict['survey'] + '/')
    # Halo Ngal
    h_ngal_dir   = os.path.join(int_dir,
                                'HALO_NGAL_CLF',
                                param_dict['cosmo_choice'],
                                param_dict['halotype'],
                                param_dict['survey'] + '/')
    ## Catalogues
    catl_outdir = os.path.join( proj_dict['data_dir'],
                                'processed',
                                param_dict['cosmo_choice'],
                                param_dict['halotype'],
                                param_dict['survey'])
    ## Photometry files
    phot_dir    = os.path.join( raw_dir,
                                'surveys_phot_files')
    ## Figures
    fig_dir     = os.path.join( proj_dict['plot_dir'],
                                param_dict['cosmo_choice'],
                                param_dict['halotype'],
                                param_dict['survey'])
    ## TAR folder
    tar_dir     = os.path.join( proj_dict['data_dir'],
                                'processed',
                                'TAR_files',
                                param_dict['cosmo_choice'],
                                param_dict['halotype'],
                                param_dict['survey'])
    ## Creating output folders for the catalogues
    mock_cat_mgc     = os.path.join(catl_outdir, 'galaxy_catalogues')
    mock_cat_mc      = os.path.join(catl_outdir, 'member_galaxy_catalogues')
    mock_cat_gc      = os.path.join(catl_outdir, 'group_galaxy_catalogues' )
    mock_cat_mc_perf = os.path.join(catl_outdir, 'perfect_member_galaxy_catalogues')
    mock_cat_gc_perf = os.path.join(catl_outdir, 'perfect_group_galaxy_catalogues' )
    ##
    ## Creating Directories
    cfutils.Path_Folder(cosmo_dir)
    cfutils.Path_Folder(catl_outdir)
    cfutils.Path_Folder(mock_cat_mgc)
    cfutils.Path_Folder(mock_cat_mc)
    cfutils.Path_Folder(mock_cat_gc)
    cfutils.Path_Folder(mock_cat_mc_perf)
    cfutils.Path_Folder(mock_cat_gc_perf)
    cfutils.Path_Folder(int_dir)
    cfutils.Path_Folder(mf_dir)
    cfutils.Path_Folder(clf_dir)
    cfutils.Path_Folder(h_ngal_dir)
    cfutils.Path_Folder(raw_dir)
    cfutils.Path_Folder(hb_files_dir)
    cfutils.Path_Folder(phot_dir)
    cfutils.Path_Folder(fig_dir)
    cfutils.Path_Folder(tar_dir)
    ##
    ## Adding to `proj_dict`
    proj_dict['cosmo_dir'       ] = cosmo_dir
    proj_dict['catl_outdir'     ] = catl_outdir
    proj_dict['mock_cat_mgc'    ] = mock_cat_mgc
    proj_dict['mock_cat_mc'     ] = mock_cat_mc
    proj_dict['mock_cat_gc'     ] = mock_cat_gc
    proj_dict['mock_cat_mc_perf'] = mock_cat_mc_perf
    proj_dict['mock_cat_gc_perf'] = mock_cat_gc_perf
    proj_dict['int_dir'         ] = int_dir
    proj_dict['mf_dir'          ] = mf_dir
    proj_dict['clf_dir'         ] = clf_dir
    proj_dict['h_ngal_dir'      ] = h_ngal_dir
    proj_dict['raw_dir'         ] = raw_dir
    proj_dict['hb_files_dir'    ] = hb_files_dir
    proj_dict['phot_dir'        ] = phot_dir
    proj_dict['fig_dir'         ] = fig_dir
    proj_dict['tar_dir'         ] = tar_dir

    return proj_dict

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

## -----------| Tools and more-related functions |----------- ##

def closest_val(val, arr):
    """
    Finds the closest value in `arr` to `val`

    Parameters
    -------------
    val: int or float
        value to be looked at

    arr: numpy.ndarray
        numpy array used for finding closest value to `val`

    Returns
    -------------
    idx_choice: int
        index for the `best matched` index to `val`
    """
    val = float(val)
    arr = num.asarray(arr)
    idx = (num.abs(arr - val)).argmin()
    idx_arr = num.where(arr == arr[idx])[0]
    try:
        if len(idx_arr) == 1:
            return idx
        else:
            return num.random.choice(idx_arr)
    except:
        raise ValueError('>> Not matches found!')

def survey_vol_calc(ra_arr, dec_arr, rho_arr):
    """
    Computes the volume of a "sphere" with given limits for 
    ra, dec, and distance

    Parameters
    ----------
    ra_arr: list or numpy.ndarray, shape (N,2)
        array with initial and final right ascension coordinates
        Unit: degrees

    dec_arr: list or numpy.ndarray, shape (N,2)
        array with initial and final declination coordinates
        Unit: degrees

    rho_arr: list or numpy.ndarray, shape (N,2)
        arrray with initial and final distance
        Unit: distance units
    
    Returns
    ----------
    survey_vol: float
        volume of the survey being analyzed
        Unit: distance**(3)
    """
    # Right ascension - Radians  theta coordinate
    theta_min_rad, theta_max_rad = num.radians(num.array(ra_arr))
    # Declination - Radians - phi coordinate
    phi_min_rad, phi_max_rad = num.radians(90.-num.array(dec_arr))[::-1]
    # Distance
    rho_min, rho_max = rho_arr
    # Calculating volume
    vol  = (1./3.)*(num.cos(phi_min_rad)-num.cos(phi_max_rad))
    vol *= (theta_max_rad) - (theta_min_rad)
    vol *= (rho_max**3) - rho_min**3
    vol  = num.abs(vol)

    return vol

def cos_rule(a, b, gamma):
    """
    Computes the `cosine rule` for 2 distances and one angle

    Parameters
    -----------
    a: float
        one of the sides of the triangle

    b: float
        the other side of the triangle

    gamma: float
        angle facing the side of the triangle in questions
        Units: degrees

    Returns
    -----------
    c: float
        measure of the size `c` in question
    """
    # Degree in radians
    gamma_rad = num.radians(gamma)
    # Third side of the triangle
    c = (a**2 + b**2 - (2*a*b*num.cos(gamma_rad)))**(.5)

    return c

def distance_diff_catl(ra, dist, gap):
    """
    Computes the necessary distance between catalogues

    Parameters
    -----------
    ra: float
        1st distance

    dist: float
        2nd distance

    Returns
    -----------
    dist_diff: float
        amount of distance necessary between mocks
    """
    ra = float(ra)
    dist = float(dist)
    gap = float(gap)
    ## Calculation of distance between catalogues
    dist_diff = (((ra + gap)**2 - (dist/2.)**2.)**(0.5)) - ra

    return dist_diff

def geometry_calc(dist_1, dist_2, alpha):
    """
    Computes the geometrical components to construct the catalogues in 
    a simulation box

    Parameters
    -----------
    dist_1: float
        1st distance used to determine geometrical components

    dist_2: float
        2nd distance used to determine geometrical components

    alpha: float
        angle used to determine geometrical components
        Unit: degrees

    Returns
    -----------
    h_total: float

    h1: float

    s1: float

    s2: float
    """
    assert(dist_1 <= dist_2)
    ## Calculating distances for the triangles
    s1 = cos_rule(dist_1, dist_1, alpha)
    s2 = cos_rule(dist_2, dist_2, alpha)
    ## Height
    h1 = (dist_1**2 - (s1/2.)**2)**0.5
    assert(h1 <= dist_1)
    h2      = dist_1 - h1
    h_total = h2 + (dist_2 - dist_1)

    return h_total, h1, s1, s2

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

def cosmo_create(param_dict, H0=100., Om0=0.25, Ob0=0.04, Tcmb0=2.7255,
    return_params=False, return_hmf=False):
    """
    Creates instance of the cosmology used throughout the project.

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables and `cosmo_choice`

    H0 : `float`, optional (default = 100.)
        This variable is the Hubble constant, in units of km/s. This variable
        is set to ``100 km/s`` by default.

    Om0 : `float`, optional (default = 0.25)
        - value of `Omega Matter at z=0`
        - Unitless
        - Range [0,1]

    Ob0 : `float`, optional (default = 0.04)
        - value of `Omega Baryon at z=0`
        - Unitless

    Tcmb0: float, optional (default = 2.7255)
        temperature of the CMB at z=0

    return_cosmo_params : `bool`, optional
        If True, it returns a list of the cosmological parameters used.
        This variable is set to ``False`` by default.

    return_hmf : `bool`, optional
        If True, it returns the halo mass function DataFrame. This
        variale is set to ``False`` by default.

    Returns
    ----------
    param_dict: python dictionary
        updated version of `param_dict` with new variables:
            - `cosmo_params`: dictionary with cosmological parameters
            - `cosmo_model`: astropy cosmology object
                    cosmology used throughout the project
            - `cosmo_hmf`: `hmf.cosmo.Cosmology` object
    """
    ## Checking cosmology choices
    cosmo_choice_arr = ['Planck', 'LasDamas']
    assert(param_dict['cosmo_choice'] in cosmo_choice_arr)
    ## Choosing cosmology
    if param_dict['cosmo_choice'] == 'Planck':
        cosmo_model = astrocosmo.Planck15.clone(H0=H0)
    elif param_dict['cosmo_choice'] == 'LasDamas':
        cosmo_model = astrocosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, 
            Tcmb0=Tcmb0)
    ## Cosmo Paramters
    cosmo_params         = {}
    cosmo_params['H0'  ] = cosmo_model.H0.value
    cosmo_params['Om0' ] = cosmo_model.Om0
    cosmo_params['Ob0' ] = cosmo_model.Ob0
    cosmo_params['Ode0'] = cosmo_model.Ode0
    cosmo_params['Ok0' ] = cosmo_model.Ok0
    ## HMF Cosmo Model
    cosmo_hmf = hmf.cosmo.Cosmology(cosmo_model=cosmo_model)
    ## Saving to 'param_dict'
    param_dict['cosmo_model' ] = cosmo_model
    param_dict['cosmo_params'] = cosmo_params
    param_dict['cosmo_hmf'   ] = cosmo_hmf

    return param_dict

def hmf_calc(cosmo_model, proj_dict, param_dict, Mmin=10, Mmax=16, 
    dlog10m=1e-3, hmf_model='warren', ext='csv', sep=',', 
    Prog_msg='1 >>   '):
    # Prog_msg=cu.Program_Msg(__file__)):
    """
    Creates file with the desired mass function

    Parameters
    ----------
    cosmo_model: astropy cosmology object
        cosmology used throughout the project

    hmf_out: string
        path to the output file for the halo mass function

    Mmin: float, optional (default = 10)
        minimum halo mass to evaluate

    Mmax: float, optional (default = 15)
        maximum halo mass to evaluate

    dlog10m: float, optional (default = 1e-2)


    hmf_model: string, optional (default = 'warren')
        Halo Mass Function choice
        Options:
            - 'warren': Uses Warren et al. (2006) HMF
            = 'tinker08': Uses Tinker et al. (2008) HMF

    ext: string, optional (default = 'csv')
        extension of output file

    sep: string, optional (default = ',')
        delimiter used for reading/writing the file

    Returns
    ----------
    param_dict: python dictionary
        dictionary with udpdated variables:
            - 'hmf_pd': pandas DataFrame
                    DataFrame of `log10 masses` and `cumulative number 
                    densities` for halos of mass > M.
    """
    ## HMF Output file
    hmf_outfile = os.path.join( proj_dict['mf_dir'],
                                '{0}_H0_{1}_HMF_{2}.{3}'.format(
                                    param_dict['cosmo_choice'],
                                    cosmo_model.H0.value,
                                    hmf_model,
                                    ext))
    if os.path.exists(hmf_outfile):
        # Removing file
        os.remove(hmf_outfile)
    ## Halo mass function - Fitting function
    if hmf_model == 'warren':
        hmf_choice_fit = hmf.fitting_functions.Warren
    elif hmf_model == 'tinker08':
        hmf_choice_fit = hmf.fitting_functions.Tinker08
    else:
        msg = '{0} hmf_model `{1}` not supported! Exiting'.format(
            Prog_msg, hmf_model)
        raise ValueError(msg)
    # Calculating HMF
    mass_func = hmf.MassFunction(Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m,
        cosmo_model=cosmo_model, hmf_model=hmf_choice_fit)
    ## Log10(Mass) and cumulative number density of haloes
    # HMF Pandas DataFrame
    hmf_pd = pd.DataFrame({ 'logM':num.log10(mass_func.m), 
                            'ngtm':mass_func.ngtm})
    # Saving to output file
    hmf_pd.to_csv(hmf_outfile, sep=sep, index=False,
        columns=['logM','ngtm'])
    # Saving to `param_dict`
    param_dict['hmf_pd'] = hmf_pd

    return param_dict

def download_files(param_dict, proj_dict):
    """
    Downloads the required files to a specific set of directories
    
    Parameters
    ------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ------------
    param_dict: python dictionary
        dictionary with the updated paths to the local files that were 
        just downloaded
    """
    ## Main ECO files directory - Web
    resolve_web_dir = os.path.join( param_dict['url_catl'],
                                    'RESOLVE',
                                    'resolve_files')
    eco_web_dir     = os.path.join( param_dict['url_catl'],
                                    'ECO',
                                    'eco_files')
    ## ECO - MHI Predictions
    mhi_file_local = os.path.join(  proj_dict['phot_dir'],
                                    'eco_mhi_prediction.txt')
    mhi_file_web   = os.path.join(  resolve_web_dir,
                                    'eco_wresa_050815_Predicted_HI.txt')
    ## ECO - Photometry file
    eco_phot_file_local = os.path.join( proj_dict['phot_dir'],
                                        'eco_wresa_050815.dat')
    eco_phot_file_web   = os.path.join( resolve_web_dir,
                                        'eco_wresa_050815.dat')
    ## Resolve-B Photometry
    res_b_phot_file_local = os.path.join(proj_dict['phot_dir'],
                                        'resolvecatalog_str_2015_07_12.fits')
    res_b_phot_file_web   = os.path.join(resolve_web_dir,
                                        'resolvecatalog_str_2015_07_12.fits')
    ## ECO-Resolve B Luminosity Function
    eco_LF_file_local = os.path.join(   proj_dict['phot_dir'],
                                        'eco_resolve_LF.csv')
    eco_LF_file_web   = os.path.join(   resolve_web_dir,
                                        'ECO_ResolveB_Lum_Function.csv')
    ## ECO Luminosities
    eco_lum_file_local = os.path.join(  proj_dict['phot_dir'],
                                        'eco_dens_mag.csv')
    eco_lum_file_web   = os.path.join(  eco_web_dir,
                                        'Dens_Mag_Interp_ECO.ascii')
    ## Halobias file
    hb_file_local      = os.path.join(  proj_dict['hb_files_dir'],
                                        'Resolve_plk_5001_so_{0}_hod1.ff'.format(
                                            param_dict['halotype']))
    ##
    ## Downloading files
    files_local_arr = [ mhi_file_local       , eco_phot_file_local,
                        res_b_phot_file_local, eco_LF_file_local  , 
                        eco_lum_file_local         ]
    files_web_arr   = [ mhi_file_web         , eco_phot_file_web  ,
                        res_b_phot_file_web  , eco_LF_file_web    ,
                        eco_lum_file_web             ]
    ## Checking that files exist
    for (local_ii, web_ii) in zip(files_local_arr,files_web_arr):
        ## Downloading file if not in `local`
        if not os.path.exists(local_ii):
            ## Checking for `web` file
            cweb.url_checker(web_ii)
            ## Downloading
            cfutils.File_Download_needed(local_ii, web_ii)
            assert(os.path.exists(local_ii))
    ##
    ## Saving paths to `param_dict'
    files_dict = {}
    files_dict['mhi_file_local'       ] = mhi_file_local
    files_dict['eco_phot_file_local'  ] = eco_phot_file_local
    files_dict['res_b_phot_file_local'] = res_b_phot_file_local
    files_dict['eco_LF_file_local'    ] = eco_LF_file_local
    files_dict['eco_lum_file_local'   ] = eco_lum_file_local
    files_dict['hb_file_local'        ] = hb_file_local
    # To `param_dict'
    param_dict['files_dict'] = files_dict
    ##
    ## Showing stored files
    print('{0} Downloaded all necessary files!'.format(param_dict['Prog_msg']))

    return param_dict

def z_comoving_calc(param_dict, proj_dict, 
    zmin=0, zmax=0.5, dz=1e-3, ext='csv', sep=','):
    """
    Computes the comoving distance of an object based on its redshift
    
    Parameters
    ------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ------------
    param_dict: python dictionary
        updated dictionary with `project` variables + `z_como_pd`, which 
        is the DataFrame with `z, d_comoving` in units of Mpc
    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Comoving Distance Table Calc ....'.format(Prog_msg))
    ## Cosmological model
    cosmo_model = param_dict['cosmo_model']
    ## File
    z_comoving_file = os.path.join( proj_dict['cosmo_dir'],
                                    '{0}_H0_{1}_z_comoving.{2}'.format(
                                        param_dict['cosmo_choice'],
                                        param_dict['cosmo_params']['H0'],
                                        ext))
    if (os.path.exists(z_comoving_file)) and (param_dict['remove_files']):
        ## Removing file
        os.remove(z_comoving_file)
    if not os.path.exists(z_comoving_file):
        ## Calculating comoving distance
        # `z_arr`     : Unitless
        # `d_comoving`:
        z_arr     = num.arange(zmin, zmax, dz)
        d_como    = cosmo_model.comoving_distance(z_arr).to(u.Mpc).value
        z_como_pd = pd.DataFrame({'z':z_arr, 'd_como':d_como})
        ## Saving to file
        z_como_pd.to_csv(z_comoving_file, sep=sep, index=False)
        cfutils.File_Exists(z_comoving_file)
    else:
        z_como_pd = pd.read_csv(z_comoving_file, sep=sep)
    ## Saving to `param_dict`
    param_dict['z_como_pd'] = z_como_pd
    if param_dict['verbose']:
        print('{0} Comoving Distance Table Calc .... Done'.format(Prog_msg))

    return param_dict

def tarball_create(param_dict, proj_dict, catl_ext='hdf5'):
    """
    Creates TAR object with mock catalogues, figures and README file

    Parameters
    -----------
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
    catl_path_arr = cfutils.Index(proj_dict['mock_cat_mc'], catl_ext)
    ## README file
    # Downloading working README file
    readme_file   = os.path.join(   proj_dict['base_dir'],
                                    'references',
                                    'README_RTD.pdf')
    cfutils.File_Download_needed(readme_file, param_dict['readme_url'])
    cfutils.File_Exists(readme_file)
    ## Saving to TAR file
    tar_file_path = os.path.join(   proj_dict['tar_dir'],
                                    '{0}_{1}_catls.tar.gz'.format(
                                        param_dict['survey_name'],
                                        param_dict['halotype']))
    # Opening file
    with tarfile.open(tar_file_path, mode='w:gz') as tf:
        tf.add(readme_file, arcname=os.path.basename(readme_file))
        for file_kk in catl_path_arr:
            ## Reading in DataFrame
            gal_pd_kk = cfreaders.read_hdf5_file_to_pandas_DF(file_kk)
            ## DataFrame `without` certain columns
            gal_pd_mod = catl_drop_cols(gal_pd_kk)
            ## Saving modified DataFrame to file
            file_mod_kk = file_kk+'.mod'
            cfreaders.pandas_df_to_hdf5_file(gal_pd_mod, file_mod_kk, key='\gal_catl')
            cfutils.File_Exists(file_mod_kk)
            # Saving to Tar-file
            tf.add(file_mod_kk, arcname=os.path.basename(file_kk))
            # Deleting extra file
            os.remove(file_mod_kk)
    tf.close()
    cfutils.File_Exists(tar_file_path)
    if param_dict['verbose']:
        print('{0} TAR file saved as: {1}'.format(Prog_msg, tar_file_path))

## -----------| Halobias-related functions |----------- ##

def hb_file_construction_extras(param_dict, proj_dict):
    """
    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    param_dict: python dictionary
        dictionary with updated 'hb_file_mod' key, which is the 
        path to the file with number of galaxies per halo ID.
    """
    Prog_msg = param_dict['Prog_msg']
    ## Local halobias file
    hb_local = param_dict['files_dict']['hb_file_local']
    ## HaloID extras file
    hb_file_mod = os.path.join(proj_dict['hb_files_dir'],
                                    os.path.basename(hb_local)+'.mod')
    ## Reading in file
    print('{0} Reading in `hb_file`...'.format(Prog_msg))
    with open(hb_local,'rb') as hb:
        idat    = cfreaders.fast_food_reader('int'  , 5, hb)
        fdat    = cfreaders.fast_food_reader('float', 9, hb)
        znow    = cfreaders.fast_food_reader('float', 1, hb)[0]
        ngal    = int(idat[1])
        lbox    = int(fdat[0])
        x_arr   = cfreaders.fast_food_reader('float' , ngal, hb)
        y_arr   = cfreaders.fast_food_reader('float' , ngal, hb)
        z_arr   = cfreaders.fast_food_reader('float' , ngal, hb)
        vx_arr  = cfreaders.fast_food_reader('float' , ngal, hb)
        vy_arr  = cfreaders.fast_food_reader('float' , ngal, hb)
        vz_arr  = cfreaders.fast_food_reader('float' , ngal, hb)
        halom   = cfreaders.fast_food_reader('float' , ngal, hb)
        cs_flag = cfreaders.fast_food_reader('int'   , ngal, hb)
        haloid  = cfreaders.fast_food_reader('int'   , ngal, hb)
    ##
    ## Counter of HaloIDs
    haloid_counts = Counter(haloid)
    # Array of `gals` in each `haloid`
    haloid_ngal = [[] for x in range(ngal)]
    for kk, halo_kk in enumerate(haloid):
        haloid_ngal[kk] = haloid_counts[halo_kk]
    haloid_ngal = num.asarray(haloid_ngal).astype(int)
    ## Converting to Pandas DataFrame
    # Dictionary
    hb_dict = {}
    hb_dict['x'          ] = x_arr
    hb_dict['y'          ] = y_arr
    hb_dict['z'          ] = z_arr
    hb_dict['vx'         ] = vx_arr
    hb_dict['vy'         ] = vy_arr
    hb_dict['vz'         ] = vz_arr
    hb_dict['halom'      ] = halom
    hb_dict['loghalom'   ] = num.log10(halom)
    hb_dict['cs_flag'    ] = cs_flag
    hb_dict['haloid'     ] = haloid
    hb_dict['haloid_ngal'] = haloid_ngal
    # To DataFrame
    hb_cols = ['x','y','z','vx','vy','vz','halom','loghalom',
                'cs_flag','haloid','haloid_ngal']
    hb_pd   = pd.DataFrame(hb_dict)[hb_cols]
    ## Saving to file
    hb_pd.to_csv(hb_file_mod, sep=" ", columns=hb_cols, 
        index=False, header=False)
    cfutils.File_Exists(hb_file_mod)
    ## Assigning to `param_dict`
    param_dict['hb_file_mod'] = hb_file_mod
    param_dict['hb_cols'    ] = hb_cols
    ## Testing `lbox`
    try:
        assert(lbox==param_dict['size_cube'])
    except:
        msg = '{0} `lbox` ({1}) does not match `size_cube` ({2})!'.format(
            Prog_msg, lbox, param_dict['size_cube'])
        raise ValueError(msg)
    # Message
    print('\n{0} Halo_ngal file: {1}'.format(Prog_msg, hb_file_mod))
    print('{0} Creating file with Ngals in each halo ... Complete'.format(Prog_msg))

    return param_dict

def cen_sat_distance_calc(clf_pd, param_dict):
    """
    Computes the distance between the central and its satellites in a given 
    DM halo

    Parameters
    ----------
    clf_pd: pandas DataFrame
        DataFrame with information from the CLF process
            - x, y, z, vx, vy, vz: Positions and velocity components
            - log(Mhalo): log-base 10 of the DM halo mass
            - `cs_flag`: Central (1) / Satellite (0) designation
            - `haloid`: ID of the galaxy's host DM halo
            - `halo_ngal`: Total number of galaxies in the DM halo
            - `M_r`: r-band absolute magnitude from ECO via abundance matching
            - `galid`: Galaxy ID

    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    ----------
    clf_pd: pandas DataFrame
        Updated version of `clf_pd` with new columns of distances
        New key: `dist_c` --> Distance to the satellite's central galaxy
    """
    ##
    ## TO DO: Fix issue with central and satellite being really close to 
    ##        the box boundary, therefore having different final distances
    ##
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Distance-Central Assignment ....'.format(Prog_msg))
    ## Centrals and Satellites
    cens            = param_dict['cens']
    sats            = param_dict['sats']
    dist_c_label    = 'dist_c'
    dist_sq_c_label = 'dist_c_sq'
    ## Galaxy coordinates
    coords  = ['x'   ,'y'   , 'z'  ]
    coords2 = ['x_sq','y_sq','z_sq']
    ## Unique HaloIDs
    haloid_unq = num.unique(clf_pd.loc[clf_pd['halo_ngal'] != 1,'haloid'])
    n_halo_unq = len(haloid_unq)
    ## CLF columns
    clf_cols = ['x','z','y','haloid','cs_flag']
    ## Copy of `clf_pd`
    clf_pd_mod = clf_pd[clf_cols].copy()
    ## Initializing new column in `clf_pd`
    clf_pd_mod.loc[:,dist_sq_c_label] = num.zeros(clf_pd.shape[0])
    ## Positions squared
    clf_pd_mod.loc[:,'x_sq'] = clf_pd_mod['x']**2
    clf_pd_mod.loc[:,'y_sq'] = clf_pd_mod['y']**2
    clf_pd_mod.loc[:,'z_sq'] = clf_pd_mod['z']**2
    ## Looping over number of halos
    # ProgressBar properties
    for ii, halo_ii in enumerate(tqdm(haloid_unq)):
        ## Halo ID subsample
        halo_ii_pd   = clf_pd_mod.loc[clf_pd_mod['haloid']==halo_ii]
        ## Cens and Sats DataFrames
        cens_coords = halo_ii_pd.loc[halo_ii_pd['cs_flag']==cens, coords]
        sats_coords = halo_ii_pd.loc[halo_ii_pd['cs_flag']==sats, coords]
        sats_idx    = sats_coords.index.values
        ## Distances from central galaxy
        cens_coords_mean = cens_coords.mean(axis=0).values
        ## Difference in coordinates
        dist_sq_arr = num.sum(
            sats_coords.subtract(cens_coords_mean, axis=1).values**2, axis=1)
        ## Assigning distances to each satellite
        clf_pd_mod.loc[sats_idx, dist_sq_c_label] = dist_sq_arr
    ##
    ## Taking the square root of distances
    clf_pd_mod.loc[:,dist_c_label] = (clf_pd_mod[dist_sq_c_label].values)**.5
    ##
    ## Assigning it to `clf_pd`
    clf_pd.loc[:, dist_c_label] = clf_pd_mod[dist_c_label].values
    if param_dict['verbose']:
        print('{0} Distance-Central Assignment .... Done'.format(Prog_msg))

    return clf_pd

## -----------| CLF-related functions |----------- ##

def clf_galprop_test(param_dict, proj_dict):
    """
    Tests whether or not the `FINAL` CLF file exists

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    -----------
    param_dict: python dictionary
        dictionary with `project` variables + `clf_opt` option, which 
        tells whether or not the final CLF file version exists
    """
    ## Local halobias file
    hb_local = param_dict['files_dict']['hb_file_local']
    ## CLF Output file - ASCII and FF
    clf_galprop_out = os.path.join( proj_dict['clf_dir'],
                            os.path.basename(hb_local) +'.clf.galprop')
    ## Checking if file exists
    if os.path.exists(clf_galprop_out):
        if param_dict['remove_files']:
            os.remove(clf_galprop_out)
            clf_opt = False
            # Saving to `param_dict`
            param_dict['clf_opt'] = clf_opt
        else:
            clf_opt = True
            clf_pd  = cfreaders.read_pandas_hdf5(clf_galprop_out)
            # Saving to `param_dict`
            param_dict['clf_opt'] = clf_opt
            param_dict['clf_pd' ] = clf_pd
    else:
        param_dict['clf_opt'] = False
    ##
    ## Saving name of `clf_galprop_out`
    param_dict['clf_galprop_out'] = clf_galprop_out

    return param_dict

def clf_assignment(param_dict, proj_dict, choice_survey=2):
    """
    Computes the conditional luminosity function on the halobias file
    
    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ----------
    clf_pd: pandas DataFrame
        DataFrame with information from the CLF process
        Format:
            - x, y, z, vx, vy, vz: Positions and velocity components
            - log(Mhalo): log-base 10 of the DM halo mass
            - `cs_flag`: Central (1) / Satellite (0) designation
            - `haloid`: ID of the galaxy's host DM halo
            - `halo_ngal`: Total number of galaxies in the DM halo
            - `M_r`: r-band absolute magnitude from ECO via abundance matching
            - `galid`: Galaxy ID
    """
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} CLF Assignment ....'.format(Prog_msg))
    ## Local halobias file
    hb_local = param_dict['files_dict']['hb_file_local']
    ## CLF Output file - ASCII and FF
    hb_clf_out = os.path.join(  proj_dict['clf_dir'],
                                os.path.basename(hb_local) +'.clf')
    hb_clf_out_ff = hb_clf_out + '.ff'
    ## HOD dictionary
    hod_dict = param_dict['hod_dict']
    ## CLF Executable
    clf_exe = os.path.join( cwpaths.get_code_c(),
                            'CLF',
                            'CLF_with_ftread')
    cfutils.File_Exists(clf_exe)
    ## CLF Commands - Executing in the terminal commands
    cmd_arr = [ clf_exe,
                hod_dict['logMmin'],
                param_dict['clf_type'],
                param_dict['hb_file_mod'],
                param_dict['size_cube'],
                param_dict['hod_dict']['znow'],
                hb_clf_out_ff,
                choice_survey,
                param_dict['files_dict']['eco_lum_file_local'],
                hb_clf_out]
    cmd_str = '{0} {1} {2} {3} {4} {5} {6} {7} < {8} > {9}'
    cmd     = cmd_str.format(*cmd_arr)
    print(cmd)
    subprocess.call(cmd, shell=True)
    ##
    ## Reading in CLF file
    clf_cols = ['x','y','z','vx','vy','vz',
                'loghalom','cs_flag','haloid','halo_ngal','M_r','galid']
    clf_pd   = pd.read_csv(hb_clf_out, sep='\s+', header=None, names=clf_cols)
    clf_pd.loc[:,'galid'] = clf_pd['galid'].astype(int)
    ## Copy of galaxy positions
    for coord_zz in ['x','y','z']:
        clf_pd.loc[:, coord_zz+'_orig'] = clf_pd[coord_zz].values
    ## Galaxy indices
    clf_pd.loc[:,'idx'] = clf_pd.index.values
    ##
    ## Remove extra files
    os.remove(hb_clf_out_ff)
    if param_dict['verbose']:
        print('{0} CLF Assignment .... Done'.format(Prog_msg))

    return clf_pd

def mr_survey_matching(clf_pd, param_dict, proj_dict):
    """
    Finds the closest r-band absolute magnitude from ECO catalogue 
    and assigns them to mock galaxies

    Parameters
    -------------
    clf_pd: pandas DataFrame
        DataFrame containing information from Halobias + CLF procedures
    
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    -------------
    clf_galprop_pd: pandas DataFrame
        DataFrame with updated values.
        New values included:
            - Morphology
            - Stellar mass
            - r-band apparent magnitude
            - u-band apparent magnitude
            - FSMGR
            - `Match_Flag`:
            - MHI
            - Survey flag: {1 == ECO, 0 == Resolve B}
    """
    Prog_msg   = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} ECO/Resolve Galaxy Prop. Assign. ....'.format(Prog_msg))
    ## Constants
    failval    = 0.
    ngal_mock  = len(clf_pd)
    ## Survey flags
    eco_flag = 1
    res_flag = 0
    ## Copy of `clf_pd`
    clf_pd_mod = clf_pd.copy()
    ## Filenames
    eco_phot_file   = param_dict['files_dict']['eco_phot_file_local']
    eco_mhi_file    = param_dict['files_dict']['mhi_file_local']
    res_b_phot_file = param_dict['files_dict']['res_b_phot_file_local']
    ## ECO Photometry catalogue
    #  - reading in dictionary
    eco_phot_dict = readsav(eco_phot_file, python_dict=True)
    #  - Converting to Pandas DataFrame
    eco_phot_pd   = Table(eco_phot_dict).to_pandas()
    ##
    ## ECO - MHI measurements
    eco_mhi_pd    = pd.read_csv(eco_mhi_file, sep='\s+', names=['MHI'])
    ## Appending to `eco_phot_pd`
    try:
        assert(len(eco_phot_pd) == len(eco_mhi_pd))
        ## Adding to DataFrame
        eco_phot_pd.loc[:,'MHI'] = eco_mhi_pd['MHI'].values
    except:
        msg = '{0} `len(eco_phot_pd)` != `len(eco_mhi_pd)`! Unequal lengths!'
        msg = msg.format(Prog_msg)
        raise ValueError(msg)
    ##
    ## Cleaning up DataFrame - r-band absolute magnitudes
    eco_phot_mod_pd = eco_phot_pd.loc[(eco_phot_pd['goodnewabsr'] != failval) &
                      (eco_phot_pd['goodnewabsr'] < param_dict['mr_limit'])]
    eco_phot_mod_pd.reset_index(inplace=True, drop=True)
    ##
    ## Reading in `RESOLVE B` catalogue - r-band absolute magnitudes
    res_phot_pd = Table(fits.getdata(res_b_phot_file)).to_pandas()
    res_phot_mod_pd = res_phot_pd.loc[\
                        (res_phot_pd['ABSMAGR'] != failval) &
                        (res_phot_pd['ABSMAGR'] <= param_dict['mr_res_b']) &
                        (res_phot_pd['ABSMAGR'] >  param_dict['mr_eco'  ])]
    res_phot_mod_pd.reset_index(inplace=True, drop=True)
    ##
    ## Initializing arrays
    morph_arr       = [[] for x in range(ngal_mock)]
    dex_rmag_arr    = [[] for x in range(ngal_mock)]
    dex_umag_arr    = [[] for x in range(ngal_mock)]
    logmstar_arr    = [[] for x in range(ngal_mock)]
    fsmgr_arr       = [[] for x in range(ngal_mock)]
    survey_flag_arr = [[] for x in range(ngal_mock)]
    mhi_arr         = [[] for x in range(ngal_mock)]
    ##
    ## Assigning galaxy properties to mock galaxies
    #
    clf_mr_arr = clf_pd_mod['M_r'].values
    eco_mr_arr = eco_phot_mod_pd['goodnewabsr'].values
    res_mr_arr = res_phot_mod_pd['ABSMAGR'].values
    ## Galaxy properties column names
    eco_cols = ['goodmorph','rpsmoothrestrmagnew','rpsmoothrestumagnew',
                'rpgoodmstarsnew','rpmeanssfr']
    res_cols = ['MORPH', 'SMOOTHRESTRMAG','SMOOTHRESTUMAG','MSTARS',
                'MODELFSMGR']
    # Looping over all galaxies
    for ii in tqdm(range(ngal_mock)):
        ## Galaxy r-band absolute magnitude
        gal_mr = clf_mr_arr[ii]
        ## Choosing which catalogue to use
        if gal_mr <= param_dict['mr_eco']:
            idx_match  = closest_val(gal_mr, eco_mr_arr)
            survey_tag = eco_flag
            ## Galaxy Properties
            (   morph_val,
                rmag_val ,
                umag_val ,
                logmstar_val,
                fsmgr_val) = eco_phot_mod_pd.loc[idx_match, eco_cols].values
            # MHI value
            mhi_val   = 10**(eco_phot_mod_pd['MHI'][idx_match] + logmstar_val)
        elif (gal_mr > param_dict['mr_eco']) and (gal_mr <= param_dict['mr_res_b']):
            idx_match  = closest_val(gal_mr, res_mr_arr)
            survey_tag = res_flag
            ## Galaxy Properties
            (   morph_val,
                rmag_val ,
                umag_val ,
                mstar_val,
                fsmgr_val) = res_phot_mod_pd.loc[idx_match, res_cols]
            ## MHI value
            mhi_val = res_phot_mod_pd.loc[idx_match, 'MHI']
            ## Fixing issue with units
            logmstar_val = num.log10(mstar_val)
        ##
        ## Assigning them to arrays
        morph_arr       [ii] = morph_val
        dex_rmag_arr    [ii] = rmag_val
        dex_umag_arr    [ii] = umag_val
        logmstar_arr    [ii] = logmstar_val
        fsmgr_arr       [ii] = fsmgr_val
        mhi_arr         [ii] = mhi_val
        survey_flag_arr [ii] = survey_tag
    ##
    ## Assigning them to `clf_pd_mod`
    clf_pd_mod.loc[:,'morph'      ] = morph_arr
    clf_pd_mod.loc[:,'rmag'       ] = dex_rmag_arr
    clf_pd_mod.loc[:,'umag'       ] = dex_umag_arr
    clf_pd_mod.loc[:,'logmstar'   ] = logmstar_arr
    clf_pd_mod.loc[:,'fsmgr'      ] = fsmgr_arr
    clf_pd_mod.loc[:,'mhi'        ] = mhi_arr
    clf_pd_mod.loc[:,'survey_flag'] = survey_flag_arr
    clf_pd_mod.loc[:,'u_r'        ] = clf_pd_mod['umag'] - clf_pd_mod['rmag']
    ##
    ## Dropping all other columns
    galprop_cols = ['morph','rmag','umag','logmstar','fsmgr','mhi',
                    'survey_flag', 'u_r']
    clf_pd_mod_prop = clf_pd_mod[galprop_cols].copy()
    ##
    ## Merging DataFrames
    clf_galprop_pd = pd.merge(clf_pd, clf_pd_mod_prop, 
                                left_index=True, right_index=True)
    ## Saving DataFrame
    cfreaders.pandas_df_to_hdf5_file(clf_galprop_pd, param_dict['clf_galprop_out'],
        key='galcatl')
    cfutils.File_Exists(param_dict['clf_galprop_out'])
    # Print statement
    if param_dict['verbose']:
        print('{0} ECO/Resolve Galaxy Prop. Assign. ....Done'.format(Prog_msg))

    return clf_galprop_pd

## -----------| Makemock-related functions |----------- ##

def resolve_a_geometry_mocks(clf_pd, param_dict, proj_dict):
    """
    Carves out the geometry of the `RESOLVE-A` survey and produces set 
    of mock catalogues
    
    Parameters
    -------------
    clf_pd: pandas DataFrame
        DataFrame containing information from Halobias + CLF procedures
    
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Creating Mock Catalogues ....'.format(Prog_msg))

    ## Coordinates dictionary
    coord_dict    = param_dict['coord_dict'].copy()
    ## Coordinate and Dataframe lists
    pos_coords_mocks = []
    ##########################################
    ###### ----- 1st Set of Mocks  -----######
    ##########################################
    clf_1_pd     = copy.deepcopy(clf_pd)
    coord_1_dict = coord_dict.copy()
    # Coordinates
    x_init_1  =  56.
    y_init_1  = -25.
    y_delta_1 =  43.
    z_init_1  =  3.
    z_delta_1 =  16.
    # Determning positions
    x_pos_1_arr = []
    y_pos_1_arr = []
    z_pos_1_arr = []
    # X
    x_pos_1_arr.append(56.)
    # Y
    while (180. - y_init_1) >= y_delta_1:
        y_pos_1_arr.append(y_init_1)
        y_init_1 += y_delta_1
    # Z
    while (180. - z_init_1) >= z_delta_1:
        z_pos_1_arr.append(z_init_1)
        z_init_1 += z_delta_1
    ## Looping over positions
    ncatls = 0
    for aa in x_pos_1_arr:
        for bb in y_pos_1_arr:
            for cc in z_pos_1_arr:
                ## Appending positions
                pos_coords_mocks.append([aa, bb, cc, 
                                         clf_1_pd.copy(), coord_1_dict])
                ncatls += 1
    ##########################################
    ###### ----- 2nd Set of Mocks  -----######
    ##########################################
    clf_2_pd     = copy.deepcopy(clf_pd)
    coord_2_dict = coord_dict.copy()
    # Changing coordinates
    x_arr_2 = clf_2_pd['x'].copy()
    y_arr_2 = clf_2_pd['y'].copy()
    clf_2_pd.loc[:,'x'] = y_arr_2
    clf_2_pd.loc[:,'y'] = x_arr_2
    # Determining positions
    x_init_2  = 56.
    y_init_2  = 100.
    z_init_2  = 3.
    z_delta_2 = 16.
    # Determning positions
    x_pos_2_arr = []
    y_pos_2_arr = []
    z_pos_2_arr = []
    # X
    x_pos_2_arr.append(56.)
    # Y
    y_pos_2_arr.append(100.)
    # Z
    while (180. - z_init_2) >= z_delta_2:
        z_pos_2_arr.append(z_init_2)
        z_init_2 += z_delta_2
    ## Looping over positions
    for aa in x_pos_2_arr:
        for bb in y_pos_2_arr:
            for cc in z_pos_2_arr:
                ## Appending positions
                pos_coords_mocks.append([aa, bb, cc,
                                         clf_2_pd, coord_2_dict])
                ## Incrementing values
                ncatls += 1
    ##########################################
    ###### ----- 3rd Set of Mocks  -----######
    ##########################################
    clf_3_pd     = copy.deepcopy(clf_pd)
    coord_3_dict = coord_dict.copy()
    # Changing coordinates
    x_arr_3 = clf_3_pd['x'].copy()
    z_arr_3 = clf_3_pd['z'].copy()
    clf_3_pd.loc[:,'x'] = z_arr_3
    clf_3_pd.loc[:,'z'] = x_arr_3
    ## Determining positions
    x_init_3   = 90.
    y_init_3   = 100.
    z_delta_3  = 16.
    z_init_3   = z_delta_3
    z_buffer_3 = 5.
    # Determning positions
    x_pos_3_arr = []
    y_pos_3_arr = []
    z_pos_3_arr = []
    # X
    x_pos_3_arr.append(90.)
    # Y
    y_pos_3_arr.append(100.)
    # Z
    while (180. - 111. - z_init_3) >= z_buffer_3:
        z_pos_3_arr.append(180. - z_init_3)
        z_init_3 += z_delta_3
    ## Looping over positions
    for aa in x_pos_3_arr:
        for bb in y_pos_3_arr:
            for cc in z_pos_3_arr:
                ## Appending positions
                pos_coords_mocks.append([aa, bb, cc,
                                         clf_3_pd, coord_3_dict])
                ## Incrementing values
                ncatls += 1
    ##############################################
    ## Creating mock catalogues
    ##############################################
    ##
    ## ----| Multiprocessing |---- ##
    ##
    ## Number of catalogues
    n_catls = len(pos_coords_mocks)
    ## CPU counts
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ## Step-size for each CPU
    if cpu_number <= len(pos_coords_mocks):
        catl_step = int(n_catls / cpu_number)
        memb_arr  = num.arange(0, n_catls+1, catl_step)
    else:
        catl_step = int((n_catls/cpu_number)**-1)
        memb_arr  = num.arange(0, n_catls+1)
    ## Array with designated catalogue numbers for each CPU
    memb_arr[-1] = n_catls
    ## Tuples of the ID of each catalogue
    memb_tuples = num.asarray([(memb_arr[xx], memb_arr[xx+1])
                            for xx in range(memb_arr.size-1)])
    ## Assigning `memb_tuples` to function `multiprocessing_catls`
    print('{0} Creating Mock Catalogues ....'.format(param_dict['Prog_msg']))
    procs = []
    for ii in range(len(memb_tuples)):
        ## Defining `proc` element
        proc = Process(target=multiprocessing_catls,
                        args=(  memb_tuples[ii], pos_coords_mocks, param_dict, 
                                proj_dict, ii))
        # Appending to main `procs` list
        procs.append(proc)
        proc.start()
    ##
    ## Joining `procs`
    for proc in procs:
        proc.join()
    ##
    ## Reinitializing `param_dict` to None
    if param_dict['verbose']:
        print('\n{0} Creating Mock Catalogues .... Done'.format(Prog_msg))
    param_dict = None

def resolve_b_geometry_mocks(clf_pd, param_dict, proj_dict):
    """
    Carves out the geometry of the `RESOLVE-B` survey and produces set 
    of mock catalogues
    
    Parameters
    -------------
    clf_pd: pandas DataFrame
        DataFrame containing information from Halobias + CLF procedures
    
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
    """
    ## Constants
    Prog_msg = param_dict['Prog_msg']
    if param_dict['verbose']:
        print('{0} Creating Mock Catalogues ....'.format(Prog_msg))

    ## Coordinates dictionary
    coord_dict    = param_dict['coord_dict'].copy()
    ## Coordinate and Dataframe lists
    pos_coords_mocks = []
    ##########################################
    ###### ----- 1st Set of Mocks  -----######
    ##########################################
    clf_1_pd     = copy.deepcopy(clf_pd)
    coord_1_dict = coord_dict.copy()
    # Coordinates
    x_init_1  =  56.
    y_init_1  = -25.
    y_delta_1 =  43.
    z_init_1  =  3.
    z_delta_1 =  16.
    # Determning positions
    x_pos_1_arr = []
    y_pos_1_arr = []
    z_pos_1_arr = []
    # X
    x_pos_1_arr.append(56.)
    # Y
    while (180. - y_init_1) >= y_delta_1:
        y_pos_1_arr.append(y_init_1)
        y_init_1 += y_delta_1
    # Z
    while (180. - z_init_1) >= z_delta_1:
        z_pos_1_arr.append(z_init_1)
        z_init_1 += z_delta_1
    ## Looping over positions
    ncatls = 0
    for aa in range(len(x_pos_1_arr)):
        for bb in range(len(y_pos_1_arr)):
            for cc in range(len(z_pos_1_arr)):
                ## Appending positions
                pos_coords_mocks.append([aa, bb, cc, 
                                         clf_1_pd.copy(), coord_1_dict])
                ncatls += 1
    ##########################################
    ###### ----- 2nd Set of Mocks  -----######
    ##########################################
    clf_2_pd     = copy.deepcopy(clf_pd)
    coord_2_dict = coord_dict.copy()
    # Changing coordinates
    x_arr_2 = clf_2_pd['x'].copy()
    y_arr_2 = clf_2_pd['y'].copy()
    clf_2_pd.loc[:,'x'] = y_arr_2
    clf_2_pd.loc[:,'y'] = x_arr_2
    # Determining positions
    x_init_2  = 56.
    y_init_2  = 100.
    z_init_2  = 3.
    z_delta_2 = 16.
    # Determning positions
    x_pos_2_arr = []
    y_pos_2_arr = []
    z_pos_2_arr = []
    # X
    x_pos_2_arr.append(56.)
    # Y
    y_pos_2_arr.append(100.)
    # Z
    while (180. - z_init_2) >= z_delta_2:
        z_pos_2_arr.append(z_init_2)
        z_init_2 += z_delta_2
    ## Looping over positions
    for aa in range(len(x_pos_2_arr)):
        for bb in range(len(y_pos_2_arr)):
            for cc in range(len(z_pos_2_arr)):
                ## Appending positions
                pos_coords_mocks.append([aa, bb, cc,
                                         clf_2_pd, coord_2_dict])
                ## Incrementing values
                ncatls += 1
    ##########################################
    ###### ----- 3rd Set of Mocks  -----######
    ##########################################
    clf_3_pd     = copy.deepcopy(clf_pd)
    coord_3_dict = coord_dict.copy()
    # Changing coordinates
    x_arr_3 = clf_3_pd['x'].copy()
    z_arr_3 = clf_3_pd['z'].copy()
    clf_3_pd.loc[:,'x'] = z_arr_3
    clf_3_pd.loc[:,'z'] = x_arr_3
    ## Determining positions
    x_init_3   = 90.
    y_init_3   = 100.
    z_delta_3  = 16.
    z_init_3   = z_delta_3
    z_buffer_3 = 5.
    # Determning positions
    x_pos_3_arr = []
    y_pos_3_arr = []
    z_pos_3_arr = []
    # X
    x_pos_3_arr.append(90.)
    # Y
    y_pos_3_arr.append(100.)
    # Z
    while (180. - 111. - z_init_3) >= z_buffer_3:
        z_pos_3_arr.append(180. - z_init_3)
        z_init_3 += z_delta_3
    ## Looping over positions
    for aa in range(len(x_pos_3_arr)):
        for bb in range(len(y_pos_3_arr)):
            for cc in range(len(z_pos_3_arr)):
                ## Appending positions
                pos_coords_mocks.append([aa, bb, cc,
                                         clf_3_pd, coord_3_dict])
                ## Incrementing values
                ncatls += 1
    ##############################################
    ## Creating mock catalogues
    ##############################################
    ##
    ## ----| Multiprocessing |---- ##
    ##
    ## Number of catalogues
    n_catls = len(pos_coords_mocks)
    ## CPU counts
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ## Step-size for each CPU
    if cpu_number <= len(pos_coords_mocks):
        catl_step = int(n_catls / cpu_number)
        memb_arr  = num.arange(0, n_catls+1, catl_step)
    else:
        catl_step = int((n_catls/cpu_number)**-1)
        memb_arr  = num.arange(0, n_catls+1)
    ## Array with designated catalogue numbers for each CPU
    memb_arr[-1] = n_catls
    ## Tuples of the ID of each catalogue
    memb_tuples = num.asarray([(memb_arr[xx], memb_arr[xx+1])
                            for xx in range(memb_arr.size-1)])
    ## Assigning `memb_tuples` to function `multiprocessing_catls`
    print('{0} Creating Mock Catalogues ....'.format(param_dict['Prog_msg']))
    procs = []
    for ii in range(len(memb_tuples)):
        ## Defining `proc` element
        proc = Process(target=multiprocessing_catls,
                        args=(  memb_tuples[ii], pos_coords_mocks, param_dict, 
                                proj_dict, ii))
        # Appending to main `procs` list
        procs.append(proc)
        proc.start()
    ##
    ## Joining `procs`
    for proc in procs:
        proc.join()
    ##
    ## Reinitializing `param_dict` to None
    if param_dict['verbose']:
        print('\n{0} Creating Mock Catalogues .... Done'.format(Prog_msg))
    param_dict = None


def eco_geometry_mocks(clf_pd, param_dict, proj_dict):
    """
    Carves out the geometry of the `ECO` survey and produces set 
    of mock catalogues
    
    Parameters
    -------------
    clf_pd: pandas DataFrame
        DataFrame containing information from Halobias + CLF procedures
    
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.
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
    clf_ul_pd     = copy.deepcopy(clf_pd)
    coord_dict_ul = coord_dict.copy()
    # Coordinates
    coord_dict_ul['ra_min']  = 90. - coord_dict_ul['ra_range']
    coord_dict_ul['ra_max']  = 90.
    coord_dict_ul['ra_diff'] = coord_dict_ul['ra_max_real'] - coord_dict_ul['ra_max']
    gap_ul     = 20.
    x_init_ul  = 10.
    y_init_ul  = param_dict['size_cube'] - coord_dict_ul['r_arr'][1] - 5.
    z_init_ul  = 10.
    z_delta_ul = gap_ul + coord_dict_ul['d_th']
    if coord_dict_ul['dec_min'] < 0.:
        z_init_ul += num.abs(coord_dict_ul['dec_min'])
    z_mocks_n_ul = int(num.floor(param_dict['size_cube']/z_delta_ul))
    ## Determining positions
    for kk in range(z_mocks_n_ul):
        pos_coords_mocks.append([   x_init_ul, y_init_ul, z_init_ul,
                                    clf_ul_pd.copy(), coord_dict_ul])
        z_init_ul += z_delta_ul
    ##############################################
    ###### ----- X-Y Upper Right Mocks -----######
    ##############################################
    clf_ur_pd     = copy.deepcopy(clf_pd)
    coord_dict_ur = copy.deepcopy(coord_dict_ul)
    # Coordinates
    coord_dict_ur['ra_min' ] = 90. - coord_dict_ur['ra_range']
    coord_dict_ur['ra_max' ] = 90.
    coord_dict_ur['ra_diff'] = coord_dict_ur['ra_max_real'] - coord_dict_ur['ra_max']
    x_init_ur = param_dict['size_cube'] - coord_dict_ur['r_arr'][1] - 5.
    y_init_ur = param_dict['size_cube'] - coord_dict_ur['r_arr'][1] - 10.
    z_init_ur = 10.
    if coord_dict_ur['dec_min'] < 0.:
        z_init_ur += num.abs(coord_dict_ur['dec_min'])
    z_mocks_n_ur = int(num.floor(param_dict['size_cube']/z_delta_ul))
    ## Determining positions
    for kk in range(z_mocks_n_ur):
        pos_coords_mocks.append([   x_init_ur, y_init_ur, z_init_ur,
                                    clf_ur_pd.copy(), coord_dict_ur])
        z_init_ur += z_delta_ul
    ##############################################
    ###### ----- X-Y Lower Left Mocks  -----######
    ##############################################
    clf_ll_pd     = copy.deepcopy(clf_pd)
    coord_dict_ll = copy.deepcopy(coord_dict_ur)
    ## Changing geometry
    coord_dict_ll['ra_min' ] = 180.
    coord_dict_ll['ra_max' ] = 180. + coord_dict_ll['ra_range']
    coord_dict_ll['ra_diff'] = coord_dict_ll['ra_max_real'] - coord_dict_ll['ra_max']
    assert( (coord_dict_ll['ra_min'] < coord_dict_ll['ra_max']) and 
            (coord_dict_ll['ra_min'] <= 360.) and
            (coord_dict_ll['ra_max'] <= 360.))
    ## New positions
    x_init_ll = coord_dict_ll['r_arr'][1]
    y_init_ll = coord_dict_ll['r_arr'][1]
    z_init_ll = 10.
    if coord_dict_ll['dec_min'] < 0.:
        z_init_ll += num.abs(coord_dict_ll['dec_min'])
    z_mocks_n_ll = int(num.floor(param_dict['size_cube']/z_delta_ul))
    ## Saving new positions
    for kk in range(z_mocks_n_ll):
        pos_coords_mocks.append([   x_init_ll, y_init_ll, z_init_ll,
                                    clf_ll_pd.copy(), coord_dict_ll])
        z_init_ll += z_delta_ul
    ##############################################
    ###### ----- X-Y Lower Right Mocks -----######
    ##############################################
    clf_lr_pd     = copy.deepcopy(clf_pd)
    coord_dict_lr = copy.deepcopy(coord_dict_ul)
    # Changing geometry
    coord_dict_lr['ra_min' ] = 270. - coord_dict_lr['ra_range']
    coord_dict_lr['ra_max' ] = 270.
    coord_dict_lr['ra_diff'] = coord_dict_lr['ra_max_real'] - coord_dict_lr['ra_max']
    # New positions
    x_init_lr = coord_dict_lr['r_arr'][1]
    x_init_lr = param_dict['size_cube'] - 10.
    y_init_lr = coord_dict_lr['r_arr'][1] - 15.
    z_init_lr = 10.
    if coord_dict_lr['dec_min'] < 0.:
        z_init_lr += num.abs(coord_dict_lr['dec_min'])
    z_mocks_n_lr = int(num.floor(param_dict['size_cube']/z_delta_ul))
    ## Saving new positions
    for kk in range(z_mocks_n_lr):
        pos_coords_mocks.append([   x_init_lr, y_init_lr, z_init_lr,
                                    clf_lr_pd.copy(), coord_dict_lr])
        z_init_lr += z_delta_ul
    ##############################################
    ## Creating mock catalogues
    ##############################################
    ##
    ## ----| Multiprocessing |---- ##
    ##
    ## Number of catalogues
    n_catls = len(pos_coords_mocks)
    ## CPU counts
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ## Step-size for each CPU
    if cpu_number <= len(pos_coords_mocks):
        catl_step = int(n_catls / cpu_number)
        memb_arr  = num.arange(0, n_catls+1, catl_step)
    else:
        catl_step = int((n_catls/cpu_number)**-1)
        memb_arr  = num.arange(0, n_catls+1)
    ## Array with designated catalogue numbers for each CPU
    memb_arr[-1] = n_catls
    ## Tuples of the ID of each catalogue
    memb_tuples = num.asarray([(memb_arr[xx], memb_arr[xx+1])
                            for xx in range(memb_arr.size-1)])
    ## Assigning `memb_tuples` to function `multiprocessing_catls`
    print('{0} Creating Mock Catalogues ....'.format(param_dict['Prog_msg']))
    procs = []
    for ii in range(len(memb_tuples)):
        ## Defining `proc` element
        proc = Process(target=multiprocessing_catls,
                        args=(  memb_tuples[ii], pos_coords_mocks, param_dict, 
                                proj_dict, ii))
        # Appending to main `procs` list
        procs.append(proc)
        proc.start()
    ##
    ## Joining `procs`
    for proc in procs:
        proc.join()
    ##
    ## Reinitializing `param_dict` to None
    if param_dict['verbose']:
        print('{0} Creating Mock Catalogues .... Done'.format(Prog_msg))
    param_dict = None

def makemock_catl(clf_ii, coord_dict_ii, zz_mock, param_dict, proj_dict):
    """
    Function that calculates distances and redshift-space distortions 
    for the galaxies that make it into the catalogues

    Parameters
    -----------
    clf_ii: pandas DataFrame
        DataFrame with the information on galaxies, along with position coords,
        velocities, etc.

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
    ## Filenames
    mock_catl_pd_file = os.path.join(   proj_dict['mock_cat_mgc'],
                                        '{0}_galcatl_cat_{1}.hdf5'.format(
                                            param_dict['survey'],
                                            zz_mock))
    ## Number of galaies
    clf_ngal    = len(clf_ii)
    speed_c = param_dict['const_dict']['c']
    ## Distances from observer to galaxies
    z_como_pd   = param_dict['z_como_pd']
    dc_max      = z_como_pd['d_como'].max()
    dc_z_interp = interp1d(z_como_pd['d_como'], z_como_pd['z'])
    ## Redshift-space distortions
    # Cartesina Coordinates
    cart_gals   = clf_ii[['x' ,'y' ,'z' ]].values
    vel_gals    = clf_ii[['vx','vy','vz']].values
    ## Initializing arrays
    r_dist_arr    = num.zeros(clf_ngal)
    ra_arr        = num.zeros(clf_ngal)
    dec_arr       = num.zeros(clf_ngal)
    cz_arr        = num.zeros(clf_ngal)
    cz_nodist_arr = num.zeros(clf_ngal)
    vel_tan_arr   = num.zeros(clf_ngal)
    vel_tot_arr   = num.zeros(clf_ngal)
    vel_pec_arr   = num.zeros(clf_ngal)
    # Looping over all galaxies
    for kk in tqdm(range(clf_ngal)):
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
    clf_ii.loc[:,'r_dist'   ] = r_dist_arr
    clf_ii.loc[:,'ra'       ] = ra_arr
    clf_ii.loc[:,'dec'      ] = dec_arr
    clf_ii.loc[:,'cz'       ] = cz_arr
    clf_ii.loc[:,'cz_nodist'] = cz_nodist_arr
    clf_ii.loc[:,'vel_tot'  ] = vel_tot_arr
    clf_ii.loc[:,'vel_tan'  ] = vel_tan_arr
    clf_ii.loc[:,'vel_pec'  ] = vel_pec_arr
    ##
    ## Selecting galaxies with `czmin` and `czmax` criteria
    #  Right Ascension
    if coord_dict_ii['ra_min'] < 0.:
        ra_min_mod = coord_dict_ii['ra_min'] + 360.
        mock_pd    = clf_ii.loc[(clf_ii['dec'] >= coord_dict_ii['dec_min']) &
                                (clf_ii['dec'] <= coord_dict_ii['dec_max']) &
                                (clf_ii['M_r'] != 0.)].copy()
        mock_pd    = mock_pd.loc[~( (mock_pd['ra'] < ra_min_mod) &
                                    (mock_pd['ra'] > coord_dict_ii['ra_max']))]
        # ra_idx1 = clf_ii.loc[(clf_ii['ra'] < (coord_dict_ii['ra_min'] + 360))&
        #                      (clf_ii['ra'] >  coord_dict_ii['ra_max'])].index
        # ra_idx1 = ra_idx1.values
        # idx_arr = num.arange(0, clf_ngal)
        # ra_idx  = num.delete(idx_arr, ra_idx1).astype(int)
    elif coord_dict_ii['ra_min'] >= 0.:
        mock_pd = clf_ii.loc[(clf_ii['ra'] >= coord_dict_ii['ra_min']) &
                             (clf_ii['ra'] <= coord_dict_ii['ra_max']) &
                             (clf_ii['dec'] >= coord_dict_ii['dec_min']) &
                             (clf_ii['dec'] <= coord_dict_ii['dec_max']) &
                             (clf_ii['M_r'] != 0.)].copy()
        # ra_idx = clf_ii.loc[(clf_ii['ra'] >= coord_dict_ii['ra_min']) &
        #                     (clf_ii['ra'] <= coord_dict_ii['ra_max'])].index
        # ra_idx = ra_idx.values
    # Declination
    # dec_idx = clf_ii.loc[   (clf_ii['dec'] >= coord_dict_ii['dec_min']) &
    #                         (clf_ii['dec'] <= coord_dict_ii['dec_max'])].index.values
    # mr_idx = clf_ii.loc[clf_ii['M_r'] != 0.].index.values
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

def catl_create_main(zz_mock, pos_coords_mocks_zz, param_dict, proj_dict):
    """
    Distributes the analyis of the creation of mock catalogues into 
    more than 1 processor

    Parameters
    -----------
    zz_mock: int
        number of the mock catalogue being analyzed

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
        clf_ii       ,
        coord_dict_ii) = pos_coords_mocks_zz_copy
    ## Size of cube
    size_cube = float(param_dict['size_cube'])
    ## Cartesian coordinates
    pos_zz = num.asarray([x_ii, y_ii, z_ii])
    ## Formatting new positions
    ## Placing the observer at `pos_zz` and centering coordinates to center 
    ## of box
    for kk, coord_kk in enumerate(['x','y','z']):
        ## Moving observer
        clf_ii.loc[:,coord_kk] = clf_ii[coord_kk] - pos_zz[kk]
        ## Periodic boundaries
        clf_ii_neg = clf_ii.loc[clf_ii[coord_kk] <= -(size_cube/2.)].index
        clf_ii_pos = clf_ii.loc[clf_ii[coord_kk] >=  (size_cube/2.)].index
        ## Fixing negative values
        if len(clf_ii_neg) != 0:
            clf_ii.loc[clf_ii_neg, coord_kk] += size_cube
        if len(clf_ii_pos) != 0:
            clf_ii.loc[clf_ii_pos, coord_kk] -= size_cube
    ##
    ## Interpolating values for redshift and comoving distance
    ## and adding redshift-space distortions
    (   mock_pd     ,
        mock_zz_file) = makemock_catl(  clf_ii, coord_dict_ii, zz_mock,
                                        param_dict, proj_dict)
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
    writing_to_output_file(mockgal_pd, mockgroup_pd, zz_mock,
        param_dict, proj_dict, perf_catl=False)

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
    cosmo_model = param_dict['cosmo_model']
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
    survey_vol = survey_vol_calc(   [0, ra_range],
                                    [dec_min, dec_max],
                                    r_arr)
    ##
    ## Survey height, and other geometrical factors
    (   h_total,
        h1     ,
        s1_top ,
        s2     ) = geometry_calc(r_arr[0], r_arr[1], ra_range)
    (   h_side ,
        h2     ,
        s1_side,
        d_th   ) = geometry_calc(r_arr[0], r_arr[1], dec_range)
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
        prop_gal    = 'M_r'
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
    if param_dict['catl_type'] == 'mstar':
        for group_zz in tqdm(range(n_groups)):
            ## Stellar mass
            group_prop = gal_pd.loc[gal_pd['groupid']==group, prop_gal]
            group_log_prop_tot = num.log10(num.sum(10**group_prop))
            ## Saving to array
            group_prop_arr[group_zz] = group_log_prop_tot
    # Luminosity-based
    elif param_dict['catl_type'] == 'mr':
        for group_zz in tqdm(range(n_groups)):
            ## Total abs. magnitude of the group
            group_prop = gal_pd.loc[gal_pd['groupid']==group_zz, prop_gal]
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
    hmf_pd = param_dict['hmf_pd']
    ## Halo mass
    Mh_ab = cm.abundance_matching.abundance_matching_f(
                                        group_prop_arr,
                                        hmf_pd,
                                        volume1=param_dict['survey_vol'],
                                        reverse=reverse_opt,
                                        dict2_names=['logM', 'ngtm'],
                                        dens2_opt=True)
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
                'idx','vel_pec','ra_orig']
    # New object `without` these columns
    gal_pd_mod = gal_pd.loc[:,~gal_pd.columns.isin(gal_cols)].copy()

    return gal_pd_mod

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
    haloid_counts = Counter(gal_pd['haloid'])
    haloid_arr    = num.unique(gal_pd['haloid'])
    ## Mean cz's
    haloid_z = num.array([gal_pd.loc[gal_pd['haloid']==xx,'cz'].mean() for \
                        xx in haloid_arr])/speed_c.value
    ## Halo masses
    haloid_mass = num.array([gal_pd.loc[gal_pd['haloid']==xx,'loghalom'].mean() for \
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
        haloid_ngal_sim = num.array([gal_pd.loc[gal_pd['haloid']==xx, 'halo_ngal'].values[0]\
                            for xx in haloid_arr])
        ## Chaning `rvir` values to zeros if halo is not complete
        rvir_bool = [1 if haloid_ngal_cat[xx]==haloid_ngal_sim[xx] else 0 \
                        for xx in range(len(haloid_arr))]
        rvir[rvir_bool] = repl_val
    ## Saving to DataFrame
    rvir_pd = pd.DataFrame({'haloid':haloid_arr, 'halo_rvir':rvir})
    ## Merging DataFrames
    # Galaxies
    mockgal_pd_new = pd.merge(  left=gal_pd      ,
                                right=rvir_pd    ,
                                how='left'       ,
                                left_on='haloid' ,
                                right_on='haloid')
    if param_dict['verbose']:
        print('{0} Halo Rvir Calc. ....'.format(Prog_msg))

    return mockgal_pd_new

## ---------| Writing to Files |------------##

def writing_to_output_file(mockgal_pd, mockgroup_pd, zz_mock, 
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

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    perf_catl: boolean, optional (default = False)
        if 'true', it saves the `perfect` version of the galaxy / group 
        catalogue.

    """
    ## Keys
    gal_key   = '/gal_catl'
    group_key = '/group_catl'
    ## Filenames
    if perf_catl:
        ## Perfect Galaxy catalogue
        gal_file = os.path.join(proj_dict['mock_cat_mc_perf'],
                                '{0}_cat_{1}_{2}_memb_cat_perf.{3}'.format(
                                    param_dict['survey'], zz_mock, 
                                    param_dict['cosmo_choice'], output_fmt))
        ## Perfect Group catalogue
        group_file = os.path.join(proj_dict['mock_cat_gc_perf'],
                                '{0}_cat_{1}_{2}_group_cat_perf.{3}'.format(
                                    param_dict['survey'], zz_mock,
                                    param_dict['cosmo_choice'], output_fmt))
    else:
        ## Normal galaxy catalogue
        gal_file = os.path.join(proj_dict['mock_cat_mc'],
                                '{0}_cat_{1}_{2}_memb_cat.{3}'.format(
                                    param_dict['survey'], zz_mock,
                                    param_dict['cosmo_choice'], output_fmt))
        ## Normal group catalogue
        group_file = os.path.join(proj_dict['mock_cat_gc'],
                                '{0}_cat_{1}_{2}_group_cat.{3}'.format(
                                    param_dict['survey'], zz_mock,
                                    param_dict['cosmo_choice'], output_fmt))
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

def mockcatls_simbox_plot(param_dict, proj_dict, catl_ext='.hdf5',
    fig_fmt='pdf', figsize=(9,9)):
    """
    Plots the distribution of the mock catalogues in the simulation box

    Parameters
    ------------
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
    catl_path_arr = cfutils.Index(proj_dict['mock_cat_mc'], catl_ext)
    n_catls       = len(catl_path_arr)
    ## Filename
    fname = os.path.join(   proj_dict['fig_dir'],
                            '{0}_{1}_{2}_xyz_mocks.{3}'.format(
                                param_dict['survey'],
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

def mocks_lum_function(param_dict, proj_dict, catl_ext='.hdf5',
    fig_fmt='pdf', figsize=(9,9)):
    """
    Computes the luminosity function of the mock catalogues

    Parameters
    ------------
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
    catl_path_arr = cfutils.Index(proj_dict['mock_cat_mc'], catl_ext)
    n_catls       = len(catl_path_arr)
    ## Filename
    fname = os.path.join(   proj_dict['fig_dir'],
                            '{0}_{1}_{2}_lum_function_mocks.{3}'.format(
                                param_dict['survey'],
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
        mr_bins = cstats.Bins_array_create(catl_kk_pd['M_r'], base=mr_bins_sep)
        N_lum   = [num.where(catl_kk_pd['M_r'] < xx)[0].size+1 for xx in mr_bins]
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

## ---------| Multiprocessing |------------##

def multiprocessing_catls(memb_tuples_ii, pos_coords_mocks, param_dict, 
    proj_dict, ii_mock):
    """
    Distributes the analyis of the creation of mock catalogues into 
    more than 1 processor

    Parameters
    -----------
    memb_tuples_ii: tuple
        tuple of catalogue indices to be analyzed

    pos_coords_mocks_ii: tuple, shape (4,)
        tuple with the positons coordinates, coordinate dictionary, 
        and DataFrame to be used

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    ii_mock: int
        number of the mock catalogue being analyzed

    Returns
    -----------
    """
    ## Program Message
    Prog_msg = param_dict['Prog_msg']
    ## Reading which catalogues to process
    start_ii, end_ii = memb_tuples_ii
    ##
    ## Looping over the desired catalogues
    for zz in range(start_ii, end_ii):
        ## Making z'th catalogue
        catl_create_main(zz, pos_coords_mocks[zz], param_dict, proj_dict)

## -----------| Main functions |----------- ##

def main(args):
    """
    Creates set of mock catalogues for ECO, Resolve A, and Resolve B surveys.
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ## Adding additonal parameters
    param_dict = add_to_dict(param_dict)
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ##
    ## Cosmological model and Halo mass function
    param_dict = cosmo_create(param_dict)
    ## Survey Details
    param_dict = survey_specs(param_dict)
    ## Halo mass function
    param_dict = hmf_calc(param_dict['cosmo_model'], proj_dict, param_dict,
        Mmin=6., Mmax=16.01, dlog10m=1.e-3, hmf_model=param_dict['hmf_model'])
    ##
    ## Downloading files
    param_dict = download_files(param_dict, proj_dict)
    ##
    ## Redshift and Comoving distance
    param_dict = z_comoving_calc(param_dict, proj_dict)
    ## Halobias Extras file - Modified Halobias file
    param_dict = hb_file_construction_extras(param_dict, proj_dict)
    ## Checking if final version of file exists
    param_dict = clf_galprop_test(param_dict, proj_dict)
    if not param_dict['clf_opt']:
        ## Conditional Luminosity Function
        clf_pd = clf_assignment(param_dict, proj_dict)
        ## Distance from Satellites to Centrals
        clf_pd = cen_sat_distance_calc(clf_pd, param_dict)
        ## Finding closest magnitude value from ECO catalogue
        clf_pd = mr_survey_matching(clf_pd, param_dict, proj_dict)
    else:
        clf_pd = param_dict['clf_pd']
    ## Carving out geometry of Survey and carrying out the analysis
    if param_dict['survey'] == 'ECO':
        eco_geometry_mocks(clf_pd, param_dict, proj_dict)
    elif param_dict['survey'] == 'A':
        resolve_a_geometry_mocks(clf_pd, param_dict, proj_dict)
    ## Plotting different catalogues in simulation box
    mockcatls_simbox_plot(param_dict, proj_dict)
    ## Luminosity function for each catalogue
    mocks_lum_function(param_dict, proj_dict)
    ##
    ## Saving everything to TARBALL
    tarball_create(param_dict, proj_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
