#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-10-25
# Last Modified: 2018-10-25
# Vanderbilt University
from __future__ import absolute_import, division, print_function
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
__all__        = ["ReadSurvey"]
"""
Utilities for creating CAM ECO Mock catalogues
"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.ml    import ml_utils        as cmlu
from cosmo_utils.mock_catalogues import catls_utils as cmcu
from cosmo_utils.utils import gen_utils    as cgu

import numpy as num
import os
import pandas as pd
import pickle

import astropy.cosmology as astrocosmo
import astropy.constants as ac
import astropy.units     as u
import astropy.table     as astro_table
import hmf

# Functions

class ReadSurvey(object):
    """
    Reads in the multiple outputs of the ML data preprocessing and
    analysis steps for this project. This class is mainly for handling
    the aspects of reading/writing output files for this project.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        -----------
        
        """
        super().__init__()
        # Assigning variables
        self.catl_type    = kwargs.get('catl_type'   , 'mr')
        self.survey       = kwargs.get('survey'      , 'ECO')
        self.hmf_model    = kwargs.get('hmf_model'   , 'warren')
        self.halotype     = kwargs.get('halotype'    , 'fof')
        self.cpu_frac     = kwargs.get('cpu_frac'    , 0.75)
        self.remove_files = kwargs.get('remove_files', False)
        self.cosmo_choice = kwargs.get('cosmo_choice', 'Planck')
        self.clf_type     = kwargs.get('clf_type'    , 2)
        self.zspace       = kwargs.get('zspace'      , 2)
        self.nmin         = kwargs.get('nmin'        , 1)
        self.seed         = kwargs.get('seed'        , 1)
        self.l_perp       = kwargs.get('l_perp'      , 0.07)
        self.l_para       = kwargs.get('l_para'      , 1.1)
        self.verbose      = kwargs.get('verbose'     , False)
        #
        # Extra variables
        self.zmed_val     = 'zmed_val'
        self.znow         = 0
        self.cens         = 1
        self.sats         = 0
        self.proj_dict    = cwpaths.cookiecutter_paths(__file__)

    ## Prefix path based on initial parameters - Mock catalogues
    def catl_file_prefix(self, hb_name, catl_numer):
        """
        Prefix name for the catalogues

        Parameters
        -----------
        hb_name : `str`
            Name prefix of the file being analyzed. This is the name of the
            file that has the information of the galaxies.

        catl_number : `str`
            Number of the catalogue being produced.

        Returns
        --------
        catl_prefix : `str`
            Prefix name for the different catalogues
        """
        # Prefix name
        catl_prefix = '{0}_catl_{1}_{2}'.format(self.survey,
                                                self.halotype,
                                                self.hmf_model)

        return catl_prefix

    ## Output folder for the catalogues
    def catl_output_dir(self, hb_name, catl_kind='memb', perf=False,
        check_exist=False, create_dir=False):
        """
        Output directory for the catalogues

        Parameters
        ------------
        hb_name : `str`
            Name prefix of the file being analyzed. This is the name of the
            file that has the information of the galaxies.

        catl_kind : {'gal', 'memb', 'group'}, `str`
            Option for which kind of catalogue is being analyzed. This
            variable is set to ``memb`` by default.
            Options:
                - `gal` : Galaxy catalogue
                - `memb` : Group Member galaxy catalogue
                - `group` : Group galaxy catalogue

        perf : `bool`, optional
            If True, it returns (creates) the directory for the ``perfect``
            catalogue. This variable is set to `False` by default.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ---------
        catl_output_dir : `str`
            Output directory for all the catalogues
        """
        # Output file
        calt_kind_arr = ['gal', 'memb', 'group']
        if not (catl_kind in calt_kind_arr):
            msg = '>>> `catl_kind` ({1}) is not a valid input! Exiting!'
            msg = msg.format(catl_kind)
            raise TypeError(msg)
        ##
        ## Output directory
        catl_outdir = os.path.join( self.proj_dict['proc_dir'],
                                    self.cosmo_choice,
                                    self.survey,
                                    hb_name)
        ##
        ## Type of Output directory
        catl_out_dict = {   'gal'  : 'galaxy_catalogues',
                            'memb' : 'member_galaxy_catalogues',
                            'group': 'group_galaxy_catalogues'}
        if perf:
            catl_output_dir = os.path.join(    catl_outdir,
                                                'perf_{0}'.format(
                                                    catl_out_dict[catl_kind]))
        else:
            catl_output_dir = os.path.join(    catl_outdir,
                                                catl_out_dict[catl_kind])
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(catl_output_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(catl_output_dir)):
                msg = '`catl_output_dir` ({0}) was not found!'.format(
                    catl_output_dir)
                raise FileNotFoundError(msg)

        return catl_output_dir

    ## Directory at which the catalogues are saved
    def halobias_outdir(self, check_exist=False, create_dir=False):
        """
        Output directory for the halobias files with the
        input galaxy information.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ---------
        hb_files_out : `str`
            Path to the output directory for the halobias files with the
            input galaxy information.
        """
        # Halobias file output directory
        hb_files_out = os.path.join(self.proj_dict['raw_dir'],
                                    'hb_files',
                                    self.survey)
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(hb_files_out)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(hb_files_out)):
                msg = '`hb_files_out` ({0}) was not found!'.format(
                    hb_files_out)
                raise FileNotFoundError(msg)

        return hb_files_out

    ## List of files of the mock halobias files
    def halobias_files_list(self, check_exist=False, ext='.hdf5'):
        """
        Lists the files within the output directory of the
        halobias mock galaxy files.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        ext : `str`, optional
            File extension of the files to look for. This variable is set
            to '.hdf5' by default.

        Returns
        ---------
        hb_files_arr : `numpy.ndarray`, shape (N,)
            Array of the files in the `halobias` directory that match
            the file extension `ext`.
        """
        # Halobias directory
        hb_files_out = self.halobias_outdir(check_exist=True, create_dir=False)
        # List of files with given file extension
        hb_files_arr = cfutils.Index(hb_files_out, ext)

        return hb_files_arr

    ## Dictionary of halobias and file names
    def halobias_files_dict(self, check_exist=False, ext='.hdf5'):
        """
        Creates a dictionary for each of the halobias filenames and their
        basenames.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        ext : `str`, optional
            File extension of the files to look for. This variable is set
            to '.hdf5' by default.

        Returns
        ---------
        hb_files_dict : `dict`
            Dictionary of the halobias files and their corresponding
            basenames, that match the file extension `ext`.
        """
        # List of files
        hb_files_arr = self.halobias_files_list(check_exist=check_exist,
            ext=ext)
        # Creating dictionary
        hb_files_dict = {os.path.splitext(os.path.basename(xx))[0]: xx for
                            xx in hb_files_arr}

        return hb_files_dict

    ## Function for the creation of the cosmological model
    def cosmo_create(self, H0=100., Om0=0.25, Ob0=0.04, Tcmb0=2.7255,
        return_params=False, return_hmf=False):
        """
        Creates instance of the cosmology used throughout this project.

        Parameters
        ----------

        H0: float, optional (default = 100.)
            value of the Hubble constant.
            Units: km/s

        Om0: float, optional (default = 0.25)
            - value of `Omega Matter at z=0`
            - Unitless
            - Range [0,1]

        Ob0: float, optional (default = 0.04)
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
        cosmo_model : `astropy.cosmology.core.FlatLambdaCDM`
            Cosmological model object used in this analysis.

        cosmo_params : `dict`, optional
            Dictionary containing the cosmological parameters used
            throughout this analysis. This object is only returned
            when ``return_params == True``.

        cosmo_hmf : `hmf.cosmo.Cosmology`, optional
            Halo Mass Function (HMF) object with masses and densities
            used throughout this project. This object is only returned
            when ``return_hmf == True``.
        """
        # Choosing cosmology
        if (self.cosmo_choice == 'Planck'):
            cosmo_model = astrocosmo.Planck15.clone(H0=H0)
        elif (self.cosmo_choice == 'LasDamas'):
            cosmo_model = astrocosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, 
                Tcmb0=Tcmb0)
        ##
        ## Cosmological parameters
        cosmo_params         = {}
        cosmo_params['H0'  ] = cosmo_model.H0.value
        cosmo_params['Om0' ] = cosmo_model.Om0
        cosmo_params['Ob0' ] = cosmo_model.Ob0
        cosmo_params['Ode0'] = cosmo_model.Ode0
        cosmo_params['Ok0' ] = cosmo_model.Ok0
        ##
        ## HMF Cosmological model
        cosmo_hmf = hmf.cosmo.Cosmology(cosmo_model=cosmo_model)
        ##
        return_list = []
        if (return_params == False) and (return_hmf == False):
            return cosmo_model
        else:
            # Adding to the list
            return_list.append(cosmo_model)
            # Cosmological parameters
            if return_params:
                return_list.append(cosmo_params)
            # HMF parameters
            if return_hmf:
                return_list.append(cosmo_hmf)

            return return_list

    ## Directory of the mass function output directory
    def mass_func_output_dir(self, check_exist=False, create_dir=False):
        """
        Output directory for the files with the info on the mass function
        used in this project.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ---------
        mf_output_dir : `str`
            Path to the output directory for the files related to the
            mass function used in this project.
        """
        # Mass Function - Output directory
        mf_output_dir = os.path.join(   self.proj_dict['int_dir'],
                                        'MF',
                                        self.cosmo_choice,
                                        self.survey)
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(mf_output_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(mf_output_dir)):
                msg = '`mf_output_dir` ({0}) was not found!'.format(
                    mf_output_dir)
                raise FileNotFoundError(msg)

        return mf_output_dir

    ## Halo Mass Function - Calculations
    def hmf_calc(self, Mmin=10, Mmax=16, dlog10m=1e-3, ext='csv', sep=',',
        return_path=False, print_path=False):
        """
        Creates file with the desired mass function, and extract the
        information with the MF.

        Parameters
        -----------
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

        return_path : `bool`, optional
            If True, it returns the path of the HMF file. This variable is
            set to ``False`` by default.

        print_path : `bool`
            If True, it prints out the path of the HMF file. This variable
            is set to ``False`` by default.

        Return
        --------
        hfm_pd : `pandas.DataFrame`
            DataFrame containing the ``log10 masses`` and ``cumulative number
            densities`` for haloes of mass larger than `M`.

        hmf_outfile : `str`, optional
            If True, it returns the path to the HMF file used in this analysis.
            This variable is set to ``False`` by default.
        """
        # Cosmological model
        cosmo_model = self.cosmo_create()
        # HMF Output file
        hmf_outfile = os.path.join( self.mass_func_output_dir(create_dir=True),
                                    '{0}_H0_{1}_HMF_{2}.{3}'.format(
                                        self.cosmo_choice,
                                        cosmo_model.H0.value,
                                        self.hmf_model,
                                        ext))
        # Removing file if necessary
        if os.path.exists(hmf_outfile):
            os.remove(hmf_outfile)
        # Halo mass function - Fitting function
        if (self.hmf_model == 'warren'):
            hmf_choice_fit = hmf.fitting_functions.Warren
        if (self.hmf_model == 'tinker08'):
            hmf_choice_fit = hmf.fitting_functions.Tinker08
        #
        # Calculating HMF
        mf_func = hmf.MassFunction(Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m,
            cosmo_model=cosmo_model, hmf_model=hmf_choice_fit)
        # HMF - Pandas DataFrame
        mf_pd = pd.DataFrame({  'logM': num.log10(mf_func.m),
                                'ngtm': mf_func.ngtm})
        # Saving to output file
        mf_pd.to_csv(hmf_outfile, sep=sep, index=False,
            columns=['logM','ngtm'])
        # Output path
        if print_path:
            print('>> HMF File: {0}'.format(hmf_outfile))

        if not return_path:
            return mf_pd
        else:
            return mf_pd, hmf_outfile

    ## Comoving Distance calculations
    def comoving_z_distance(self, zmin=0, zmax=0.5, dz=1e-3, ext='csv',
        sep=',', print_filepath=False):
        """
        Computes the comoving distance of an object based on its
        redshift ``z``.

        Parameters
        -----------
        zmin : `float`, optional
            Minimum redshift used for the calculations. This variable is set
            to ``0`` by default.

        zmax : `float`, optional
            Maximum redshift used for the calculation. This variable is set
            to ``0.5`` by default.

        dz : `float, optional
            Step-size/increments in redshift. This variable is set to
            ``0.001`` by default.

        ext : {'csv', 'txt'}, optional
            File extension for the output file. This variable is set to
            `'csv`` by default.

        sep : {',', ' '} `str`, optional
            Delimiter used for the output file. This variable is set to
            ``,`` by default.

        print_filepath : `bool`, optional
            If True, it prints out the path of the output file. This variable
            is set to ``False`` by default.

        Returns
        -----------
        z_dc_pd : `pandas.DataFrame`
            DataFrame containing the interpolated values of comoving
            distances ``d_como`` along with the redshifts ``z`` values.
        """
        ## Checking input parameters
        # File extension - Value
        ext_arr = ['csv', 'txt']
        if not (ext in ext_arr):
            msg = '`ext` ({0}) is not a valid input value!'.format(ext)
            raise ValueError(msg)
        # File extension - Type
        if not isinstance(ext, str):
            msg = '`ext` ({0}) is not a valid input type!'.format(type(ext))
            raise TypeError(msg)
        # Delimiter extension - Value
        sep_arr = [',', ' ']
        if not (sep in sep_arr):
            msg = '`sep` ({0}) is not a valid input value!'.format(sep)
            raise ValueError(msg)
        # Delimiter extension - Type
        if not isinstance(sep, str):
            msg = '`sep` ({0}) is not a valid input type!'.format(type(sep))
            raise TypeError(msg)
        ##
        ## Verbose
        if self.verbose:
            print('>>> Comoving Distance Table Calculations ...')
        ##
        ## Cosmological model and cosmological parameters
        (   cosmo_model,
            cosmo_params) = self.cosmo_create(return_params=True)
        # Cosmological directory
        cosmo_dir = self.cosmo_outdir(create_dir=True)
        # Cosmology file
        z_dc_file = os.path.join(  cosmo_dir,
                                    '{0}_H0_{1}_z_dc.{2}'.format(
                                        self.cosmo_choice,
                                        cosmo_params['H0'],
                                        ext))
        # Checking if file exists
        if os.path.exists(z_dc_file):
            if self.remove_files:
                os.remove(z_dc_file)
                create_opt = True
            else:
                create_opt = False
        else:
            create_opt = True
        ##
        ## Creating file if needed
        if create_opt:
            ## Calculating comoving distance
            ## `z_arr` : Unitless
            ## `z_dc`  : Mpc
            z_arr = num.arange(zmin, zmax, dz)
            z_dc  = cosmo_model.comoving_distance(z_arr).to(u.Mpc).value
            z_dc_pd = pd.DataFrame({'z': z_arr, 'dc': z_dc})
            # Saving to file
            z_dc_pd.to_csv(z_dc_file, sep=sep, index=False)
            cfutils.File_Exists(z_dc_file)
        else:
            z_dc_pd = pd.read_csv(z_dc_file, sep=sep)
        #
        # File path
        if print_filepath:
            print('`z_dc_file`: {0}'.format(z_dc_file))

        return z_dc_pd

    ## Figure Output directory
    def fig_outdir(self, hb_name, check_exist=False, create_dir=False):
        """
        Output directory for the figures produced in this project.

        Parameters
        -----------
        hb_name : `str`
            Name prefix of the file being analyzed. This is the name of the
            file that has the information of the galaxies.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ---------
        fig_output_dir : `str`
            Path to the output directory for the files related to the
            mass function used in this project.
        """
        # Figure directory - Output
        fig_output_dir = os.path.join(  self.proj_dict['plot_dir'],
                                        self.cosmo_choice,
                                        self.halotype,
                                        self.survey,
                                        hb_name)
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(fig_output_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(fig_output_dir)):
                msg = '`fig_output_dir` ({0}) was not found!'.format(
                    fig_output_dir)
                raise FileNotFoundError(msg)

        return fig_output_dir

    ## Location of the TAR folder
    def tar_outdir(self, check_exist=False, create_dir=False):
        """
        Output directory for the TAR file produced in this project.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the directory exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ---------
        tar_output_dir : `str`
            Path to the TAR output directory, which includes the resuls for
            each of the catalogues.
        """
        # Figure directory - Output
        tar_output_dir = os.path.join(  self.proj_dict['proc_dir'],
                                        'TAR_files',
                                        self.cosmo_choice,
                                        self.halotype,
                                        self.survey)
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(tar_output_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(tar_output_dir)):
                msg = '`tar_output_dir` ({0}) was not found!'.format(
                    tar_output_dir)
                raise FileNotFoundError(msg)

        return tar_output_dir

    ## Output TAR file path
    def tar_output_file(self, hb_name, check_exist=False):
        """
        Path to the output TAR file, which includes the figures and catalogues
        for each catalogue.

        Parameters
        -----------
        hb_name : `str`
            Name prefix of the file being analyzed. This is the name of the
            file that has the information of the galaxies.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the directory exists.
            This variable is set to `False` by default.

        Returns
        -----------
        tar_outpath : `str`
            Path to the output TAR file, which includes the figures and
            catalogues for each catalogue.
        """
        # Path to the Output TAR directory
        tar_output_dir = self.tar_outdir(create_dir=True)
        # Path to the output TAR file
        tar_outpath = os.path.join( tar_output_dir,
                                    '{0}_{1}_catls.tar.gz'.format(
                                        hb_name,
                                        self.survey))
        # Check for its existence
        if check_exist:
            if not (os.path.exists(tar_outpath)):
                msg = '`tar_outpath` ({0}) was not found!'.format(
                    tar_outpath)
                raise FileNotFoundError(msg)

        return tar_outpath

    ## Location of the TAR folder
    def cosmo_outdir(self, check_exist=False, create_dir=False):
        """
        Output directory for cosmology-related files.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the directory exists.
            This variable is set to `False` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        ---------
        cosmo_dir : `str`
            Path to the directory that contains cosmologica-related
            files.
        """
        # Figure directory - Output
        cosmo_dir = os.path.join(  self.proj_dict['raw_dir'],
                                        self.cosmo_choice,
                                        'cosmo_dir')
        # Creating directory
        if create_dir:
            cfutils.Path_Folder(cosmo_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(cosmo_dir)):
                msg = '`cosmo_dir` ({0}) was not found!'.format(
                    cosmo_dir)
                raise FileNotFoundError(msg)

        return cosmo_dir

    ## Geometrical components of catalogues
    def geometry_calc(self, dist_1, dist_2, alpha):
        """
        Computes the geometrical components to construct the catalogues
        in a simulation box

        Parameters
        -----------
        dist_1 : `float`
            First distance used to determine geometrical components.

        dist_2 : `float`
            Second distance used to determine geometrical components.

        alpha : `float`
            Angle used to determine geometrical components. This variable is
            in units of ``degrees``.

        Returns
        -----------
        h_total : `float`

        h1 : `float`

        s1 : `float`

        s2 : `float`
        """
        assert(dist_1 <= dist_2)
        # Calculating distances for the triangles
        s1 = self.cos_rule(dist_1, dist_1, alpha)
        s2 = self.cos_rule(dist_2, dist_2, alpha)
        # Height
        h1        = (dist_1**2 - (s1 * 0.5)**2.)**0.5
        assert(h1 <= dist_1)
        h2        = dist_1 - h1
        h_total   = h2 + (dist_2 - dist_1)

        return h_total, h1, s1, s2

    # Cosine rule
    def cos_rule(self, a, b, gamma):
        """
        Computes the `cosine rule` for 2 distances and 1 angle.

        Parameters
        ------------
        a : `float`
            One of the sides ofm the triangle

        b : `float`
            Second side of the triangle

        gamma : `float`
            Angle facing the side of the triangle in question, This variable
            is in units of ``degrees``.

        Returns
        ------------
        c : `float`
            Third side of the triangle.
        """
        # Degrees to radians
        gamma_rad = num.radians(gamma)
        # Third side of the triangle
        c = (a**2 + b**2 - (2*a*b*num.cos(gamma_rad)))**0.5

        return c

    # Survey Volume
    def survey_vol_calc(self, ra_arr, dec_arr, rho_arr):
        """
        Computes the volume of a ``sphere`` given limits for `ra`, `dec`,
        and `distance`.

        Parameters
        -----------
        ra_arr : `numpy.ndarray`, shape (N,2)
            Array with initial and final right ascension coordinates.
            This variable is in units of ``degrees`.

        dec_arr : `numpy.ndarray`, shape (N,2)
            Array with initial and final declination coordinates.
            This variable is in units of ``degrees``.

        rho_arr : `numpy.ndarray`, shape (N,2)
            Array with initial and final distances. This variable is in
            units of ``distance units``.

        Returns
        ---------
        survey_vol : `float`
            Volume of the galaxy survey based on ra, dec, and distance.
            This variable is in units of ``distance units ^3``.
        """
        # Right ascension - Radians - Theta coordinates
        theta_min_rad, theta_max_rad = num.radians(num.array(ra_arr))
        # Declination - Radians - Phi coordinates
        phi_min_rad, phi_max_rad = num.radians(90. - num.array(dec_arr))[::-1]
        # Distance
        rho_min, rho_max = num.array(rho_arr)
        # Calculating the galaxy survey volume
        survey_vol  = (1./3.) * (num.cos(phi_min_rad) - num.cos(phi_max_rad))
        survey_vol *= (theta_max_rad) - (theta_min_rad)
        survey_vol *= (rho_max**3) - (rho_min**3)
        survey_vol  = num.abs(survey_vol)

        return survey_vol


















    