#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import interp1d
from scipy.signal import medfilt, medfilt2d
import pandas as pd
import time
from scipy.interpolate import UnivariateSpline
from datetime import datetime
from matplotlib.dates import date2num, DateFormatter
import matplotlib.animation as animation

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Libraries for plotting, reading data:
import seaborn as sns 
sns.set_style("ticks") # Set seaborn "ticks" style for better styled plots
from astropy.io import fits
from astropy.utils.data import download_file
# Library for some power-spectral density analysis:
from astropy.timeseries import LombScargle

# Corner (for posterior distribution plotting):
import corner
# Juliet (for transit fitting & model evaluation:)
import juliet
#plt.style.use('dark_background')

from barycorrpy import utc_tdb
from astropy.time import Time
from transitspectroscopy import spectroscopy

import batman
import starry
import lmfit
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from scipy.optimize import curve_fit

import scipy
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from astropy.utils.data import get_pkg_data_filename, download_file
from astropy.table import Table, Column, MaskedColumn
from astropy.io import fits, ascii
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
import astropy.units as u
from scipy.interpolate import interp1d, splev, splrep
import scipy.optimize as opt
from scipy.io import readsav
from scipy import stats
from scipy.signal import savgol_filter
import scipy.signal as signal
import glob
import lmfit
import pickle
from os import path,mkdir
from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm
import warnings
import pandas as pd
import os
import shutil
import numba

get_ipython().run_line_magic('matplotlib', 'inline')



# function that opens each orbit fits file and gets the science exposures
# optional kwargs to also get the dq extensions and jit vectors, for later use in cleaning/detrending the data
# ADD: get rid of one second exposures
# ADD: automatically throws out the first exposure of each orbit, option to also throw out the first orbit
# ADD: error message if jit file doesn't exist
def get_data(files, dq = True, jit = True, keep_first_orbit = True):
    
    # initializing some lists to hold the data
    data_hold = []
    headers_hold = []
    jitter_hold = []
    dqs_hold = []
    
    # keeping track of how many exposures in each orbit
    visits_idx = []
    
    
    for fits_file in files:
        hdulist = fits.open(fits_file)
        sciextlist = []
        dqextlist = []

        # finding which fits extensions correspond to the science and dq frames, if applicable
        for i in range(len(hdulist.info(0))):
            if hdulist.info(0)[i][1] == "SCI":
                sciextlist.append(i)
            if dq == True:
                if hdulist.info(0)[i][1] == "DQ":
                    dqextlist.append(i)
        
        # getting the data and header for each of the science frames
        data_lst = []
        header_lst = []
        for j in sciextlist:
            data, header = fits.getdata(fits_file, ext = j, header=True)
            data_lst.append(data)
            header_lst.append(header)

        # getting the data quality frames
        if dq == True:
            dq_lst = []
            for k in dqextlist:
                dqs = fits.getdata(fits_file, ext = k, header = False)
                dq_lst.append(dqs)
                
        if jit == True:
            # gets corresponding .jit file for each .fits file
            corresponding_jit_file = fits.open(fits_file.replace("flt","jit")) 
            
            # gets the names of the different jitter vectors
            jitter_vector_list = corresponding_jit_file[1].columns.names 
            
            # initialize an intermediate list
            jitter_lst = []

            for jitter_hdu in corresponding_jit_file: # iterates through each exposure of each file
                if jitter_hdu.name == 'jit':
                    dummy_jit_array = []
                    for jitvect in jitter_vector_list: # iterates through each jitter vector name
                        jitter_points = jitter_hdu.data[jitvect]
                        jitter_points[jitter_points > 1e30] = np.median(jitter_points) # kills weird edge cases
                        dummy_jit_array.append(np.mean(jitter_points)) # saves the mean jitter value inside of each exposure
                    jitter_lst.append(dummy_jit_array)
        
        
        #i'm not returning this at the moment but can if they're needed
        visits_idx.append(len(data_lst)) 
        
        # adding each orbit's data to master list
        data_hold = data_hold + data_lst
        headers_hold = headers_hold + header_lst
        
        if dq == True:
            dqs_hold = dqs_hold + dq_lst
            
        if jit == True:
            jitter_hold = jitter_hold + jitter_lst
        
    if jit == True:
        # sort jitter vectors
        jitter_dict = {}
        for i in range(len(jitter_vector_list)):
            jitter_dict[jitter_vector_list[i]] = [item[i] for item in jitter_hold]
            # NORMALIZE JITTER VECTORS HERE
        
        
    
    if dq == True and jit == True:
        return data_hold, headers_hold, jitter_dict, dqs_hold
    elif dq == True and jit == False:
        return data_hold, headers_hold, dqs_hold
    elif dq == False and jit == True:
        return data_hold, headers_hold, jitter_dict
    else:
        return data_hold, headers_hold



# using the best medfilt window size of 5, find the points for which the residual value is 
# greater than 5 sigma and mark them - 1d
def residual_outliers_1d(spectra_cut, n = 5): 
    medfilt_result = medfilt(spectra_cut, 5)
    
    residuals = medfilt_result-spectra_cut
    stdev_residuals = 1.4826*np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    #stdev_residuals = np.sqrt(np.var(residuals))
    outlier_locations = np.where(abs(residuals)>stdev_residuals*n)
    
    # option to plot positions of the outliers 
    #plt.figure(figsize=(15,10))
    #plt.imshow(spectra_cut)
    #plt.scatter(outlier_locations[1],outlier_locations[0], color = "red")
    #plt.show()
    
    return outlier_locations

# function to do a basic centroid trace fit to the data - should look into doing this with moffat instead
def trace_spectrum(data, xi, xf, y_guess, profile_radius = 20, gauss_filter_width = 10):
    # x-axis
    x = np.arange(xi,xf)
    # y-axis
    y = np.arange(data.shape[0])
    
    # Define array that will save centroids at each x:
    y_vals = np.zeros(len(x))
    
    for i in range(len(x)):
        # Find centroid within profile_radius pixels of the initial guess:
        idx = np.where(np.abs(y-y_guess)<profile_radius)[0]
        y_vals[i] = np.nansum(y[idx]*data[:,x[i]][idx])/np.nansum(data[:,x[i]][idx])
        y_guess = y_vals[i]
    return x,y_vals


# function to mark bad pixels in files based on data quality (dq) frames included in the fits files
# default flagged pixel to use is 16, but can add a list of whichever you want to remove
def dq_clean(files, dqs, flags = [16]):
    
    # initializing list of bad pixels
    bads = []
    
    # in each of the included files, mark the location of the pixels with the value given in the flags kwarg
    for i in range(len(files)):
        for j in flags:
            bad = np.where(dqs[i] == j)
            bad_indices = list(zip(bad[0], bad[1]))
            bads.append(bad_indices)
    bads = np.array(bads)
    
    # make a copy of the original fed in files array shape
    files_clean = np.zeros_like(files)
    
    # make the value of each bad pixel -1, to be fit over later
    for j in range(len(files)):
        c_frame = files[j]
        for index in bads[j]:
            c_frame[index] = -1
        files_clean[j] = c_frame
    
    # return the files with the bad pixel locations set as -1
    return files_clean


def difference_clean(files, difference_sigma, wind_size, wind_sigma):
    
    # initializing lists
    differences = []
    labels = []
    
    # each image will be taken with a difference with the rest of the images in files
    # i know it's redundant to take differences both ways, but i think it makes sense for the median files
    for i in range(len(files)):
        standard = files[i]
        for j in range(len(files)):
            if j == i:
                continue
            else:
                diff = standard - files[j]
                differences.append(diff)
                label = [i,j]
                labels.append(label)
    
    medians = []
    
    # create median frames using all of the subtractions from each frame together
    for k in range(len(files)):
        sublist = []
        for n in range(len(labels)):
            if labels[n][0] == k:
                sublist.append(differences[n])
            else:
                continue
            
        median_frame = np.nanmedian(sublist, axis = 0)   

        medians.append(median_frame)
        
    # in each median frame, in each row sigma reject in windows given by wind_size and wind_sigma
    cr_loc = []
    cr_loc_frame = []
    files_clean = np.zeros_like(files)
    frame_num = 0
    for m in medians:
        frame = np.copy(files[frame_num])
        cr_loc_single = []
        cr_loc_frame_single = np.zeros_like(medians[0])
        for row_idx in range(len(m)):
            row = m[row_idx]
            row_med = np.nanmedian(row)
            row_stdev = np.sqrt(np.var(row))
            row_cut = row_med+(row_stdev*sigma)
            q = wind_size
            for p in range(len(row)-wind_size):
                window = row[p:q]
                wind_med = np.nanmedian(window)
                wind_stdev = np.sqrt(np.var(m))
                wind_cut = wind_med+(wind_stdev*sigma)
                cut_use = max(wind_cut, row_cut)
                for val in range(len(window)):
                    if window[val] > cut_use:
                        cr_loc_single.append([m,p+val])
                        cr_loc_frame_single[row_idx][p+val] = 1
                        frame[row_idx][p+val] = -2
                p = p+wind_size
                q = q+wind_size
            
        cr_loc.append(cr_loc_single)
        cr_loc_frame.append(cr_loc_frame_single)
        files_clean[frame_num] = frame 
        frame_num = frame_num + 1
    
    return files_clean



def hc_clean(files, hc_sigma, hc_wind_size):
    
    # initializing list
    hcs = []
    
    splines_hcs = np.zeros_like(files)
    files_clean = np.zeros_like(files)
    
    # hot and cold pixels
    for frame_idx in range(len(files)):
        frame = np.copy(files[frame_idx])
        for column in range(len(files[0][0])):
            test_spline = UnivariateSpline(np.arange(0,len(frame[:,column]),1), frame[:,column], s=100)
            splines_hcs[frame_idx][:,column] = (test_spline(np.arange(0,len(frame[:,column]),1)))

        # leave borders around the edges to not mess up the box, we don't really care about edges anyways
        for p in range(hc_window_size+1,len(splines_hcs[frame_idx])-hc_window_size-1,1):
            for q in range(hc_window_size+1,len(splines_hcs[frame_idx][0])-hc_window_size-1,1):
                box = splines_hcs[frame_idx][p-hc_window_size:p+hc_window_size, q-hc_window_size:q+hc_window_size]
                box_less = box[box != splines_hcs[frame_idx][p,q]]
                box_med = np.nanmedian(box_less)
                box_stdev = np.sqrt(np.var(box_less))
                
                if splines_hcs[frame_idx][p,q] < box_med - (box_stdev * sigma_hc) or splines_hcs[frame_idx][p,q] > box_med + (box_stdev * sigma_hc):
                    hcs.append([p,q])
                    frame[p][q] = -3
        
        files_clean[frame_idx] = frame
    
    return files_clean


def spline_clean(files, traces, spline_sigma):

    files_clean = np.zeros_like(files)
    # take average of spline fits across frames
    splines = np.zeros_like(files[0])
    # for each column
    for column in range(len(files[0][0])):
        # create temporary list to hold the individual splines for that column
        spline_fits = []
        for i in files:
            #test_spline = UnivariateSpline(np.arange(0,len(i[:,column]),1), i[:,column], s = 900)
            test_spline = UnivariateSpline(np.arange(0,len(i[:,column]),1), i[:,column])
            spline_fits.append(test_spline(np.arange(0,len(i[:,column]),1)))
            #plt.plot(test_spline(np.arange(0,len(i[:,column]),1)))
            #plt.show()

        # take the median of the splines from each frame
        med = np.nanmedian(spline_fits, axis = 0)
        #plt.plot(med)
        #plt.show()

        # normalize the spline before appending
        norm_med = med/np.nanmax(med)
        splines[:,column] = norm_med
    
    # now, use the splines to go through each frame and reject problems
    crs = []
    spline_use = np.zeros_like(splines)
    for frame_idx in range(len(files)):
        frame = np.copy(files[frame_idx])
        # use the median spline to reject cosmic rays for each frame's column
        #for k in range(len(frame[0])):
        # make this into groups? and then average sigma between them? no that won't work cause sigma changes too much
        #for k in range(600,800,1):
        for k in range(2,len(frame[0])-2,1):
            spline_use_single = np.nanmedian(splines[:,k-2:k+2], axis = 1)
            spline_use[:,k] = spline_use_single*np.nanmax(frame[:,k])
            #plt.plot(spline_use[:,k])
            #plt.show()
            #plt.plot((splines[:,k]*np.nanmax(frame[:,k])))
            #plt.plot((frame[:,k]))
            #plt.yscale("log")
            #plt.show()
            # scale the normalized median spline to the max of the frame before taking residual
            resid = frame[:,k] - (spline_use[:,k])
            resid_stdev = np.sqrt(np.var(resid))
            cutoff = resid_stdev * sigma_spline
            #print(cutoff)
            #plt.plot(resid)
            #plt.show()
            #j = 0
            for m in range(len(resid)):
                # if the residual value is greater than residual stdev * sigma, then mark it as a problem
                #print(traces[frame_idx][1][m])
                if m > traces[frame_idx][1][m]+3 or m < traces[frame_idx][1][m]-3:
                    #print(m)
                    if resid[m] > cutoff:
                        #cr_frame.append([k,m])
                        #j = j+1
                        frame[m,k] = -4
                else:
                    pass
            #print(j)
            # even after averaging, normalizing the spline and then scaling that to the max value of the new column, still strong 
            # residuals around the center at 10 sigma level
                
        files_clean[frame_idx] = frame
        
    return files_clean, splines


def spline_clean(files, splines = None):
    if splines == None:
        # take average of spline fits across frames
        splines = np.zeros_like(files[0])
        # for each column
        for column in range(len(files[0][0])):
            # create temporary list to hold the individual splines for that column
            spline_fits = []
            for i in files:
                #test_spline = UnivariateSpline(np.arange(0,len(i[:,column]),1), i[:,column], s = 900)
                test_spline = UnivariateSpline(np.arange(0,len(i[:,column]),1), i[:,column])
                spline_fits.append(test_spline(np.arange(0,len(i[:,column]),1)))
                #plt.plot(test_spline(np.arange(0,len(i[:,column]),1)))
                #plt.show()

            # take the median of the splines from each frame
            med = np.nanmedian(spline_fits, axis = 0)
            #plt.plot(med)
            #plt.show()

            # normalize the spline before appending
            norm_med = med/np.nanmax(med)
            splines[:,column] = norm_med
        
    # there are columns full of bad pixels - if an entire column has more than 50% pixels marked as bad (=-1)
    # then use the average of the two surrounding columns 
    files_clean = np.zeros_like(files)
    # use the splines to correct all bad pixels (<0)
    for frame_idx in range(len(files)):
        frame = np.copy(files[frame_idx])
        #spline_use = np.zeros_like(splines)
        #for column in range(len(frame[0])-1):
        #    spline_use_c = splines[:,column]*np.nanmax(frame[:,column])
        #    spline_use[:,column] = spline_use_c
        for column in range(2,len(frame[0])-2,1):
            bad_counter = 0
            for pixel in range(len(frame)):
                if frame[pixel][column] == -1:
                    bad_counter = bad_counter + 1
            #print(bad_counter)
            if bad_counter >= len(frame)/3:
                frame[:,column] = (frame[:,column-1]+frame[:,column+1])/2
                #print("yes")
            for val in range(len(frame)):
                #print(frame.shape, splines.shape)
                if frame[val][column] in [-1,-2,-3,-4]:
                    #print("yes")
                    #print(frame[val][column])
                    #frame[val][column] = (spline_use[val][column])
                    frame[val][column] = (frame[val][column+1]+frame[val][column-1])/2
                    #print(frame[val][column])
        files_clean[frame_idx] = frame
        
    return files_clean


# how best to name the variables and still be able to feed them in? use the same name for all the arrays? seems dangerous
def clean_data(files, dq_correct = True, dqs = None, difference_correct = True, difference_sigma = 5,                wind_size = 20, wind_sigma = 5, hc_correct = True, hc_sigma = 3, hc_wind_size = 2,                spline_correct = True, traces = None, spline_sigma = 3):
    
    if dq_correct == True:
        if dqs == None:
            print("Oops! You need the data quality frames corresponding to each exposure for the dq_correct." +                   "You can get these from the get_data function with dq = True")
            return
        else:
            marked_1 = dq_clean(files, dqs)
    else:
        marked_1 = np.copy(files)
        
    if difference_correct == True:
        marked_2 = difference_clean(marked_1, difference_sigma, wind_size, wind_sigma)
    else: 
        marked_2 = np.copy(marked_1)
    
    if hc_correct == True:
        marked_3 = hc_clean(marked_2, hc_sigma, hc_wind_size)
    else:
        marked_3 = np.copy(marked_2)
        
    if spline_correct == True:
        if traces == None:
            print("Oops! You need basic spectral traces for the spline fit option :)")
            return
        else:
            marked_4, splines = spline_clean(marked_3, traces, spline_sigma)
            cleaned_data = spline_clean(marked_4, splines = splines)
    else:
        marked_4 = np.copy(marked_3)
        cleaned_data = spline_clean(marked_4)
    
    return cleaned_data



def times_to_bjd(headers):
    times = []
    exptimes = []
    expstart = []
    expend = []
    for i in headers:
        #times.append(i["DATE-OBS"]+i["TIME-OBS"])
        times.append(i["DATE-OBS"]+"T"+i["TIME-OBS"])
        #print(i["DATE-OBS"])
        exptimes.append(i["EXPTIME"])
        expstart.append(i["EXPSTART"])
        expend.append(i["EXPEND"])

    # change this time to bjd using package barycorr

    jd_conv = 2400000.5
    t_start = Time(np.array(expstart)+jd_conv, format='jd', scale='utc')
    t_end = Time(np.array(expend)+jd_conv, format='jd', scale='utc')
    #t = t.plot_date

    t_start_bjd = utc_tdb.JDUTC_to_BJDTDB(t_start,starname = "WASP-69")#hip_id=8102 , lat=-30.169283, longi=-70.806789, alt=2241.9)
    t_end_bjd = utc_tdb.JDUTC_to_BJDTDB(t_end,starname = "WASP-69")# , lat=-30.169283, longi=-70.806789, alt=2241.9)

    return t_start_bjd, t_end_bjd


def spectral_extraction(data, trace, method = "optimal", correct_bkg = False, aperture_radius = 15., ron = 1., gain = 1.,                         nsigma = 10, polynomial_spacing = 0.75, polynomial_order = 3):
    
    if method == "optimal":
        spectrum = spectroscopy.getOptimalSpectrum(data, trace, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order)#, min_column = 600)
    elif method == "simple":
        x = np.arange(len(data)) # i think? test
        spectrum = spectroscopy.getSimpleExtraction(data, x, trace, aperture_radius, correct_bkg = correct_bkg)#, min_column = 600)

    return spectrum



# cross-correlation

def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y Deviations
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    maxlag :  Lag at which cross-correlation function is maximized
    """
    corr = signal.correlate(x-np.mean(x), y-np.mean(y), mode="full")
    scx=np.sum((y-np.mean(y))**2)
    scy=np.sum((x-np.mean(x))**2)
    corr = corr/(np.sqrt(scx*scy))
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    mindx=np.argmax(corr) #index of cross corr peak
    
    srad = 3 #Find peak shift from central region srad pixels wide on each side
    sublag=lags[mindx-srad:mindx+(srad+1)]
    subcf=corr[mindx-srad:mindx+(srad+1)]
    result=np.polyfit(sublag, subcf, 2)
    maxlag = - result[1]/(2.*result[0])
    return lags, corr,maxlag


# ADD SYSTEMATICS OPTION FOR GP

def model_light_curve(p, t, params, jitters):
    """
    Generates a light curve with systematics. Uses the lmfit 'p' dictionary.
    """
    params.t0  = p['t0']
    params.per = p['per']
    params.rp  = np.sqrt(p['rp2'])
    params.a   = p['a']
    params.inc = inclination(p['b'], p['a'])
    params.ecc = p['ecc']
    params.w   = p['w']
    #params.limb_dark = "quadratic"
    #params.u = [p['c1'], p['c2']]

    if p['rp2'] == 0:
        light_curve = np.ones(len(t)) # if it doesn't want a transit, give it a flat line
    else:
        light_curve = batman.TransitModel(params, t).light_curve(params)
    
    # zaf said best detrenders were the roll ones
    systematics =  p["v2_roll"]*np.array(jitters["V2_roll"]) + p["v3_roll"]*np.array(jitters["V3_roll"]) +     p["lat"]*np.array(jitters["Latitude"]) + p["long"]*np.array(jitters["Longitude"]) +     p["RA"]*np.array(jitters["RA"]) + p["DEC"]*np.array(jitters["DEC"]) + 1 #p["offset"]
    
    #systematics = p['xs']*xs + p['ys']*ys + p['xys']*xys + p['xs2']*xs2 + p['ys2']*ys2 \
    #           + p['common_mode']*common_mode + p['lin']*tlin + 1 
    
    #systematics = exp(-t/p['exp_tmscl']) + p['jitter_amp']*telescope_pos # leaving out systematics right now
    
    #plt.plot(light_curve)
    #plt.plot(systematics)
    
    model = p['f0'] * light_curve * systematics
    light_curve_plot = batman.TransitModel(params, t_final).light_curve(params)
    plt.plot(t_final, p['f0'] * light_curve_plot)
    #plt.plot(model)
    #plt.plot(model)
    #plt.show()
    #print(model)
    return model, systematics, light_curve

    
def residual(p, t, params, data, err, jitters):#, telescope_pos, err):
    """
    Outputs the residual of the model and data.
    """
    
    model = model_light_curve(p, t, params, jitters)[0]#, telescope_pos)[0]
    sys = model_light_curve(p, t, params, jitters)[1]
    
    plt.scatter(t, data/sys, color = "purple", label = "with systematics")
    plt.legend()
    plt.show()
    plt.plot(t, model)
    plt.scatter(t, data, color = "blue", label = "Original", alpha = 0.5)
    plt.legend()
    plt.show()
    
    if err == None:
        err = np.sqrt(p['f0']) # if no errorbars specified, assume shot noise uncertainty from baseline flux

    chi2 = sum((data-model)**2/err**2)
    print(chi2)
    res = np.std((data-model)/max(model))
    
    return (data-model)/err

#Limb Darkeneing
@custom_model
def nonlinear_limb_darkening(x, c0=0.0, c1=0.0, c2=0.0, c3=0.0):
    """
    Define non-linear limb darkening model with four parameters c0, c1, c2, c3.
    """
    model = (1. - (c0 * (1. - x ** (1. / 2)) + c1 * (1. - x ** (2. / 2)) + c2 * (1. - x ** (3. / 2)) + c3 *
                   (1. - x ** (4. / 2))))
    return model


@custom_model
def quadratic_limb_darkening(x, aLD=0.0, bLD=0.0):
    """
    Define linear limb darkening model with parameters aLD and bLD.
    """
    model = 1. - aLD * (1. - x) - bLD * (1. - x) ** (4. / 2.)
    return model

def limb_dark_fit(grating, wsdata, M_H, Teff, logg, dirsen, ld_model='1D'):
    """
    Calculates stellar limb-darkening coefficients for a given wavelength bin.

    Currently supports:
    HST STIS G750L, G750M, G430L gratings
    HST WFC3 UVIS/G280, IR/G102, IR/G141 grisms

    What is used for 1D models - Kurucz (?)
    Procedure from Sing et al. (2010, A&A, 510, A21).
    Uses 3D limb darkening from Magic et al. (2015, A&A, 573, 90).
    Uses photon FLUX Sum over (lambda*dlamba).
    :param grating: string; grating to use ('G430L','G750L','G750M', 'G280', 'G102', 'G141')
    :param wsdata: array; data wavelength solution
    :param M_H: float; stellar metallicity
    :param Teff: float; stellar effective temperature (K)
    :param logg: float; stellar gravity
    :param dirsen: string; path to main limb darkening directory
    :param ld_model: string; '1D' or '3D', makes choice between limb darkening models; default is 1D
    :return: uLD: float; linear limb darkening coefficient
    aLD, bLD: float; quadratic limb darkening coefficients
    cp1, cp2, cp3, cp4: float; three-parameter limb darkening coefficients
    c1, c2, c3, c4: float; non-linear limb-darkening coefficients
    """

    print('You are using the', str(ld_model), 'limb darkening models.')

    if ld_model == '1D':

        direc = os.path.join(dirsen, 'Kurucz')

        print('Current Directories Entered:')
        print('  ' + dirsen)
        print('  ' + direc)

        # Select metallicity
        M_H_Grid = np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
        M_H_Grid_load = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 21, 22, 23, 24])
        optM = (abs(M_H - M_H_Grid)).argmin()
        MH_ind = M_H_Grid_load[optM]

        # Determine which model is to be used, by using the input metallicity M_H to figure out the file name we need
        direc = 'Kurucz'
        file_list = 'kuruczlist.sav'
        sav1 = readsav(os.path.join(dirsen, file_list))
        model = bytes.decode(sav1['li'][MH_ind])  # Convert object of type "byte" to "string"

        # Select Teff and subsequently logg
        Teff_Grid = np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500])
        optT = (abs(Teff - Teff_Grid)).argmin()

        logg_Grid = np.array([4.0, 4.5, 5.0])
        optG = (abs(logg - logg_Grid)).argmin()

        if logg_Grid[optG] == 4.0:
            Teff_Grid_load = np.array([8, 19, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 138])

        elif logg_Grid[optG] == 4.5:
            Teff_Grid_load = np.array([9, 20, 31, 42, 53, 64, 75, 86, 97, 108, 119, 129, 139])

        elif logg_Grid[optG] == 5.0:
            Teff_Grid_load = np.array([10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 120, 130, 140])

        # Where in the model file is the section for the Teff we want? Index T_ind tells us that.
        T_ind = Teff_Grid_load[optT]
        header_rows = 3    #  How many rows in each section we ignore for the data reading
        data_rows = 1221   # How  many rows of data we read
        line_skip_data = (T_ind + 1) * header_rows + T_ind * data_rows   # Calculate how many lines in the model file we need to skip in order to get to the part we need (for the Teff we want).
        line_skip_header = T_ind * (data_rows + header_rows)

        # Read the header, in case we want to have the actual Teff, logg and M_H info.
        # headerinfo is a pandas object.
        headerinfo = pd.read_csv(os.path.join(dirsen, direc, model), delim_whitespace=True, header=None,
                                 skiprows=line_skip_header, nrows=1)

        Teff_model = headerinfo[1].values[0]
        logg_model = headerinfo[3].values[0]
        MH_model = headerinfo[6].values[0]
        MH_model = float(MH_model[1:-1])

        print('\nClosest values to your inputs:')
        print('Teff: ', Teff_model)
        print('M_H: ', MH_model)
        print('log(g): ', logg_model)

        # Read the data; data is a pandas object.
        data = pd.read_csv(os.path.join(dirsen, direc, model), delim_whitespace=True, header=None,
                              skiprows=line_skip_data, nrows=data_rows)

        # Unpack the data
        ws = data[0].values * 10   # Import wavelength data
        f0 = data[1].values / (ws * ws)
        f1 = data[2].values * f0 / 100000.
        f2 = data[3].values * f0 / 100000.
        f3 = data[4].values * f0 / 100000.
        f4 = data[5].values * f0 / 100000.
        f5 = data[6].values * f0 / 100000.
        f6 = data[7].values * f0 / 100000.
        f7 = data[8].values * f0 / 100000.
        f8 = data[9].values * f0 / 100000.
        f9 = data[10].values * f0 / 100000.
        f10 = data[11].values * f0 / 100000.
        f11 = data[12].values * f0 / 100000.
        f12 = data[13].values * f0 / 100000.
        f13 = data[14].values * f0 / 100000.
        f14 = data[15].values * f0 / 100000.
        f15 = data[16].values * f0 / 100000.
        f16 = data[17].values * f0 / 100000.

        # Make single big array of them
        fcalc = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16])
        phot1 = np.zeros(fcalc.shape[0])

        # Define mu
        mu = np.array([1.000, .900, .800, .700, .600, .500, .400, .300, .250, .200, .150, .125, .100, .075, .050, .025, .010])

        # Passed on to main body of function are: ws, fcalc, phot1, mu

    elif ld_model == '3D':

        direc = os.path.join(dirsen, '3DGrid')

        print('Current Directories Entered:')
        print('  ' + dirsen)
        print('  ' + direc)

        # Select metallicity
        M_H_Grid = np.array([-3.0, -2.0, -1.0, 0.0])  # Available metallicity values in 3D models
        M_H_Grid_load = ['30', '20', '10', '00']  # The according identifiers to individual available M_H values
        optM = (abs(M_H - M_H_Grid)).argmin()  # Find index at which the closes M_H values from available values is to the input M_H.

        # Select Teff
        Teff_Grid = np.array([4000, 4500, 5000, 5500, 5777, 6000, 6500, 7000])  # Available Teff values in 3D models
        optT = (abs(Teff - Teff_Grid)).argmin()  # Find index at which the Teff values is, that is closest to input Teff.

        # Select logg, depending on Teff. If several logg possibilities are given for one Teff, pick the one that is
        # closest to user input (logg).

        if Teff_Grid[optT] == 4000:
            logg_Grid = np.array([1.5, 2.0, 2.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 4500:
            logg_Grid = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5000:
            logg_Grid = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5500:
            logg_Grid = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5777:
            logg_Grid = np.array([4.4])
            optG = 0

        elif Teff_Grid[optT] == 6000:
            logg_Grid = np.array([3.5, 4.0, 4.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 6500:
            logg_Grid = np.array([4.0, 4.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 7000:
            logg_Grid = np.array([4.5])
            optG = 0

        # Select Teff and Log g. Mtxt, Ttxt and Gtxt are then put together as string to load correct files.
        Mtxt = M_H_Grid_load[optM]
        Ttxt = "{:2.0f}".format(Teff_Grid[optT] / 100)
        if Teff_Grid[optT] == 5777:
            Ttxt = "{:4.0f}".format(Teff_Grid[optT])
        Gtxt = "{:2.0f}".format(logg_Grid[optG] * 10)

        #
        file = 'mmu_t' + Ttxt + 'g' + Gtxt + 'm' + Mtxt + 'v05.flx'
        print('Filename:', file)

        # Read data from IDL .sav file
        sav = readsav(os.path.join(direc, file))  # readsav reads an IDL .sav file
        ws = sav['mmd'].lam[0]  # read in wavelength
        flux = sav['mmd'].flx  # read in flux
        Teff_model = Teff_Grid[optT]
        logg_model = logg_Grid[optG]
        MH_model = str(M_H_Grid[optM])

        print('\nClosest values to your inputs:')
        print('Teff  : ', Teff_model)
        print('M_H   : ', MH_model)
        print('log(g): ', logg_model)

        f0 = flux[0]
        f1 = flux[1]
        f2 = flux[2]
        f3 = flux[3]
        f4 = flux[4]
        f5 = flux[5]
        f6 = flux[6]
        f7 = flux[7]
        f8 = flux[8]
        f9 = flux[9]
        f10 = flux[10]

        # Make single big array of them
        fcalc = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
        phot1 = np.zeros(fcalc.shape[0])

        # Mu from grid
        # 0.00000    0.0100000    0.0500000     0.100000     0.200000     0.300000   0.500000     0.700000     0.800000     0.900000      1.00000
        mu = sav['mmd'].mu

        # Passed on to main body of function are: ws, fcalc, phot1, mu

    ### Load response function and interpolate onto kurucz model grid

    # FOR STIS
    if grating == 'G430L':
        sav = readsav(os.path.join(dirsen, 'G430L.STIS.sensitivity.sav'))  # wssens,sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 3

    if grating == 'G750M':
        sav = readsav(os.path.join(dirsen, 'G750M.STIS.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 0.554

    if grating == 'G750L':
        sav = readsav(os.path.join(dirsen, 'G750L.STIS.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 4.882

    # FOR WFC3
    if grating == 'G141':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'G141.WFC3.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 1

    if grating == 'G102':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'G141.WFC3.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 1

    if grating == 'G280':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'G280.WFC3.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 1

    # FOR JWST
    if grating == 'NIRSpecPrism':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'NIRSpec.prism.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 12


    widek = np.arange(len(wsdata))
    wsHST = wssens
    wsHST = np.concatenate((np.array([wsHST[0] - wdel - wdel, wsHST[0] - wdel]),
                            wsHST,
                            np.array([wsHST[len(wsHST) - 1] + wdel,
                                      wsHST[len(wsHST) - 1] + wdel + wdel])))

    respoutHST = sensitivity / np.max(sensitivity)
    respoutHST = np.concatenate((np.zeros(2), respoutHST, np.zeros(2)))
    inter_resp = interp1d(wsHST, respoutHST, bounds_error=False, fill_value=0)
    respout = inter_resp(ws)  # interpolate sensitivity curve onto model wavelength grid

    wsdata = np.concatenate((np.array([wsdata[0] - wdel - wdel, wsdata[0] - wdel]), wsdata,
                             np.array([wsdata[len(wsdata) - 1] + wdel, wsdata[len(wsdata) - 1] + wdel + wdel])))
    respwavebin = wsdata / wsdata * 0.0
    widek = widek + 2  # need to add two indicies to compensate for padding with 2 zeros
    respwavebin[widek] = 1.0
    data_resp = interp1d(wsdata, respwavebin, bounds_error=False, fill_value=0)
    reswavebinout = data_resp(ws)  # interpolate data onto model wavelength grid

    # Integrate over the spectra to make synthetic photometric points.
    for i in range(fcalc.shape[0]):  # Loop over spectra at diff angles
        fcal = fcalc[i, :]
        Tot = int_tabulated(ws, ws * respout * reswavebinout)
        phot1[i] = (int_tabulated(ws, ws * respout * reswavebinout * fcal, sort=True)) / Tot

    if ld_model == '1D':
        yall = phot1 / phot1[0]
    elif ld_model == '3D':
        yall = phot1 / phot1[10]

    Co = np.zeros((6, 4))   # NOT-REUSED

    A = [0.0, 0.0, 0.0, 0.0]  # c1, c2, c3, c4      # NOT-REUSED
    x = mu[1:]     # wavelength
    y = yall[1:]   # flux
    weights = x / x   # NOT-REUSED

    # Start fitting the different models
    fitter = LevMarLSQFitter()

    # Fit a four parameter non-linear limb darkening model and get fitted variables, c1, c2, c3, c4.
    corot_4_param = nonlinear_limb_darkening()
    corot_4_param = fitter(corot_4_param, x, y)
    c1, c2, c3, c4 = corot_4_param.parameters

    # Fit a three parameter non-linear limb darkening model and get fitted variables, cp2, cp3, cp4 (cp1 = 0).
    corot_3_param = nonlinear_limb_darkening()
    corot_3_param.c0.fixed = True  # 3 param is just 4 param with c0 = 0.0
    corot_3_param = fitter(corot_3_param, x, y)
    cp1, cp2, cp3, cp4 = corot_3_param.parameters

    # Fit a quadratic limb darkening model and get fitted parameters aLD and bLD.
    quadratic = quadratic_limb_darkening()
    quadratic = fitter(quadratic, x, y)
    aLD, bLD = quadratic.parameters

    # Fit a linear limb darkening model and get fitted variable uLD.
    linear = nonlinear_limb_darkening()
    linear.c0.fixed = True
    linear.c2.fixed = True
    linear.c3.fixed = True
    linear = fitter(linear, x, y)
    uLD = linear.c1.value

    print('\nLimb darkening parameters:')
    print("4param \t{:0.8f}\t{:0.8f}\t{:0.8f}\t{:0.8f}".format(c1, c2, c3, c4))
    print("3param \t{:0.8f}\t{:0.8f}\t{:0.8f}".format(cp2, cp3, cp4))
    print("Quad \t{:0.8f}\t{:0.8f}".format(aLD, bLD))
    print("Linear \t{:0.8f}".format(uLD))

    return uLD, c1, c2, c3, c4, cp1, cp2, cp3, cp4, aLD, bLD


def int_tabulated(X, F, sort=False):
    Xsegments = len(X) - 1

    # Sort vectors into ascending order.
    if not sort:
        ii = np.argsort(X)
        X = X[ii]
        F = F[ii]

    while (Xsegments % 4) != 0:
        Xsegments = Xsegments + 1

    Xmin = np.min(X)
    Xmax = np.max(X)

    # Uniform step size.
    h = (Xmax + 0.0 - Xmin) / Xsegments
    # Compute the interpolates at Xgrid.
    # x values of interpolates >> Xgrid = h * FINDGEN(Xsegments + 1L) + Xmin
    z = splev(h * np.arange(Xsegments + 1) + Xmin, splrep(X, F))

    # Compute the integral using the 5-point Newton-Cotes formula.
    ii = (np.arange((len(z) - 1) / 4, dtype=int) + 1) * 4

    return np.sum(2.0 * h * (7.0 * (z[ii - 4] + z[ii]) + 32.0 * (z[ii - 3] + z[ii - 1]) + 12.0 * z[ii - 2]) / 45.0)





