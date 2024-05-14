#load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
#load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
#load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
#load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
#***********************************************
#begin
#***********************************************

import numpy as np
import scipy 
from scipy import signal
from scipy import fftpack
from scipy.fftpack import fftfreq
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
#matplotlib.style.use('ggplot')
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from matplotlib import rc
import netCDF4 as nc
import os
from wrf import getvar, to_np, ll_to_xy, vertcross, smooth2d, CoordPair, get_basemap, latlon_coords, interplevel
import sys
from decimal import Decimal
from datetime import datetime, timedelta
#from numpy import loadtxt

residuals_w_new = np.zeros(0)
TKE_new = np.zeros(0)

data_u = np.loadtxt("locations/TRI.d04.UU.aug19.shade")
data_v = np.loadtxt("locations/TRI.d04.VV.aug19.shade")
data_w = np.loadtxt("locations/TRI.d04.WW.aug19.shade")
data_th = np.loadtxt("locations/TRI.d04.TH.aug19.shade")
data_tk = np.loadtxt("locations/TRI.d04.TK.aug19.shade")
data_q = np.loadtxt("locations/TRI.d04.QV.aug19.shade")

#n = ([1,2,3,4,5])

chunk_size = 13487       #This is for 10 min. 10790 was for 8 min data size
chunk_size_ave = 40461       # 30 min
#chunk_size_ave = 80922       # 60 min
#chunk_size_ave =  6740        # 5 min

data_th_v = data_th*(1+0.61*data_q)

def moving_average(a, n=chunk_size) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

fname1 = "TKE_adv_ave_TRI_aug19.{i}.csv"
fname2 = "TKE_b_ave_TRI_aug19.{i}.csv"
fname3 = "TKE_shu_ave_TRI_aug19.{i}.csv"
fname4 = "TKE_shv_ave_TRI_aug19.{i}.csv"
fname5 = "u_ave_TRI_aug19.{i}.csv"
fname6 = "v_ave_TRI_aug19.{i}.csv"
fname7 = "w_ave_TRI_aug19.{i}.csv"
fname8 = "tt_ave_TRI_aug19.{i}.csv"
fname9 = "TKE_shw_ave_TRI_aug19.{i}.csv"

for i in [x for x in range(40) if x !=0]:

   model_u = moving_average(data_u[:,i])
   residuals_u= data_u[0:model_u.size,i] - model_u
   model_v = moving_average(data_v[:,i])
   residuals_v = data_v[0:model_v.size,i] - model_v
   model_w = moving_average(data_w[:,i])
   residuals_w = data_w[0:model_w.size,i] - model_w
   model_th_v = moving_average(data_th_v[:,i])
   residuals_th_v = data_th_v[0:model_th_v.size,i] - model_th_v
   TKE = 0.5*((residuals_u)**2+(residuals_v)**2+(residuals_w)**2) + data_tk[0:model_u.size,i]

   ustar = ((residuals_u*residuals_w)**2 + (residuals_v*residuals_w)**2)**0.25
   TKE_b = (9.8/data_th_v[0:model_th_v.size,i])*(residuals_w*residuals_th_v)

   TKE_sh1 = -(residuals_u*residuals_w)
   TKE_sh2 = -(residuals_v*residuals_w)
   TKE_sh3 = -(residuals_w*residuals_w)
   u = data_u[0:model_u.size,i]
   v = data_v[0:model_v.size,i]
   w = data_w[0:model_w.size,i]  
   tt = residuals_w*TKE   

   groups1 = [TKE[x:x+chunk_size_ave] for x in range(0, len(TKE), chunk_size_ave)]
   TKE_ave= np.zeros(len(groups1))
   for j in range(len(groups1)):
     group1 = np.array(groups1[j])
     TKE_ave[j] = np.nanmean(group1)
   np.savetxt(fname1.format(i=i),TKE_ave,delimiter=",") 
 
   groups2 = [TKE_b[x:x+chunk_size_ave] for x in range(0, len(TKE_b), chunk_size_ave)]
   TKE_b_ave= np.zeros(len(groups2))
   for j in range(len(groups2)):
     group2 = np.array(groups2[j])
     TKE_b_ave[j] = np.nanmean(group2)
   np.savetxt(fname2.format(i=i),TKE_b_ave,delimiter=",")
 
   groups3 = [TKE_sh1[x:x+chunk_size_ave] for x in range(0, len(TKE_sh1), chunk_size_ave)]
   TKE_sh1_ave= np.zeros(len(groups3))
   for j in range(len(groups3)):
     group3 = np.array(groups3[j])
     TKE_sh1_ave[j] = np.nanmean(group3)
   np.savetxt(fname3.format(i=i),TKE_sh1_ave,delimiter=",")
  
   groups4 = [TKE_sh2[x:x+chunk_size_ave] for x in range(0, len(TKE_sh2), chunk_size_ave)]
   TKE_sh2_ave= np.zeros(len(groups4))
   for j in range(len(groups4)):
     group4 = np.array(groups4[j])
     TKE_sh2_ave[j] = np.nanmean(group4)
   np.savetxt(fname4.format(i=i),TKE_sh2_ave,delimiter=",")

   groups5 = [u[x:x+chunk_size_ave] for x in range(0, len(u), chunk_size_ave)]
   u_ave= np.zeros(len(groups5))
   for j in range(len(groups5)):
     group5 = np.array(groups5[j])
     u_ave[j] = np.nanmean(group5)
   np.savetxt(fname5.format(i=i),u_ave,delimiter=",")

   groups6 = [v[x:x+chunk_size_ave] for x in range(0, len(v), chunk_size_ave)]
   v_ave= np.zeros(len(groups6))
   for j in range(len(groups6)):
     group6 = np.array(groups6[j])
     v_ave[j] = np.nanmean(group6)
   np.savetxt(fname6.format(i=i),v_ave,delimiter=",")
 
   groups7 = [w[x:x+chunk_size_ave] for x in range(0, len(w), chunk_size_ave)]
   w_ave= np.zeros(len(groups7))
   for j in range(len(groups7)):
     group7 = np.array(groups7[j])
     w_ave[j] = np.nanmean(group7)
   np.savetxt(fname7.format(i=i),w_ave,delimiter=",")

   groups8 = [tt[x:x+chunk_size_ave] for x in range(0, len(tt), chunk_size_ave)]
   tt_ave= np.zeros(len(groups8))
   for j in range(len(groups8)):
     group8 = np.array(groups8[j])
     tt_ave[j] = np.nanmean(group8)
   np.savetxt(fname8.format(i=i),tt_ave,delimiter=",")

   groups4 = [TKE_sh3[x:x+chunk_size_ave] for x in range(0, len(TKE_sh3), chunk_size_ave)]
   TKE_sh3_ave= np.zeros(len(groups4))
   for j in range(len(groups4)):
     group4 = np.array(groups4[j])
     TKE_sh3_ave[j] = np.nanmean(group4)
   np.savetxt(fname9.format(i=i),TKE_sh3_ave,delimiter=",")
  



