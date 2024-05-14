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
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
import netCDF4 as nc
import os
from wrf import getvar, to_np, ll_to_xy, vertcross, smooth2d, CoordPair, get_basemap, latlon_coords, interplevel
import sys
from decimal import Decimal
from datetime import datetime, timedelta
import matplotlib.lines as mlines
#from numpy import loadtxt


TKE_ave_1_v = np.loadtxt("TKE_adv_ave_TRI_aug19.1.csv")
TKE_ave_2_v = np.loadtxt("TKE_adv_ave_TRI_aug19.2.csv")
TKE_shu_1_v = np.loadtxt("TKE_shu_ave_TRI_aug19.1.csv")
TKE_shv_1_v = np.loadtxt("TKE_shv_ave_TRI_aug19.1.csv")
TKE_shu_2_v = np.loadtxt("TKE_shu_ave_TRI_aug19.2.csv")
TKE_shv_2_v = np.loadtxt("TKE_shv_ave_TRI_aug19.2.csv")
u_1_v = np.loadtxt("u_ave_TRI_aug19.1.csv")
v_1_v = np.loadtxt("v_ave_TRI_aug19.1.csv")
u_2_v = np.loadtxt("u_ave_TRI_aug19.2.csv")
v_2_v = np.loadtxt("v_ave_TRI_aug19.2.csv")
w_1_v = np.loadtxt("w_ave_TRI_aug19.1.csv")
w_2_v = np.loadtxt("w_ave_TRI_aug19.2.csv")
b_1_v = np.loadtxt("TKE_b_ave_TRI_aug19.1.csv")
tt_1_v = np.loadtxt("tt_ave_TRI_aug19.1.csv")
tt_2_v = np.loadtxt("tt_ave_TRI_aug19.2.csv")

#vertical budget
TKE_adv_v = w_2_v[:]*(TKE_ave_2_v[:] - TKE_ave_1_v[:])/20.0     #the level diff at 40 m is 20m, and at 250 is 25 m.
TKE_sh_v = TKE_shu_2_v[:]*(u_2_v[:] - u_1_v[:])/20.0 + TKE_shv_2_v[:]*(v_2_v[:] - v_1_v[:])/20.0
TKE_tt_v = (tt_2_v[:] - tt_1_v[:])/20.0
TKE_b_v = b_1_v[:]

time = np.loadtxt("time_ave_aug19_shade.csv")

diss = pd.read_csv('/glade/work/mina/data_python_sodar/diss_20m.csv')
diss_o = pd.read_csv('/glade/work/mina/data_python_sodar/diss_20m_sonic.csv')

TKE_ave_1_o = np.loadtxt("TKE_adv_ave_TRI_aug19_ob.csv")
TKE_shu_1_o = np.loadtxt("TKE_shu_ave_TRI_aug19_ob.csv")
TKE_shv_1_o = np.loadtxt("TKE_shv_ave_TRI_aug19_ob.csv")
TKE_shw_1_o = np.loadtxt("TKE_shw_ave_TRI_aug19_ob.csv")
u_1_o = np.loadtxt("u_ave_TRI_aug19_ob.csv")
v_1_o = np.loadtxt("v_ave_TRI_aug19_ob.csv")
w_1_o = np.loadtxt("w_ave_TRI_aug19_ob.csv")
b_1_o = np.loadtxt("TKE_b_ave_TRI_aug19_ob.csv")
tt_1_o = np.loadtxt("tt_ave_TRI_aug19_ob.csv")

#vertical budget Observations
TKE_adv_v_o = w_1_o[0:46]*(TKE_ave_1_o[0:46])/2.0  
TKE_sh_v_o = TKE_shu_1_o[0:46]*(u_1_o[0:46])/2.0 + TKE_shv_1_o[0:46]*(v_1_o[0:46])/2.0 + TKE_shw_1_o[0:46]*(w_1_o[0:46])/2.0
TKE_tt_v_o = tt_1_o[0:46]/2.0
TKE_b_v_o = b_1_o[0:46]
time_o = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,20,20.5,21,21.5,22,22.5,23]

#--------------------------------------------------------------------
data_u_tri_19 = np.loadtxt("../locations/TRI.d04.UU.20.aug19.shade",usecols=[0])
data_u260_tri_19 = np.loadtxt("../locations/TRI.d04.UU.260.aug19.shade",usecols=[0])
data_u40_tri_19 = np.loadtxt("../locations/TRI.d04.UU.40.aug19.shade",usecols=[0])
data_v_tri_19= np.loadtxt("../locations/TRI.d04.VV.20.aug19.shade",usecols=[0])
data_w_tri_19 = np.loadtxt("../locations/TRI.d04.WW.20.aug19.shade",usecols=[0])
data_tk_tri_19 = np.loadtxt("../locations/TRI.d04.TK.20.aug19.shade",usecols=[0])
time_tri_19 = np.loadtxt("../locations/TRI.d04.time.aug19.shade",usecols=[0])
sonic_u_tri_19 = np.loadtxt("../locations/sonic_raw_aug19_TRI.U",usecols=[0])
sonic_v_tri_19 = np.loadtxt("../locations/sonic_raw_aug19_TRI.V",usecols=[0])
sonic_w_tri_19 = np.loadtxt("../locations/sonic_raw_aug19_TRI.W",usecols=[0])
sonic_time_tri_19 = np.loadtxt("../locations/sonic_raw_aug19_TRI.time",usecols=[0])
chunk_size = 13487       #This is for 10 min. 10790 was for 8 min data size
chunk_size_ave = 40461       # 30 min
#chunk_size_ave = 80922       # 60 min
#chunk_size_ave =  6740        # 5 min

def moving_average(a, n=chunk_size) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

model_u_tri_19 = moving_average(data_u_tri_19)
residuals_u_tri_19 = data_u_tri_19[0:model_u_tri_19.size] - model_u_tri_19
model_v_tri_19 = moving_average(data_v_tri_19)
residuals_v_tri_19 = data_v_tri_19[0:model_u_tri_19.size] - model_v_tri_19
model_w_tri_19 = moving_average(data_w_tri_19)
residuals_w_tri_19 = data_w_tri_19[0:model_u_tri_19.size] - model_w_tri_19
TKE_tri_19 = 0.5*((residuals_u_tri_19)**2+(residuals_v_tri_19)**2+(residuals_w_tri_19)**2) + data_tk_tri_19[0:model_u_tri_19.size]


groups = [TKE_tri_19[x:x+chunk_size_ave] for x in range(0, len(TKE_tri_19), chunk_size_ave)]
TKE_ave_tri_19= np.zeros(len(groups))
for i in range(len(groups)):
  group = np.array(groups[i])
  TKE_ave_tri_19[i] = np.nanmean(group)

groups_time = [time_tri_19[x:x+chunk_size_ave] for x in range(0, len(time_tri_19), chunk_size_ave)]
time_ave_tri_19= np.zeros(len(groups_time))
for i in range(len(groups_time)):
  group_time = np.array(groups_time[i])
  time_ave_tri_19[i] = np.nanmean(group_time)

u = data_u260_tri_19[0:model_u_tri_19.size]
u1 = data_u40_tri_19[0:model_u_tri_19.size]

groups_u = [u[x:x+chunk_size_ave] for x in range(0, len(u), chunk_size_ave)]
u_ave= np.zeros(len(groups_u))
for j in range(len(groups_u)):
  groupu = np.array(groups_u[j])
  u_ave[j] = np.nanmean(groupu)

groups_u = [u1[x:x+chunk_size_ave] for x in range(0, len(u1), chunk_size_ave)]
u1_ave= np.zeros(len(groups_u))
for j in range(len(groups_u)):
  groupu = np.array(groups_u[j])
  u1_ave[j] = np.nanmean(groupu)
#--------------------------------------------------------------------------------------
chunk_size_sonic = 11800       #11411  #10 min
chunk_size_sonic_ave = 34233  #30 min
#chunk_size_sonic_ave = 68466  #60min

def moving_average(a, n=chunk_size_sonic) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

model_u_s_tri_19 = moving_average(sonic_u_tri_19)
residuals_u_s_tri_19 = sonic_u_tri_19[0:model_u_s_tri_19.size] - model_u_s_tri_19
model_v_s_tri_19 = moving_average(sonic_v_tri_19)
residuals_v_s_tri_19 = sonic_v_tri_19[0:model_u_s_tri_19.size] - model_v_s_tri_19
model_w_s_tri_19 = moving_average(sonic_w_tri_19)
residuals_w_s_tri_19 = sonic_w_tri_19[0:model_u_s_tri_19.size] - model_w_s_tri_19
TKE_sonic_tri_19 = 0.5*((residuals_u_s_tri_19)**2+(residuals_v_s_tri_19)**2+(residuals_w_s_tri_19)**2)

#TKE calculations for sonic raw data with filtering applied
u = residuals_u_s_tri_19
f = fftpack.fftfreq(u.size,0.05)
uf = fftpack.fft(residuals_u_s_tri_19)
uff = uf
uff[np.abs(f) > 1] = 0.
uffc = fftpack.ifft(uff)
v = residuals_v_s_tri_19
vf = fftpack.fft(residuals_v_s_tri_19)
vff = vf
vff[np.abs(f) > 1] = 0.
vffc = fftpack.ifft(vff)
w = residuals_w_s_tri_19
wf = fftpack.fft(residuals_w_s_tri_19)
wff = wf
wff[np.abs(f) > 1] = 0.
wffc = fftpack.ifft(wff)
tkef_tri_19 = 0.5*(uffc*uffc+vffc*vffc+wffc*wffc)

#Averaging raw data
#original sonic signal

g_time = [sonic_time_tri_19[x:x+chunk_size_sonic_ave] for x in range(0, len(sonic_time_tri_19), chunk_size_sonic_ave)]
sonic_time_ave_tri_19 = np.zeros(len(g_time))
for i in range(len(g_time)):
  time_s = np.array(g_time[i])
  sonic_time_ave_tri_19[i] = np.nanmedian(time_s)

#fft filtered

gfs_sonic = [tkef_tri_19[x:x+chunk_size_sonic_ave] for x in range(0, len(tkef_tri_19), chunk_size_sonic_ave)]
TKE_ave_sonic_filter_tri_19 = np.zeros(len(gfs_sonic))
for i in range(len(gfs_sonic)):
  gf_sonic = np.array(gfs_sonic[i])
  TKE_ave_sonic_filter_tri_19[i] = np.nanmean(gf_sonic)



fontP = FontProperties()
fontP.set_size('xx-small')
plt.figure()
plt.rcParams.update(plt.rcParamsDefault)
plt.subplot(3,1,1)
axes = plt.gca()
plt.xlim([0, 24])
plt.ylim([0, 6])
plt.ylabel('TKE (m$^2$ s$^{-2}$)',fontsize=16)
axes.tick_params(axis='both',which='major',labelsize=16)
axes.xaxis.set_tick_params(width=3)
axes.yaxis.set_tick_params(width=3)
#axes.minorticks_on()
#axes.xaxis.set_tick_params(which='minor', bottom=False)
#plt.xlabel('Local Time (h)')
plt.plot(sonic_time_ave_tri_19[0:TKE_ave_sonic_filter_tri_19.size],TKE_ave_sonic_filter_tri_19,color='darkslategray',marker='o',markerfacecolor='darkslategray', markersize=7, markeredgecolor='darkslategray', linewidth = 2.5,label="Obs (TKE)")
plt.plot(time_ave_tri_19[0:TKE_ave_tri_19.size] - 7.0,TKE_ave_tri_19,'darkslategray',linewidth = 2.5,label="Model (TKE)")
plt.grid(color='lightgrey', linestyle='-', linewidth=1.2)
for axis in ['top', 'bottom', 'left', 'right']:
    axes.spines[axis].set_linewidth(1.5)  # change width
ax2=plt.twinx()
ax2.set_ylim([0, 10])
axes.tick_params(axis='both',which='major',labelsize=20)
axes.xaxis.set_tick_params(width=3)
axes.yaxis.set_tick_params(width=3)
ax2.set_ylabel('U (m s$^{-1}$)',fontsize=16)
ax2.plot(time_ave_tri_19[0:TKE_ave_tri_19.size] - 7.0,u_ave,'lightseagreen',linestyle='--',linewidth = 2,label="Model cross-ridge wind at 250 m agl")
ax2.plot(time_ave_tri_19[0:TKE_ave_tri_19.size] - 7.0,u1_ave,'green',linestyle='--',linewidth = 2,label="Model cross-ridge wind at 20 m agl")
plt.legend(bbox_to_anchor=(0.55, 1.0), borderaxespad=0.,fontsize=14,fancybox=True,shadow=False)



axes = plt.gca()
plt.subplot(3,1,2)
axes = plt.gca()
#plt.xlim([0, 10])
plt.xlim([0, 24])
plt.ylim([-0.5, 0.5])
plt.ylabel('TKE Budget (m$^2$s$^{-3}$)', fontsize = 14)
axes.tick_params(axis='both',which='major',labelsize=16)
#ax.set_xticks([0,,4,6])
axes.minorticks_on()
axes.xaxis.set_tick_params(which='minor', bottom=False)
#axes.yaxis.set_tick_params(which='minor', bottom=False)
axes.xaxis.set_tick_params(width=3)
axes.yaxis.set_tick_params(width=3)
#plt.xlabel('Local Time (h)', fontsize = 18)
#plt.plot(time-7.0,TKE_sh_v,color='mediumvioletred',linewidth=2.5,label="Shear")
plt.plot(time-7.0,TKE_adv_v,color='cornflowerblue',linewidth=2.5,label="Advect")
plt.plot(time-7.0,TKE_tt_v,color='seagreen',linewidth=2.5,label="TTransp")
plt.plot(diss['time'],-diss['diss_TRI'],'black',linewidth=3,label="Diss")
plt.plot(diss_o['time'],-diss_o['diss_TRI'],'black',marker='o',markerfacecolor='black',markersize=7, markeredgecolor='grey',linewidth=2)
plt.plot(time_o,-TKE_adv_v_o[0:45],color='cornflowerblue',marker='o',markerfacecolor='cornflowerblue', markersize=7, markeredgecolor='grey',linewidth=2.5)
plt.plot(time_o,-TKE_tt_v_o[0:45],color='seagreen',marker='o',markerfacecolor='seagreen', markersize=7, markeredgecolor='grey',linewidth=2.5)
plt.plot(time-7.0,TKE_sh_v,color='mediumvioletred',linewidth=2.5,label="Shear")
plt.plot(time_o,-TKE_sh_v_o[0:45],color='mediumvioletred',marker='o',markerfacecolor='mediumvioletred', markersize=7, markeredgecolor='grey',linewidth=2.5)

plt.grid(color='lightgrey', linestyle='-', linewidth=1.2)
for axis in ['top', 'bottom', 'left', 'right']:
    axes.spines[axis].set_linewidth(1.5)  # change width
plt.legend(loc = 'lower left', borderaxespad=0.,fontsize='12',fancybox=True,shadow=False,ncol = 2)

axes = plt.gca()
plt.subplot(3,1,3)
axes = plt.gca()
plt.xlim([0, 24])
plt.ylim([-0.05, 0.05])
plt.ylabel('TKE Budget (m$^2$s$^{-3}$)', fontsize = 14)
axes.tick_params(axis='both',which='major',labelsize=16)
axes.minorticks_on()
axes.xaxis.set_tick_params(which='minor', bottom=False)
axes.xaxis.set_tick_params(width=3)
axes.yaxis.set_tick_params(width=3)
plt.xlabel('Local Time (h)', fontsize = 18)
plt.plot(time-7.0,TKE_b_v*10.0,color='violet',linewidth=2.5,label="Buoyancy")
plt.plot(time_o,TKE_b_v_o[0:45]*10.0,color='violet',marker='o',markerfacecolor='violet', markersize=7, markeredgecolor='grey',linewidth=2.5)
plt.grid(color='lightgrey', linestyle='-', linewidth=1.2)
for axis in ['top', 'bottom', 'left', 'right']:
    axes.spines[axis].set_linewidth(1.5)  # change width
l1 = plt.legend(loc = 'lower left', borderaxespad=0.,fontsize='12',fancybox=True,shadow=False,ncol=2)
b_patch = mpatches.Patch(color='b', label='model')
model_star = mlines.Line2D([], [], color='black', linestyle='solid',
                          markersize=5, label='Model')
obs_star = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=5, label='Observations')
model_dir = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                          markersize=5, label='Model WD')
#l2 = plt.legend(loc = 'lower right', bbox_to_anchor=(0.7, -0.3), shadow=False, ncol=3, fontsize='10',handles=[obs_star, model_star, model_dir])
axes.add_artist(l1)
#axes.add_artist(l2)
plt.savefig('TKE_allbudget_TRI_aug19_timeseries_20m.png')   # save the figure to file
plt.show()
plt.close()
