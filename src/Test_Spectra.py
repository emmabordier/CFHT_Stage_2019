import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import  GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model	#To define your own model for fitting
from reproject import reproject_interp 			#To reproject MIPS(Spitzer) on Herschel
from scipy.optimize import curve_fit
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum
from specutils.fitting import fit_lines
from astropy import units as u
from time import sleep
import os as os
import glob
import csv
from Herschel import *

direc='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/'
direc1='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/Data/'


def open_source(source_name,instru, colour_pacs=None,colour_spire=None):			#instru= PACS/Map/SPIRE_SPARSE
	index_obj=info_ID(instru)[0].index(source_name)
	obs_id=info_ID(instru)[1][index_obj]
	#name=glob.glob(path+str(obs_id))
	if instru=='PACS':
		path=direc1+str(instru)+'/HPS3DEQI'+str(colour_pacs)+'/'
		a=glob.glob(path+'hpacs'+str(obs_id)+'*.fits', recursive=True)

	if instru=='Map':
		path=direc1+'SPIRE_MAPPING/HR_'+str(colour_spire)+'/'
		a=glob.glob(path+'hspirespectrometer'+str(obs_id)+'*.fits', recursive=True)

	hdulist=fits.open(a[0])

	return  hdulist



def plot_spectra(source_name):
	hdulist=open_source(source_name,'PACS',colour_pacs='R')
	Pixel_size_deg=abs(hdulist[1].header['CDELT1']*hdulist[1].header['CDELT2'])	#en deg^2
	Pixel_size_sr=Pixel_size_deg*(3600*3600)/(4.25E10)			#1deg^2=3600*3600 arcesec^2 and 1sr=1rad^2=4.25E10 arcsec2
	Flux=(hdulist[1].data)/Pixel_size_sr						# En Jy/px > Jy/sr
	Flux=Flux/1E6 												# Jy/sr > MJy/sr	
	wvl=np.arange((hdulist[1].header['NAXIS3']))	# En microns
	wave=hdulist[1].header['CRVAL3']+hdulist[1].header['CDELT3']*wvl	

	#map_123=np.load('Test.npy',allow_pickle=True)
	map_123=np.load(str(source_name)+'_123.npy',allow_pickle=True)
	num_name=source_name[4:8]
	hdu_MIPS24=fits.open(direc+'Bulles_Nicolas/MB'+str(num_name)+'/MIPS24_MB'+str(num_name)+'.fits')
	wcs=map_123[2]


	#Flux_mean_pixel=np.nanmean(Flux,axis=(0))
	spectrum=[]
	for i in range(1,np.shape(Flux)[1]):
		for j in range (1,np.shape(Flux)[2]):
			spectrum.append(Flux[:,i,j])

	mean_spectrum=np.nanmean(Flux,axis=(1,2))
	spectrum=np.array(spectrum)
	map_123[0][np.where(map_123[0]==0)]=np.nan

	ax=plt.subplot(2,2,1, projection=wcs)
	im=plt.imshow(map_123[0])#, vmin=-2, vmax=50)
	#levels = np.arange(-2.0, 50, 0.4)
	ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)), levels=[50,60,70,80,100], colors='white',linewidths=0.6)#,vmin=-0.1, vmax=0.75)
	ax.set_xlim(0,np.shape((map_123[0]))[0]-1)
	ax.set_ylim(0,np.shape((map_123[0]))[1]-1)
	ax.set_xlabel('RA (J2000)')
	ax.set_ylabel('DEC (J2000)')
	plt.colorbar()
	ax.coords[0].set_ticks(exclude_overlapping=True)

	plt.subplot(2,2,2,projection=WCS(hdu_MIPS24[0].header))
	plt.imshow(hdu_MIPS24[0].data, vmin=0, vmax=200)#, cmap='binary')
	plt.colorbar()
	#cb1.set_label('Some Units')
	#levels = np.arange(0, 50, 0.4)
	C=plt.contour(hdu_MIPS24[0].data,levels=np.arange(50,100,20),colors='white',linewidths=0.6)#, vmin=0, vmax=20)
	#plt.colorbar()
	#plt.colorbar(C, shrink=0.8, extend='both')

	plt.subplot(2,2,3)
	plt.plot(wave,mean_spectrum)

	plt.figure()
	pixel_number=np.arange(1,len(spectrum)+1,1)
	shift=0
	#plt.subplot(2,2,4)
	for i in range(1,len(spectrum)):
		if np.shape(np.where(np.isfinite(spectrum[i-1])))[1]==0:
			continue
		
		else:
			plt.plot(wave,spectrum[i])#+shift)
			#plt.yticks(shift)
			#shift+=200
	
	#locs, labels = yticks()
	#locs, labels=plt.yticks( np.arange(len(spectrum)), pixel_number )
	#plt.yticks(pixel_number)
	#plt.tick_params(axis='y', labelsize=14)
	plt.xlim(120,125)
	#plt.suptitle('PACS- Red range '+ str(source_name))
	plt.show()

	return wave, spectrum

'''
wave, spectrum=plot_spectra('MGE_4191')
for i in range(1,len(spectrum),10):
		if np.shape(np.where(np.isfinite(spectrum[i-1])))[1]==0:
			continue
		
		else:
			plt.plot(wave,spectrum[i]+100)
			plt.show()
			sleep(1)
			plt.clf()'''












