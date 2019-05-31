
#LINES_HERSCHEL.PY updated for PACS spectra by fitting line by line 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import  GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling import models, fitting
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from scipy.optimize import curve_fit
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum
from specutils.fitting import fit_lines
from astropy import units as u
import os as os
import glob
import csv
import random
from Herschel import *
from Lines_Herschel import *

def fit_continuum(Wave,Flux,Deg):				#Returns the fit of a continuum for a given Wave/Flux array. Deg is to define the degree of your 1D polynom
	fit_pol=fitting.LinearLSQFitter()
	l_init_continuum = models.Polynomial1D(degree=Deg)

	return l_init_continuum #, y_continuum_poly

def dust(wvl,amplitude=1.,temperature=18,beta=2):
	k=1.38e-23
	c=3E14
	h=6.63e-34

	return amplitude*(wvl*1E-6)**(-beta-2)/(np.exp(h*c/(wvl*k*temperature))-1)


def dust_total(wvl,amplitude1=1.,amplitude2=2.,temperature=18,beta=2):			
	k=1.38e-23
	c=3E14
	h=6.63e-34
	'''
	y=wvl
	y[wvl<200]=amplitude1*(wvl[wvl<200]*1E-6)**(-beta-2)/(np.exp(h*c/(wvl[wvl<200]*k*temperature))-1)
	y[wvl>200]=amplitude2*(wvl[wvl>200]*1E-6)**(-beta-2)/(np.exp(h*c/(wvl[wvl>200]*k*temperature))-1)'''
	
	'''
	index_pacs=np.where(wvl<200)
	index_spire=np.where(wvl>200)
	for i in index_pacs:
		y[i]=amplitude1*(wvl[i]*1E-6)**(-beta-2)/(np.exp(h*c/(wvl[i]*k*temperature))-1)
	for j in index_spire:
		y[j]=amplitude2*(wvl[j]*1E-6)**(-beta-2)/(np.exp(h*c/(wvl[j]*k*temperature))-1)'''
	amplitude=0
	index_pacs=np.where(wvl<200)
	index_spire=np.where(wvl>200)
	for i in index_pacs:
		if wvl[i]>200:
			amplitude=amplitude1
	for j in index_spire:
		amplitude=amplitude2

	return amplitude*(wvl*1E-6)**(-beta-2)/(np.exp(h*c/(wvl*k*temperature))-1)


def fit_lines_gauss(Flux_range,init_pos,stddev,emission):		#Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	#fit_g = fitting.LevMarLSQFitter()

	ampli=Flux_range[np.where(Flux_range==np.nanmax(Flux_range))]

	if emission=='Em':
		ampli=ampli
		g_init= models.Gaussian1D(amplitude=ampli ,mean=init_pos, stddev=stddev, bounds={"amplitude": (0,1),"mean": (init_pos-0.1, init_pos+0.1), "stddev": (0,0.2)})
	if emission=='Abs':
		ampli=-ampli
		g_init= models.Gaussian1D(amplitude=ampli ,mean=init_pos, stddev=stddev, bounds={"amplitude": (-1,0),"mean": (init_pos-0.1, init_pos+0.1), "stddev": (0,0.2)})

	if init_pos==122.5:
		g_init= models.Gaussian1D(amplitude=ampli ,mean=init_pos, stddev=stddev, bounds={"amplitude": (0,1),"mean": (init_pos-0.1, init_pos+0.1), "stddev": (0,0.3)})
	
	if init_pos==57.35 or init_pos==63.17:
		g_init= models.Gaussian1D(amplitude=ampli ,mean=init_pos, stddev=stddev, bounds={"amplitude": (0,1),"mean": (init_pos-0.1, init_pos+0.1), "stddev": (0,0.02)})
	
	return g_init


def Fit_line(source_name,line,row,column):		#row/colonne définit la position d'un pixel  instru=MAPS/SPIRE_Map
	fit_g = fitting.LevMarLSQFitter()
	Wave_red,Flux_red= flux_PACS(source_name,'R')[2],flux_PACS(source_name,'R')[4]
	Wave_blue,Flux_blue= flux_PACS(source_name,'B')[2],flux_PACS(source_name,'B')[4]

	my_dict=csv_to_dict('Bulles.csv')
	lines=list(my_dict[source_name])[5:12]

	#l_init_continuum=fit_continuum(Wave,Flux,1)
	#g_init=l_init_continuum

	if line=='NII_122_em' or line=='OH_119' or line=='OI_145' or line=='NII_122_abs' or line=='Thing_123':
		cond=np.where(np.isfinite(Flux_red[:,row,column]))
		Wave, Flux=Wave_red[cond], Flux_red[:,row,column][cond]
		Noise=np.nanstd(Flux[np.where((Wave>=90)&(Wave<=105))])
	if line=='NIII_57' :
		cond=np.where(np.isfinite(Flux_blue[:,row,column]))
		Wave, Flux=Wave_blue[cond], Flux_blue[:,row,column][cond]
		Noise=np.nanstd(Flux[np.where(((Wave>=56)&(Wave<=57))|((Wave>=57.5)&(Wave<=58.5)))])
	if line=='OI_63' :
		cond=np.where(np.isfinite(Flux_blue[:,row,column]))
		Wave, Flux=Wave_blue[cond], Flux_blue[:,row,column][cond]
		Noise=np.nanstd(Flux[np.where(((Wave>=62)&(Wave<=63))|((Wave>=63.5)&(Wave<=64.5)))])
	#print(cond)
	if np.shape(cond)[1]!=0 :

		'''
		if line=='NII_122_em' and line=='Thing_123':
			NII=np.where((Wave>120.5) & (Wave<123.5))
			Wave_range,Flux_range=Wave[NII],Flux[NII]
			l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
			g_init=l_init_continuum+fit_lines_gauss(121.9,'Em')+fit_lines_gauss(122.5,'Em')'''

		if line=='NII_122_em' or line=='Thing_123':
			#if  int(my_dict[source_name]['NII_122_em'])==1 and  int(my_dict[source_name]['Thing_123'])==1:
			index=np.where((Wave>120.5) & (Wave<124))
			pos=121.9
			Wave_range,Flux_range=Wave[index],Flux[index]
			l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
			g_init=l_init_continuum+fit_lines_gauss(Flux_range,pos,0.1,'Em')+fit_lines_gauss(Flux_range,122.5,0.05,'Em')					
			fit_line=fit_g(g_init,Wave_range,Flux_range)
			if line=='NII_122_em':
				amplitude, std=fit_line.amplitude_1[0],abs(fit_line.stddev_1[0])
			if line=='Thing_123':
				amplitude, std=fit_line.amplitude_2[0],abs(fit_line.stddev_2[0])
			'''
			else:
				print('Hello')
				NII=np.where((Wave>120.5) & (Wave<122.5))
				Wave_range,Flux_range=Wave[NII],Flux[NII]
				l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
				g_init=l_init_continuum+fit_lines_gauss(121.9,0.01,'Em')
				fit_line=fit_g(g_init,Wave_range,Flux_range)
				amplitude, std=fit_line.amplitude_1[0],abs(fit_line.stddev_1[0])	'''
		
		if line=='NII_122_abs':
			index=np.where((Wave>120.5) & (Wave<122.5))
			pos=121.9
			Wave_range,Flux_range=Wave[index],Flux[index]
			l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
			g_init=l_init_continuum+fit_lines_gauss(Flux_range,pos,0.01,'Abs')
			fit_line=fit_g(g_init,Wave_range,Flux_range)
			amplitude, std=fit_line.amplitude_1[0],abs(fit_line.stddev_1[0])	
	
		if line=='NIII_57':
			index=np.where((Wave>53) & (Wave<60))
			pos=57.35
			Wave_range,Flux_range=Wave[index],Flux[index]
			l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
			g_init=l_init_continuum+fit_lines_gauss(Flux_range,pos,0.004,'Em')
			fit_line=fit_g(g_init,Wave_range,Flux_range)
			amplitude, std=fit_line.amplitude_1[0],abs(fit_line.stddev_1[0])	
			
		if line=='OH_119':
			index=np.where((Wave>118) & (Wave<120.25))
			pos=119.2
			Wave_range,Flux_range=Wave[index],Flux[index]
			l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
			g_init=l_init_continuum+fit_lines_gauss(Flux_range,pos,0.05,'Abs')+fit_lines_gauss(Flux_range,119.45,0.05,'Abs')
			fit_line=fit_g(g_init,Wave_range,Flux_range)
			amplitude, std=fit_line.amplitude_1[0]+fit_line.amplitude_2[0],abs(fit_line.stddev_1[0])+abs(fit_line.stddev_2[0])

		if line=='OI_63':
			index=np.where((Wave>62.5) & (Wave<63.5))
			pos=63.17
			Wave_range,Flux_range=Wave[index],Flux[index]
			l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
			g_init=l_init_continuum+fit_lines_gauss(Flux_range,pos,0.009,'Em')
			fit_line=fit_g(g_init,Wave_range,Flux_range)
			amplitude, std=fit_line.amplitude_1[0],abs(fit_line.stddev_1[0])	

		if line=='OI_145':
			index=np.where((Wave>144) & (Wave<146.25))
			pos=145.5
			Wave_range,Flux_range=Wave[index],Flux[index]
			l_init_continuum=fit_continuum(Wave_range,Flux_range,1)
			g_init=l_init_continuum+fit_lines_gauss(Flux_range,pos,0.008,'Em')
			fit_line=fit_g(g_init,Wave_range,Flux_range)
			amplitude, std=fit_line.amplitude_1[0],abs(fit_line.stddev_1[0])	

		
		#fit_line=fit_g(g_init,Wave_range,Flux_range)
		Line_flux=amplitude*std*np.sqrt(2*np.pi)
		#Signal_to_noise=abs(amplitude)/Noise
		Signal_to_noise=abs(Line_flux)/(Noise*pos/1600)
		#print(amplitude,std,Noise)

	else:
		Wave,Flux,index,pos, Noise, Wave_range, Flux_range, fit_line=0,0,0,0,0,0,0,0
		Line_flux=0
		Signal_to_noise=0



	return Wave,Flux, Noise,index,pos, Wave_range, Flux_range, fit_line, Line_flux, Signal_to_noise


def fit_all_lines(source_name):
	my_dict=csv_to_dict('Bulles.csv')
	lines=list(my_dict[source_name])[5:12]
	fit_g = fitting.LevMarLSQFitter()

	Flux_image_R, Flux_spectrum_R,Wave_red,WCS_R,Flux_red= flux_PACS(source_name,'R')#,flux_PACS(source_name,'R')[4]
	Flux_image_B, Flux_spectrum_B, Wave_blue, WCS_B, Flux_blue= flux_PACS(source_name,'B')#,flux_PACS(source_name,'B')[4]
	
	Source={}
	Source['Wave']=Wave_blue,Wave_red
	Source['Flux']=Flux_blue, Flux_red
	Source['WCS']=WCS_B,WCS_R

	for line in lines:
		#print(line)
		if line=='NII_122_em' or line=='OH_119' or line=='OI_145' or line=='NII_122_abs' or line=='Thing_123':
			Flux_signal_to_noise=np.zeros_like(Flux_image_R)
			Line_flux_int=np.zeros_like(Flux_image_R)

			for row in range(np.shape(Flux_image_R)[0]):
				for column in range(np.shape(Flux_image_R)[1]):
					#if int(my_dict[source_name][line])==1:
						#print(line,row,column)
						#Wave,Flux, Noise, Wave_range, Flux_range, fit_line, 
					Line_flux, Signal_to_noise=Fit_line(source_name,line,row,column)[8:]
					Line_flux_int[row,column]=Line_flux
					Flux_signal_to_noise[row,column]=Signal_to_noise


		if line=='NIII_57' or line=='OI_63' :
			Flux_signal_to_noise=np.zeros_like(Flux_image_B)
			Line_flux_int=np.zeros_like(Flux_image_B)
					
			for row in range(np.shape(Flux_image_B)[0]):
				for column in range(np.shape(Flux_image_B)[1]):
					#if int(my_dict[source_name][line])==1:
					Line_flux, Signal_to_noise=Fit_line(source_name,line,row,column)[8:]
					Line_flux_int[row,column]=Line_flux
					Flux_signal_to_noise[row,column]=Signal_to_noise
			

		Source[str(line)]=Line_flux_int,Flux_signal_to_noise
		print(line)
	write_dict(Source,str(source_name)+'_PACS.npy')

	return Source, Line_flux_int, Flux_signal_to_noise

def plot_spectrum_fitted(source_name, row, column):
	my_dict=csv_to_dict('Bulles.csv')
	lines=list(my_dict[source_name])[5:12]
	fit_g = fitting.LevMarLSQFitter()

	Wave_red,WCS_R,Flux_red= flux_PACS(source_name,'R')[2:]
	Wave_blue,WCS_B,Flux_blue= flux_PACS(source_name,'B')[2:]

	plt.figure()
	plt.plot(Wave_red,Flux_red[:,row,column])
	plt.plot(Wave_blue,Flux_blue[:,row,column])

	for line in lines:
	
		Wave,Flux, Noise,index,pos, Wave_range, Flux_range, fit_line, Line_flux, Signal_to_noise=Fit_line(source_name,line,row,column)
		if fit_line!=0:
			y_line=fit_line(Wave_range)
		else:
			continue
		plt.plot(Wave_range,y_line)
		print(line)
		print(Line_flux, Signal_to_noise, Noise)

	plt.show()

	return 

#np.where(Source[line][1]==np.nanmax(Source[line][1]))

def plot_line_map(source_name):
	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()
	lines=list(my_dict[source_name])[5:12]


	Source=np.load(direc1+'MAPS5/'+str(source_name)+'_PACS.npy',allow_pickle=True).item()
	WCS_R=Source['WCS'][1]
	WCS_B=Source['WCS'][0]

	#line='NII_122_em'

	num_name=source_name[4:8]
	hdu_MIPS24=fits.open(direc+'Bulles_Nicolas/MB'+str(num_name)+'/MIPS24_MB'+str(num_name)+'.fits')

	Flux_range_Mips=np.nanmax(hdu_MIPS24[0].data[100:150,100:150])-np.nanmin(hdu_MIPS24[0].data[100:150,100:150])
	#Flux_range_Mips=np.nanmax(hdu_MIPS24[0].data)-np.nanmin(hdu_MIPS24[0].data)
	Levels=list(np.arange(np.nanmin(hdu_MIPS24[0].data[100:150,100:150]),np.nanmax(hdu_MIPS24[0].data[100:150,100:150]),Flux_range_Mips/10.))

	for line in lines:
		if  line=='OH_119'  or line=='NII_122_abs' :
			type_line='Absorption'
		else:
			type_line='Emission'
		plt.figure()
		ax=plt.subplot(1,2,1,projection=WCS_R)
		im=plt.imshow(Source[line][0])
		ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
		size=np.shape(Source[line][0])
		ax.set_xlim(-0.5,np.shape(Source[line][0])[1]-0.5)
		ax.set_ylim(-0.5,np.shape(Source[line][0])[0]-0.5)
		ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
		ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
		ax.coords[0].set_ticks(exclude_overlapping=True)
		plt.title(str(type_line)+' line '+str(line),fontsize=10)
		plt.colorbar(im, ax=ax , label=r"Flux ($W.m^{-2}.sr^{-1}$)")#,aspect=20)#,format='%.e')

			#plt.tight_layout()
		#plt.suptitle('Line Flux maps for '+str(source_name)+' in PACS R-Band  Image size: '+str(size), fontsize=11,fontstyle='italic')
		#plt.tight_layout()
		#plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=2.5)

		ax=plt.subplot(1,2,2,projection=WCS_R)
		im=plt.imshow(Source[line][1])
		ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
		size=np.shape(Source[line][0])
		ax.set_xlim(-0.5,np.shape(Source[line][1])[1]-0.5)
		ax.set_ylim(-0.5,np.shape(Source[line][1])[0]-0.5)
		ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
		ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
		ax.coords[0].set_ticks(exclude_overlapping=True)
		plt.title('Emission line '+str(line) +' Signal to noise ',fontsize=10)
		plt.colorbar(im, ax=ax , label=r"SNR",aspect=20)#,format='%.e')
		
			#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in PACS R-Band  Image size: '+str(size), fontsize=11,fontstyle='italic')
		#plt.tight_layout()
		plt.tight_layout()#(pad=3.0, w_pad=0.5, h_pad=2.5)

	plt.show()

	return

def all_spectrum(source_name):
	my_dict=csv_to_dict('Bulles.csv')
	lines=list(my_dict[source_name])[5:12]
	fit_g = fitting.LevMarLSQFitter()

	Wave_red,WCS_R,Flux_red= flux_PACS(source_name,'R')[2:]
	Wave_blue,WCS_B,Flux_blue= flux_PACS(source_name,'B')[2:]
	Source=np.load(direc1+'MAPS5/'+str(source_name)+'_PACS.npy',allow_pickle=True).item()

	a=1
	for i in range(5,10,1):
		for j in range(5,10,1):
			plt.subplot(5,5,a)
			plt.plot(Wave_red,Flux_red[:,i,j])
			plt.plot(Wave_blue,Flux_blue[:,i,j])
			for line in lines:
				Wave,Flux, Noise,index,pos, Wave_range, Flux_range, fit_line, Line_flux, Signal_to_noise=Fit_line(source_name,line,i,j)
				if fit_line!=0:
					y_line=fit_line(Wave_range)
				else:
					continue
				plt.plot(Wave_range,y_line)
				plt.title('pixel: ('+str(i)+','+str(j)+')')
				#print(line)
				#print(Line_flux, Signal_to_noise, Noise)
		a+=1	
	plt.show()

	return































