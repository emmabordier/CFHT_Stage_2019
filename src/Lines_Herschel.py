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

direc1='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/Data/'
direc='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/'

#Ces listes ont été trouvées avec le ipython Notebook CFHT_Nom.ipynb. Il existe des fichiers .txt et .csv répertoriant ces données pour chaque instrument
OBS_ID_PACS=[1342226189, 1342231302, 1342231303, 1342231738, 1342231739, 1342231740, 1342231741, 1342231742, 1342231743, 1342231744, 1342231745, 1342231746, 1342231747, 1342238730, 1342239689, 1342239741, 1342239757, 1342240165, 1342240166, 1342240167, 1342241267, 1342242444, 1342242445, 1342242446, 1342243105, 1342243106, 1342243107, 1342243108, 1342243505, 1342243506, 1342243507, 1342243513, 1342243901, 1342250917, 1342252270]
OBS_ID_SPIRE_MAP=[1342262927, 1342254039, 1342254040, 1342254041, 1342254042, 1342262919, 1342262924, 1342262926, 1342265807]
OBS_ID_SPIRE_SPARSE=[1342253970, 1342253971, 1342262922, 1342262923, 1342262925, 1342265810, 1342268284, 1342268285, 1342268286]
OBJECT_PACS=['MGE_4384', 'MGE_3438', 'MGE_3448', 'MGE_3269', 'MGE_3280', 'MGE_3739', 'MGE_3736', 'MGE_3719', 'MGE_3222', 'MGE_3354', 'MGE_3360', 'MGE_3681', 'MGE_3670', 'MGE_4048', 'MGE_4191', 'MGE_4206', 'MGE_4134', 'MGE_4121', 'MGE_4095', 'MGE_4218', 'MGE_3899', 'MGE_4552', 'MGE_4436', 'MGE_4486', 'MGE_4485', 'MGE_4111', 'MGE_4110', 'MGE_4204', 'MGE_3149', 'MGE_4602', 'MGE_4473', 'MGE_4524', 'MGE_3834', 'MGE_4239', 'MGE_4167']
OBJECT_SPIRE_MAP=['MGE_4121', 'MGE_3269', 'MGE_3280', 'MGE_3739', 'MGE_3736', 'MGE_4384', 'MGE_4204', 'MGE_4111', 'MGE_4485']
OBJECT_SPIRE_SPARSE=['MGE_3681', 'MGE_3448', 'MGE_4048', 'MGE_4206', 'MGE_4095', 'MGE_4134', 'MGE_3149', 'MGE_4602', 'MGE_4524']

def open_source(source_name,instru, colour_pacs=None,colour_spire=None):			#instru= PACS/Map/Sparse
	index_obj=info_ID(instru)[0].index(source_name)
	obs_id=info_ID(instru)[1][index_obj]
	#name=glob.glob(path+str(obs_id))
	if instru=='PACS':
		path=direc1+str(instru)+'/HPS3DEQI'+str(colour_pacs)+'/'
		a=glob.glob(path+'hpacs'+str(obs_id)+'*.fits', recursive=True)

	if instru=='Map':
		path=direc1+'SPIRE_MAPPING/HR_'+str(colour_spire)+'/'
		a=glob.glob(path+'hspirespectrometer'+str(obs_id)+'*.fits', recursive=True)

	if instru=='Sparse':
		path=direc1+'SPIRE_SPARSE/HR_spectrum_ext/'
		a=glob.glob(path+'hspirespectrometer'+str(obs_id)+'*.fits', recursive=True)

	hdulist=fits.open(a[0])

	return  hdulist

############################################################  PACS-lines   #################################################################
############################################################################################################################################

def Gauss(x, a, x0, sigma):					#Utile pour définir les modèles
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_continuum(Wave,Flux,Deg):				#Returns the fit of a continuum for a given Wave/Flux array. Deg is to define the degree of your 1D polynom
	fit_pol=fitting.LinearLSQFitter()
	l_init_continuum = models.Polynomial1D(degree=Deg)#,c0=1.,c1=1.)
	#continuum_poly=fit_pol(l_init_continuum, Wave, Flux)
	#y_continuum_poly= continuum_poly(Wave)

	return l_init_continuum #, y_continuum_poly


def dust(wvl,amplitude=1.,temperature=18,beta=2):
	#freq=freq*10**9
	#freq0=100e9
  	#beta=1.8
  	#T=18
  	k=1.38e-23
  	c=3E8
  	h=6.63e-34
  	return amplitude*(wvl)**(-beta-2)/(np.exp(h*c/(wvl*k*temperature))-1)

def fit_lines_gauss(Wave,Flux,init_pos,emission):		#Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	fit_g = fitting.LevMarLSQFitter()
	#fit_pol=fitting.LinearLSQFitter()
	#l_init_continuum = fit_continuum(Wave,Flux,2)
	#BB=blackbody_lambda(Wave,temperature=20*u.K)
	
	
	if emission=='Em':
		ampli=1E-7
		#np.nanmax(Flux[np.where((Wave>=init_pos-0.1) & (Wave<=init_pos+0.1))])
		#ampli=np.nanmax(Flux_range)
		#try:
		#	mean= float(Wave[np.where(Flux==np.nanmax(Flux_range))])
		#except:
		#	mean=ampli
	if emission=='Abs':
		ampli=-1E-7
		#(Flux[np.where((Wave>=init_pos-0.1) & (Wave<=init_pos+0.1))])
		#ampli=-abs(np.nanmin(Flux_range))
		#try:
		#	mean= float(Wave[np.where(Flux==np.nanmin(Flux_range))])
		#except:
		#	mean=ampli'''

	#ampli=np.nanmax(Flux[np.where((Wave>=init_pos-0.1) & (Wave<=init_pos+0.1))])

	g_init= models.Gaussian1D(amplitude=ampli ,mean=init_pos, stddev=0.08, bounds={"mean": (init_pos-0.1, init_pos+0.1), "stddev": (0,0.025*init_pos/60)})
	#u.W/u.m/u.m/u.sr/u.um
	#g_line= g_init+l_init_continuum

	#fit_line=fit_g(g_line,Wave,Flux)
	#y_line= fit_line(Wave)

	return g_init

def fit_lines_lorentz(Wave,Flux,Flux_range,emission):		#Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	fit_g = fitting.LevMarLSQFitter()
	fit_pol=fitting.LinearLSQFitter()
	l_init_continuum = fit_continuum(Wave,Flux,2)
	
	if emission=='Em':
		ampli=np.nanmax(Flux_range)
		try:		
			mean= float(Wave[np.where(Flux==np.nanmax(Flux_range))])
		except:
			mean=amplitude						#If Flux is full of 0, the function won't find "mean". But we can give it the same value than ampli, i.e 0 
	if emission=='Abs':
		ampli=-abs(np.nanmin(Flux_range))
		try:
			mean= float(Wave[np.where(Flux==np.nanmin(Flux_range))])
		except:
			mean=ampli

	g_init= models.Lorentz1D(amplitude=ampli, x_0=mean, fwhm=0.01, bounds={"mean": (mean-0.01*mean, mean+0.01*mean),})
	#g_line= g_init+l_init_continuum

	fit_line=fit_g(g_line,Wave,Flux)
	y_line= fit_line(Wave)

	return y_line, g_init, fit_line

#def fit_function_pacs(source_name, Wave_R,Wave_B,Flux_R,Flux_B):			#Calculate the fit function according to the lines for a given source  / instru= PACS/SPIRE_Map/SPIRE_Sparse
def fit_function_pacs(source_name,colour, Wave,Flux):#,Noise):			#Colour=R/B
	my_dict=csv_to_dict('Bulles.csv')
	BB=models.BlackBody1D(temperature=20*u.K)

	l_init_continuum=fit_continuum(Wave,Flux,2)
	#DustModel=custom_model(dust)
	#dust_init=DustModel(amplitude=3E-7,temperature=300,beta=2)
	#g_line=dust_init
	#dust_init=DustModel(amplitude=1E-7,beta=2,temperature=600)
	g_line=l_init_continuum
	#g_line=dust_init
	label=''
	fitted_lines=[]

	if colour=='R':

		#dust_init=DustModel(amplitude=3E-7,temperature=300,beta=2)
		#g_line=dust_init
		if int(my_dict[source_name]['NII_122_em'])==1:
			#NII= Flux[np.where((Wave>121.5) & (Wave<122.5))]
			g_line+=fit_lines_gauss(Wave,Flux,121.9,'Em')
			#if g.amplitude_0>Noise:

			#g_line_R_l+=fit_lines_lorentz(Wave_R,Flux_R,NII,'Em')[1]
			label+=' + NII_122_em'
			fitted_lines.append('NII_122_em')

		if int(my_dict[source_name]['OH_119'])==1:
			#OH_1= Flux[np.where((Wave>119.21) & (Wave<119.25))] 	#OH: 119.23 et 119.44
			#OH_2= Flux[np.where((Wave>119.42) & (Wave<119.46))]

			g_init_OH_1, g_init_OH_2=fit_lines_gauss(Wave,Flux,119.2,'Abs'),fit_lines_gauss(Wave,Flux,119.4,'Abs')
			g_line+=g_init_OH_1+g_init_OH_2
			label+=' + OH_119'
			fitted_lines.append('OH_119')

		if int(my_dict[source_name]['NII_122_abs'])==1:
			#NII_abs= Flux[np.where((Wave>121) & (Wave<123))]
			g_line+=fit_lines_gauss(Wave,Flux,121.9,'Abs')
			label+=' + NII_122_abs'
			fitted_lines.append('NII_122_abs')

		if int(my_dict[source_name]['OI_145'])==1:
			#OI_145=Flux[np.where((Wave>144.8) & (Wave<145.8))]
			g_line+=fit_lines_gauss(Wave,Flux,145.5,'Em')
			label+=' + OI'
			fitted_lines.append('OI_145')

		
		if int(my_dict[source_name]['Thing_123'])==1:
			#Thing_123= Flux[np.where((Wave>122.5) & (Wave<123.5))]
			g_line+=fit_lines_gauss(Wave,Flux,122.5,'Em')
			label+=' + Thing'
			fitted_lines.append('Thing_123')

	if colour=='B':

		#dust_init=DustModel(amplitude=1E-6,temperature=100,beta=2)
		#g_line=dust_init
		if int(my_dict[source_name]['NIII_57'])==1:
			#NIII=Flux[np.where((Wave>55) & (Wave<59))] 

			g_line+=fit_lines_gauss(Wave,Flux,57.35,'Em')
			label+=' + NIII'
			fitted_lines.append('NIII_57')

		if int(my_dict[source_name]['OI_63'])==1:
			#OI_63=Flux[np.where((Wave>62) & (Wave<64))]

			g_line+=fit_lines_gauss(Wave,Flux,63.2,'Em')
			label+=' +OI'
			fitted_lines.append('OI_63')

	


	return g_line, label, fitted_lines #g_line_R_l, g_line_B_l, label_R, label_B 

def fit_one_spectrum_pacs(source_name,colour,mode,row=None,column=None):	#mode=one or mean

	if colour=='R':
		Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R, Flux_red= flux_PACS(source_name,'R')		#Flux_spectrum is a mean spectrum (mean over all pixels), Flux_image= is a mean image (mean spectrum value for each pixel)
		wcs=wcs_R
		
		if mode=='mean':
			#cond=np.where(np.isfinite(Flux_spectrum_R))
			#Wave, Flux=Wave_R[cond], Flux_spectrum_R[cond]
			Wave,Flux=Wave_R, Flux_spectrum_R
			title=': mean spectrum'
			title_spec='Average on the total image'
		if mode=='one':
			if row==None and column==None:			#If row and column not specified, take a ramndom position (row and column)
				cond=np.where(np.isfinite(Flux_red))	#returns a 3D array
				row=random.choice(np.arange(np.min(cond[1]+1),np.max(cond[1])))	#1 is the row dimension and 2 the column dimension
				column=random.choice(np.arange(np.min(cond[2]+1),np.max(cond[2])))
			#print(row,column)	

			#cond=np.where(np.isnan(Flux_red[:,row,column]))
			#if np.shape(cond)[1]!=0:
			#		Flux_red[cond,row,column]=0
			
			cond=np.where(np.isfinite(Flux_red[:,row,column]))
			Wave, Flux=Wave_R[cond], Flux_red[cond,row,column][0]#*1E7
			Noise=np.std(Flux[np.where((Wave>=90)&(Wave<=110))])
			#Wave, Flux=Wave_R, Flux_red[:,row,column]*1E7
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'

		#if np.shape(cond)[1]!=0:
		'''
		g_line, label, fitted_lines = fit_function_pacs(source_name,'R' ,Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		weights=np.zeros_like(Flux)
		weights[Flux>0]=np.sqrt(Flux[Flux>0])
		fit_line=fit_g(g_line,Wave*u.um,Flux,weights=weights)
		y_line= fit_line(Wave)'''
	
		'''
		else:
			g_line, label = 0,0
			fitted_lines=''
			fit_line=0
			y_line= 0'''

		#Residual flux

	if colour=='B':
		
		Flux_image_B, Flux_spectrum_B, Wave_B, wcs_B, Flux_blue= flux_PACS(source_name,'B')
		wcs=wcs_B

		if mode=='mean':
			#cond=np.where(np.isfinite(Flux_spectrum_B))
			#Wave, Flux=Wave_B[cond], Flux_spectrum_B[cond]
			Wave, Flux=Wave_B, Flux_spectrum_B
			title=': mean spectrum'
			title_spec='Average on the total image'
		if mode=='one':
			if row==None and column==None:			#If row and column not specified, take a ramndom position (row and column)
				cond=np.where(np.isfinite(Flux_blue))	#returns a 3D array
				row=random.choice(np.arange(np.min(cond[1]+1),np.max(cond[1])))	#1 is the row dimension and 2 the column dimension
				column=random.choice(np.arange(np.min(cond[2]+1),np.max(cond[2])))
			
			cond=np.where(np.isfinite(Flux_blue[:,row,column]))
			#if np.shape(cond)[1]!=0:					#	on met tout simplement les NaN à 0 pour pouvoir garder la même dimension partout 
			#	Flux_blue[cond,row,column]=0

			#cond=np.where(np.isfinite(Flux_blue[:,row,column]))
			Wave, Flux=Wave_B[cond], Flux_blue[cond,row,column][0]
			Noise=np.std(Flux[np.where((Wave>=55)&(Wave<=75))])
			#Wave, Flux=Wave_B, Flux_blue[:,row,column]*1E7
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'
		
		#if np.shape(cond)[1]!=0:
	if np.shape(cond)[1]!=0:
		g_line, label, fitted_lines =fit_function_pacs(source_name,colour, Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		weights=np.zeros_like(Flux)
		weights[Flux>0]=np.sqrt(Flux[Flux>0])
		fit_line=fit_g(g_line,Wave,Flux)#, weights=weights)
		y_line= fit_line(Wave)
	
		
	else:
		g_line, label = 0,0
		fitted_lines=''
		fit_line=0
		y_line= 0

	Residual=Flux-y_line

	
	return Flux, Wave, Noise, wcs, fit_line, Residual, label, title, title_spec, fitted_lines

def plot_fit_one_spectrum_pacs(source_name,mode,row=None,column=None):
	Flux_R,Wave_R, Noise_R, wcs_R, fit_line_R, Residual_R, label_R, title, title_spec_R, fitted_lines_R=fit_one_spectrum_pacs(source_name,'R',mode,row,column)
	Flux_B,Wave_B, Noise_B, wcs_B, fit_line_B, Residual_B, label_B, title, title_spec_B, fitted_lines_B=fit_one_spectrum_pacs(source_name,'B',mode,row,column)
	
	y_line_R,y_line_B= fit_line_R(Wave_R),fit_line_B(Wave_B)
	'''
	for line in fitted_lines
	if fit_line
	y_line_R,y_line_B= fit_line_R(Wave_R),fit_line_B(Wave_B)'''
	
	#Plots
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(Wave_R, Flux_R, '-b',label='Noise= '+str(Noise_R))
	plt.plot(Wave_B, Flux_B, '-r', label='Noise= '+str(Noise_B))
	plt.plot(Wave_R,y_line_R,'-y',label=str(title_spec_R)+str(label_R[:]))
	plt.plot(Wave_B,y_line_B,'-g',label=str(title_spec_B)+ str(label_B[:]))
	plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.legend()
	plt.title(r'Fit des lines PACS '+str(title))
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.$$\mu$$m^{-1}$)')
	
	# Residual flux plot 
	plt.subplot(1,2,2)
	plt.plot(Wave_R,Residual_R,'-c', label='Gauss')
	plt.plot(Wave_B, Residual_B, '-c')
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.$$\mu$$m^{-1}$)')
	plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
	plt.legend()
	plt.title(r'Residu: Données-Fit des lines'+str(title))

	plt.suptitle(r'Fit des lines PACS pour '+str(source_name)+r' avec un model Gaussien $+$ Polynome deg 2 (continuum)')
	plt.show()
	'''
	plt.figure()
	plt.subplot(1,2,1)'''

	return fit_line_B, fit_line_R

############################################################  SPIRE-lines   #################################################################
############################################################################################################################################


def sinc_function(x,amplitude=1.,x0=0.,sigma=1.):#,bounds=1):
	return amplitude*np.sinc((x-x0)/1.2)				#sigma:instruments High resolution 
	#return amplitude*(np.sin(sigma*np.pi*(x-x0))/(sigma*np.pi*(x-x0)))

def fit_lines_sinc(Wave,Flux,Flux_range,emission):		#Colour=R,B/ Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	fit_g = fitting.LevMarLSQFitter()
	#fit_pol=fitting.LinearLSQFitter()
	l_init_continuum = models.Polynomial1D(degree=7)
	
	SincModel=custom_model(sinc_function)

	if emission=='Em':
		ampli=np.nanmax(Flux_range)
		try:
			x0= float(Wave[np.where(Flux==np.nanmax(Flux_range))])
		except:
			x0=ampli
	if emission=='Abs':
		ampli=-abs(np.nanmin(Flux_range))
		try:
			x0= float(Wave[np.where(Flux==np.nanmin(Flux_range))])
		except:
			x0=ampli
	'''
	try:
		sigma=(Wave[np.where(Flux==np.nanmax(Flux_range))[0][0]]-Wave[np.where(Flux==np.nanmin(Flux_range))[0][0]])*2./np.pi 	 #Calcul approximatif d'un sigma car très sensible aux paramètres initiaux.
	except:
		sigma=1.'''
	sigma=1.2

	g_init= SincModel(amplitude=ampli, x0=x0, sigma=sigma, bounds={"x0": (x0-0.02*x0, x0+0.02*x0)})#"sigma": (0, 1.5*sigma)})
	#g_line=g_init+l_init_continuum

	#fit_line=fit_g(g_line,Wave,Flux, acc=1E-14, epsilon=1E-14, weights=np.sqrt(Flux))
	#y_line= fit_line(Wave)

	return g_init

def fit_function_spire(source_name,colour, Wave, Flux):			#Calculate the fit function according to the lines for a given source  / colour=SSW/SLW

	my_dict=csv_to_dict('Bulles.csv')

	if colour=='SSW':
		l_init_continuum=fit_continuum(Wave,Flux,7)
		g_line=l_init_continuum
		label=''
		fitted_lines=[] 

		if int(my_dict[source_name]['NII_1461'])==1:
			NII= Flux[np.where((Wave>1457) & (Wave<1465))]
			g_line+=fit_lines_sinc(Wave,Flux,NII,'Em')
			label+=' + NII'
			fitted_lines.append('NII_1461')

		if int(my_dict[source_name]['OH_971'])==1:
			OH_971=Flux[np.where((Wave>970) & (Wave<974))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_971,'Abs')
			label+=' + OH'
			fitted_lines.append('OH_971')

		if int(my_dict[source_name]['OH_1033'])==1:
			OH_1033=Flux[np.where((Wave>1031) & (Wave<1034))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_1033,'Abs')
			label+=' + OH'
			fitted_lines.append('OH_1033')

		if int(my_dict[source_name]['CO_98'])==1:
			CO_98=Flux[np.where((Wave>1035) & (Wave<1037))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_98,'Em')
			fitted_lines.append('CO_98')

		if int(my_dict[source_name]['CO_109'])==1:
			CO_109=Flux[np.where((Wave>1149) & (Wave<1153))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_109,'Em')
			fitted_lines.append('CO_109')

		if int(my_dict[source_name]['CO_1110'])==1:
			CO_1110=Flux[np.where((Wave>1265) & (Wave<1269))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_1110,'Em')
			fitted_lines.append('CO_1110')

		if int(my_dict[source_name]['HF_10'])==1:
			HF_10=Flux[np.where((Wave>1230) & (Wave<1234))]
			g_line+=fit_lines_sinc(Wave,Flux,HF_10,'Abs')
			label+=' + HF'
			fitted_lines.append('HF_10')

		if int(my_dict[source_name]['H2O_1113'])==1:
			H2O_1113=Flux[np.where((Wave>1112) & (Wave<1114))]
			g_line+= fit_lines_sinc(Wave,Flux,H2O_1113,'Abs')
			label+=' + H2O'
			fitted_lines.append('H2O_1113')

		if int(my_dict[source_name]['H2O_1115'])==1:
			H2O_1115=Flux[np.where((Wave>1114) & (Wave<1116))]
			g_line+=fit_lines_sinc(Wave,Flux,H2O_1115,'Abs')
			fitted_lines.append('H2O_1115')
	
	if colour=='SLW':
		l_init_continuum=fit_continuum(Wave,Flux,7)
		g_line=l_init_continuum
		label=''
		fitted_lines=[]

		if int(my_dict[source_name]['CI_10'])==1:
			CI_10=Flux[np.where((Wave>490) & (Wave<493.8))]
			g_line+=fit_lines_sinc(Wave,Flux,CI_10,'Em')
			label+=' + CI'
			fitted_lines.append('CI_10')

		if int(my_dict[source_name]['CI_21'])==1:
			CI_21=Flux[np.where((Wave>808.25) & (Wave<810.75))]
			g_line+=fit_lines_sinc(Wave,Flux,CI_21,'Em')
			fitted_lines.append('CI_21')
			
		if int(my_dict[source_name]['CO_43'])==1:
			CO_43=Flux[np.where((Wave>459) & (Wave<463))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_43,'Em')
			label+=' + C0'
			fitted_lines.append('CO_43')

		if int(my_dict[source_name]['CO_54'])==1:
			CO_54=Flux[np.where((Wave>573) & (Wave<579))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_54,'Em')
			fitted_lines.append('CO_54')

		if int(my_dict[source_name]['CO_65'])==1:
			CO_65=Flux[np.where((Wave>689) & (Wave<693))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_65,'Em')
			fitted_lines.append('CO_65')

		if int(my_dict[source_name]['CO_76'])==1:
			CO_76=Flux[np.where((Wave>804) & (Wave<807))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_76,'Em')
			fitted_lines.append('CO_76')
			
		if int(my_dict[source_name]['CO_87'])==1:
			CO_87=Flux[np.where((Wave>919) & (Wave<923))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_87,'Em')
			fitted_lines.append('CO_87')

		if int(my_dict[source_name]['OH_909'])==1:
			OH_909=Flux[np.where((Wave>907) & (Wave<911))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_909,'Abs')
			label+=' + OH'
			fitted_lines.append('OH_909')

		if int(my_dict[source_name]['OH_971'])==1:
			OH_971=Flux[np.where((Wave>969) & (Wave<973))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_971,'Abs')
			label+=' + OH'
			fitted_lines.append('OH_971')

		if int(my_dict[source_name]['CH_835'])==1:
			CH_835=Flux[np.where((Wave>833) & (Wave<837))]
			g_line+=fit_lines_sinc(Wave,Flux,CH_835,'Abs')
			label+=' + CH'
			fitted_lines.append('CH_835')

	return g_line,label, fitted_lines

def fit_one_spectrum_spire_map(source_name,colour,mode,row=None,column=None):

	if colour=='SSW':
		Flux_mean_HR_SSW_image,Flux_mean_HR_SSW_spectrum, Wave_HR_SSW, wcs_HR_SSW, Flux_SSW=plot_image_flux_SPIRE_Map(source_name,'HR','SSW')
		wcs=wcs_HR_SSW
		
		if mode=='mean':
			cond=np.where(np.isfinite(Flux_mean_HR_SSW_spectrum))
			Wave, Flux=Wave_HR_SSW[cond], Flux_mean_HR_SSW_spectrum[cond]
			#Wave, Flux=Wave_HR_SSW, Flux_mean_HR_SSW_spectrum
			title=': mean spectrum'
			title_spec='Average on the total image'
		if mode=='one':
			if row==None and column==None:			#If row and column not specified, take a ramndom position (row and column)
				cond=np.where(np.isfinite(Flux_SSW))	#returns a 3D array
				row=random.choice(np.arange(np.min(cond[1]+1),np.max(cond[1])))	#1 is the row dimension and 2 the column dimension
				column=random.choice(np.arange(np.min(cond[2]+1),np.max(cond[2])))
			#print(row,column)
			#cond=np.where(np.isfinite(Flux_SSW[:,row,column]))
			
			#cond=np.where(np.isnan(Flux_SSW[:,row,column]))
			cond=np.where(np.isfinite(Flux_SSW[:,row,column]))

			#if np.shape(cond)[1]!=0:
			#	Flux_SSW[cond,row,column]=0
			#Wave, Flux=Wave_HR_SSW[cond], Flux_SSW[cond,row,column][0]
			Wave, Flux=Wave_HR_SSW[cond], Flux_SSW[cond,row,column][0]
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'
		'''
		if np.shape(cond)[1]!=0:
			g_line, label, fitted_lines =fit_function_spire(source_name,'SSW', Wave, Flux)
			fit_g = fitting.LevMarLSQFitter()
			fit_line=fit_g(g_line,Wave,Flux)
			y_line= fit_line(Wave)
	
		
		else:
			g_line, label = 0,0
			fitted_lines=''
			fit_line=0
			y_line= 0

		Residual=Flux-y_line'''

	if colour=='SLW':
		Flux_mean_HR_SLW_image,Flux_mean_HR_SLW_spectre, Wave_HR_SLW, wcs_HR_SLW, Flux_SLW=plot_image_flux_SPIRE_Map(source_name,'HR','SLW')
		wcs=wcs_HR_SLW

		if mode=='mean':
			cond=np.where(np.isfinite(Flux_mean_HR_SLW_spectre))
			Wave, Flux=Wave_HR_SLW[cond], Flux_mean_HR_SLW_spectre[cond]
			#Wave, Flux=Wave_HR_SLW, Flux_mean_HR_SLW_spectre
			title=': mean spectrum'
			title_spec='Average on the total image'
		if mode=='one':
			if row==None and column==None:			#If row and column not specified, take a ramndom position (row and column)
				cond=np.where(np.isfinite(Flux_SLW))	#returns a 3D array
				row=random.choice(np.arange(np.min(cond[1]+1),np.max(cond[1])))	#1 is the row dimension and 2 the column dimension
				column=random.choice(np.arange(np.min(cond[2]+1),np.max(cond[2])))
				#print(row,column)
			
			#cond=np.where(np.isnan(Flux_SLW[:,row,column]))
			cond=np.where(np.isfinite(Flux_SLW[:,row,column]))

			#if np.shape(cond)[1]!=0:
			#	Flux_SLW[cond,row,column]=0

			#cond=np.where(np.isfinite(Flux_SLW[:,row,column]))
			#Wave, Flux=Wave_HR_SLW[cond], Flux_SLW[cond,row,column][0]
			Wave, Flux=Wave_HR_SLW[cond], Flux_SLW[cond,row,column][0]
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'

	if np.shape(cond)[1]!=0:
		g_line, label, fitted_lines =fit_function_spire(source_name,colour, Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		fit_line=fit_g(g_line,Wave,Flux)
		y_line= fit_line(Wave)
		
		
	else:
		g_line, label = 0,0
		fitted_lines=''
		fit_line=0
		y_line= 0

	Residual=Flux-y_line

	return Flux,Wave, wcs, fit_line, Residual, label, title, title_spec, fitted_lines


def plot_fit_one_spectrum_spire_map(source_name,mode,row=None,column=None):
	Flux_SSW ,Wave_SSW , wcs_SSW, fit_line_SSW, Residual_SSW, label_SSW, title, title_spec_SSW, fitted_lines =fit_one_spectrum_spire_map(source_name,'SSW',mode,row,column)
	Flux_SLW ,Wave_SLW , wcs_SLW, fit_line_SLW, Residual_SLW, label_SLW, title, title_spec_SLW, fitted_lines =fit_one_spectrum_spire_map(source_name,'SLW',mode,row,column)
	y_line_SSW,y_line_SLW= fit_line_SSW(Wave_SSW),fit_line_SLW(Wave_SLW)

	#Plots
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(Wave_SSW, Flux_SSW, '-b')
	plt.plot(Wave_SLW, Flux_SLW, '-b', label='data')
	plt.plot(Wave_SSW,y_line_SSW,'-',color='orangered',label=str(title_spec_SSW)+' '+ str(label_SSW[:]))
	plt.plot(Wave_SLW,y_line_SLW,'-',color='hotpink',label=str(title_spec_SLW)+' '+ str(label_SLW[:]))
	plt.legend()
	plt.title(r'Fit des lines SPIRE_MAP '+str(title))
	plt.xlabel(r'Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	
	# Residual flux plot 
	plt.subplot(1,2,2)
	plt.plot(Wave_SSW,Residual_SSW,'-c', label='Gauss')
	plt.plot(Wave_SLW, Residual_SLW, '-c')
	plt.xlabel(r'Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	plt.legend()
	plt.title(r'Residu: Données-Fit des lines'+str(title))

	plt.suptitle(r'Fit des lines SPIRE pour '+str(source_name)+r' avec un model Sinc $+$ Polynome deg 6 (continuum)')
	plt.show()

	return fit_line_SSW, fit_line_SLW  


############################################################  SPIRE-SPARSE MAPS  ########################################################
#################################################################################################################################################

from matplotlib.patches import Circle
from astropy.visualization.wcsaxes import SphericalCircle
from astropy import units as u
from astropy.coordinates import Angle

detector_SLW=['SLWA1','SLWA2','SLWA3','SLWB1','SLWB2','SLWB3','SLWB4','SLWC1','SLWC2','SLWC3','SLWC4','SLWC5','SLWD1','SLWD2','SLWD3','SLWD4','SLWE1','SLWE2','SLWE3']
detector_SSW=['SSWA1','SSWA2','SSWA3','SSWA4','SSWB1','SSWB2','SSWB3','SSWB4','SSWB5','SSWC1','SSWC2','SSWC3','SSWC4','SSWC5','SSWC6','SSWD1','SSWD2','SSWD3','SSWD4','SSWD6','SSWD7','SSWE1','SSWE2','SSWE3','SSWE4','SSWE5','SSWE6','SSWF1','SSWF2','SSWF3','SSWF5','SSWG1','SSWG2','SSWG3','SSWG4']
central_detector_SLW='SLWC3'
central_detector_SSW='SSWD4'

def dict_detector():				#Creates a dictionnary with 
	
	SPARSE={}
	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()
	for source in sources:
		if int(my_dict[str(source)]['SPIRE_SPARSE'])==1:
			hdulist=open_source(source,'Sparse')
			SPARSE[str(source)]={}
			for detector in detector_SLW:
				coord=[]
				coord.append(hdulist[str(detector)].header['RA'])
				coord.append(hdulist[str(detector)].header['DEC'])
				SPARSE[str(source)][str(detector)]=coord
			for detector in detector_SSW:
				coord=[]
				coord.append(hdulist[str(detector)].header['RA'])
				coord.append(hdulist[str(detector)].header['DEC'])
				SPARSE[str(source)][str(detector)]=coord


	write_dict(SPARSE,'SPARSE_detectors.npy')

	return SPARSE

def info_detectors(source_name):
	Detectors=np.load(direc1+'SPARSE_detectors.npy',allow_pickle=True).item()
	detector_SLW=list(Detectors[source_name].keys())[:19]
	detector_SSW=list(Detectors[source_name].keys())[19:]
	Beam_SSW=Angle('8"',unit=u.deg).degree					#Beam=16 but we want the radius so 8
	Beam_SLW=Angle('17"',unit=u.deg).degree
	FOV=Angle('1.3m',unit=u.deg).degree

	return Beam_SLW, Beam_SSW, FOV, detector_SLW, detector_SSW


def footprint(source_name):

	Detectors=np.load(direc1+'SPARSE_detectors.npy',allow_pickle=True).item()
	Beam_SLW, Beam_SSW, FOV, detector_SLW, detector_SSW=info_detectors(source_name)

	num_name=source_name[4:8]
	hdu_MIPS24=fits.open(direc+'Bulles_Nicolas/MB'+str(num_name)+'/MIPS24_MB'+str(num_name)+'.fits')

	Flux_range_Mips=np.nanmax(hdu_MIPS24[0].data[100:150,100:150])-np.nanmin(hdu_MIPS24[0].data[100:150,100:150])
	Levels=list(np.arange(np.nanmin(hdu_MIPS24[0].data[100:150,100:150]),np.nanmax(hdu_MIPS24[0].data[100:150,100:150]),Flux_range_Mips/10.))

	plt.figure()
	ax=plt.subplot(1,1,1,projection=WCS(hdu_MIPS24[0].header))
	plt.imshow(hdu_MIPS24[0].data, vmin=np.nanmin(hdu_MIPS24[0].data[100:150,100:150]) , vmax=np.nanmax(hdu_MIPS24[0].data[100:150,100:150]))#[100:150,100:150])
	#plt.colorbar(label="Flux (MJy/sr)")
	#ax.contour(hdu_MIPS24[0].data, levels=Levels, colors='white',linewidths=0.5)#, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=[50,60,70,80,100], colors='white',linewidths=0.5)
	#plt.colorbar()
	plt.xlabel('RA (J2000)')
	plt.ylabel('DEC (J2000)')
	plt.title('Source '+str(source_name)+r' observed with Spitzer MIPS 24 $\mu$m', fontstyle='italic',fontsize=11)
	#plt.grid()

	r = SphericalCircle((Detectors[source_name][central_detector_SLW][0]* u.deg, Detectors[source_name][central_detector_SLW][1] * u.deg), Beam_SLW * u.degree, edgecolor='mediumaquamarine',label='SLW',facecolor='mediumaquamarine',alpha=0.5, linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
	ax.add_patch(r)

	r = SphericalCircle((Detectors[source_name][central_detector_SSW][0]* u.deg, Detectors[source_name][central_detector_SSW][1] * u.deg), Beam_SSW * u.degree, edgecolor='tomato',label='SSW',facecolor='tomato',linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
	ax.add_patch(r)

	for detector in detector_SLW:
		if detector==central_detector_SLW:
			continue
		r = SphericalCircle((Detectors[source_name][detector][0]* u.deg, Detectors[source_name][detector][1] * u.deg), Beam_SLW * u.degree, edgecolor='mediumaquamarine',facecolor='none', linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
		ax.add_patch(r)
		#plt.text(Detectors[source_name][detector][0]+0.001, Detectors[source_name][detector][1]-0.001, str(detector[3:]), color='yellow',transform=ax.get_transform('fk5'))
		plt.text(Detectors[source_name][detector][0], Detectors[source_name][detector][1], str(detector[3:]), color='mediumaquamarine',horizontalalignment='center',verticalalignment='center',transform=ax.get_transform('fk5'))
	#plt.legend('SLW')

	for detector in detector_SSW:
		if detector==central_detector_SLW:
			continue
		r = SphericalCircle((Detectors[source_name][detector][0]* u.deg, Detectors[source_name][detector][1] * u.deg), Beam_SSW * u.degree, edgecolor='tomato',facecolor='none',linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
		ax.add_patch(r)
		plt.text(Detectors[source_name][detector][0], Detectors[source_name][detector][1], str(detector[3:]), color='tomato',horizontalalignment='center',verticalalignment='center',alpha=0.8,transform=ax.get_transform('fk5'))
	
	#plt.legend('SSW')
	r = SphericalCircle((Detectors[source_name][central_detector_SLW][0]* u.deg, Detectors[source_name][central_detector_SLW][1] * u.deg), FOV * u.degree, label='FOV',edgecolor='seagreen',linewidth=2,facecolor='none',transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
	ax.add_patch(r)
	plt.legend()
	plt.suptitle('Footprint of the SPIRE Sparse FTS bolometers for source '+ str(source_name))

	plt.show()

	return


def fit_one_spectrum_spire_sparse(source_name,color,Detector=None):#,Detector_SLW=None):

	#print(Detector)
	if color=='SSW':
		hdulist_SSW= open_source(source_name,'Sparse','SSW')
		if Detector is None:						#On prend le détecteur central
			#hdulist_SSW= open_source(source_name,'Sparse','SSW')
			Wave_SSW,Flux_SSW= hdulist_SSW['SSWD4'].data['wave'], hdulist_SSW['SSWD4'].data['flux']
			#cond=np.where(np.isnan(Flux_SSW))
			cond=np.where(np.isfinite(Flux_SSW))
			
			#if np.shape(cond)[1]!=0:
				#Flux_SSW[cond]=0
			
			Wave, Flux=Wave_SSW[cond], Flux_SSW[cond]#*1E20
			title=': one spectrum' 
			title_spec='detector = '+str(central_detector_SSW)

		if Detector is not None:
			Wave_SSW,Flux_SSW= hdulist_SSW[str(Detector)].data['wave'], hdulist_SSW[str(Detector)].data['flux']
			#cond=np.where(np.isnan(Flux_SSW))
			cond=np.where(np.isfinite(Flux_SSW))

			#if np.shape(cond)[1]!=0:
			#	Flux_SSW[cond]=0
			
			Wave, Flux=Wave_SSW[cond], Flux_SSW[cond]
			title=': one spectrum' 
			title_spec='detector = '+str(Detector)

		g_line, label, fitted_lines =fit_function_spire(source_name,'SSW', Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		fit_line=fit_g(g_line,Wave,Flux)#,acc=1E-14, epsilon=1E-14, weights=np.sqrt(Flux))
		y_line= fit_line(Wave)

		Residual=Flux-y_line

	if color=='SLW':
		hdulist_SLW= open_source(source_name,'Sparse','SLW')
		if Detector is None:						#On prend le détecteur central
			#hdulist_SSW= open_source(source_name,'Sparse','SSW')
			Wave_SLW,Flux_SLW= hdulist_SLW['SLWC3'].data['wave'], hdulist_SLW['SLWC3'].data['flux']
			#cond=np.where(np.isnan(Flux_SLW))
			cond=np.where(np.isfinite(Flux_SLW))

			#if np.shape(cond)[1]!=0:
			#	Flux_SLW[cond]=0
			
			Wave, Flux=Wave_SLW[cond], Flux_SLW[cond]#*1E20
			title=': one spectrum' 
			title_spec='detector = '+str(central_detector_SLW)

		if Detector is not None:
			Wave_SLW,Flux_SLW= hdulist_SLW[str(Detector)].data['wave'], hdulist_SLW[str(Detector)].data['flux']
			#cond=np.where(np.isnan(Flux_SLW))
			cond=np.where(np.isfinite(Flux_SLW))

			#if np.shape(cond)[1]!=0:
			#	Flux_SLW[cond]=0
			
			Wave, Flux=Wave_SLW[cond], Flux_SLW[cond]
			title=': one spectrum' 
			title_spec='detector = '+str(Detector)

		g_line, label, fitted_lines =fit_function_spire(source_name,'SLW', Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		fit_line=fit_g(g_line,Wave,Flux)#, acc=1E-14, epsilon=1E-14, weights=np.sqrt(Flux))
		y_line= fit_line(Wave)

		Residual=Flux-y_line
	

	return Flux,Wave, fit_line, Residual, label, title, title_spec, fitted_lines


def plot_fit_one_spectrum_spire_sparse(source_name,Detector_SSW=None, Detector_SLW=None):
	#print(detector)
	Flux_SSW ,Wave_SSW ,fit_line_SSW, Residual_SSW, label_SSW, title, title_spec_SSW, fitted_lines =fit_one_spectrum_spire_sparse(source_name,'SSW')#Detector)
	Flux_SLW ,Wave_SLW , fit_line_SLW, Residual_SLW, label_SLW, title, title_spec_SLW, fitted_lines =fit_one_spectrum_spire_sparse(source_name,'SLW')#Detector_SSW, Detector_SLW)
	
	y_line_SSW,y_line_SLW= fit_line_SSW(Wave_SSW),fit_line_SLW(Wave_SLW)

	Residu_SLW=np.sum(Residual_SLW**2)
	Residu_SSW=np.sum(Residual_SSW**2)

	#Plots
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(Wave_SSW, Flux_SSW, '-b')
	plt.plot(Wave_SLW, Flux_SLW, '-b', label='data')
	plt.plot(Wave_SSW,y_line_SSW,'-',color='orangered',label=str(title_spec_SSW)+' '+ str(label_SSW[:]))
	plt.plot(Wave_SLW,y_line_SLW,'-',color='hotpink',label=str(title_spec_SLW)+' '+ str(label_SLW[:]))
	plt.legend()
	plt.title(r'Fit des lines SPIRE_MAP '+str(title))
	plt.xlabel(r'Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	
	# Residual flux plot 
	plt.subplot(1,2,2)
	plt.plot(Wave_SSW,Residual_SSW,'-c', label='Gauss')
	plt.plot(Wave_SLW, Residual_SLW, '-c')
	plt.xlabel(r'Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	plt.legend()
	plt.title(r'Residu: Données-Fit des lines'+str(title))

	plt.suptitle(r'Fit des lines SPIRE pour '+str(source_name)+r' avec un model Sinc $+$ Polynome deg 6 (continuum)')
	plt.show()

	return fit_line_SSW, fit_line_SLW, Residu_SLW, Residu_SSW


############################################################  MAPS PACS/SPIRE   #################################################################
#################################################################################################################################################

def fit_map_spectrum(source_name,instru): #instru=PACS, SPIRE_MAP
	
	if instru=='PACS':
		fit_R, Res_R=fit_one_spectrum_pacs(source_name,'R','mean')[3],fit_one_spectrum_pacs(source_name,'R','mean')[4]
		fit_B, Res_B=fit_one_spectrum_pacs(source_name,'B','mean')[3],fit_one_spectrum_pacs(source_name,'B','mean')[4]
		

		Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R, Flux_red= flux_PACS(source_name,'R')
		Flux_image_B, Flux_spectrum_B, Wave_B, wcs_B, Flux_blue= flux_PACS(source_name,'B')
	
		fit_line_R=np.zeros((np.shape(Flux_image_R)[0],np.shape(Flux_image_R)[1],len(fit_R.parameters)))
		fit_line_B=np.zeros((np.shape(Flux_image_B)[0],np.shape(Flux_image_B)[1],len(fit_B.parameters)))
		Residual_R=np.zeros((np.shape(Flux_image_R)[0],np.shape(Flux_image_R)[1],len(Res_R)))
		Residual_B=np.zeros((np.shape(Flux_image_B)[0],np.shape(Flux_image_B)[1],len(Res_B)))
		
		
		#Retrieve all the fit parameters for the R-Band
		for i in range(np.shape(Flux_image_R)[0]):
			for j in range(np.shape(Flux_image_R)[1]):
				Flux,Wave_R, wcs_R, fit_line, Residual, label, title, title_spec, fitted_lines=fit_one_spectrum_pacs(source_name,'R','one',i,j)
				
				if fit_line!=0:
					Residual_R[i,j,:]=Residual
					param_names=fit_line.param_names
					fit_line_R[i,j,:]=fit_line.parameters
					fitted_lines_R=fitted_lines
				else:
					fit_line_R[i,j,:]=np.nan

		#Dictionnary where we keep what we need. In each key = Line, we save the flux for this special line for each pixel
		Source_R={}
		Source_R['Object']=str(source_name)
		Source_R['WCS']=wcs_R
		Source_R['Wave']=Wave_R
		Source_R['Param_names']=param_names
		Source_R['Residual']=Residual_R
		Source_R['Fit']=fit_line_R

		a=0
		for line in fitted_lines_R:
			#print(line)
			a=fitted_lines_R.index(line)
			#print(a)
			Line_flux_R=np.zeros_like(Flux_image_R)
			for i in range(len(fit_line_R[:,0,0])):
				for j in range(len(fit_line_R[0,:,0])):
					if np.isnan(fit_line_R[i,j,0]):
						Line_flux_R[i,j]=0
					else:
						amplitude, mean, std=(fit_line_R[i,j,:][3*a+3]),(fit_line_R[i,j,:][3*a+4]),abs(fit_line_R[i,j,:][3*a+5])		#Poly degre 2 #On retrouve les paramètres A et sigma pour le calcul du flux des lines
						Line_flux_R[i,j]=(amplitude*std*np.sqrt(2*np.pi))
				
			
			Source_R[str(line)]=Line_flux_R


		#Retrieve all the fit parameters for the B-Band
		for i in range(np.shape(Flux_image_B)[0]):
			for j in range(np.shape(Flux_image_B)[1]):
				Flux,Wave_B, wcs_B, fit_line, Residual, label, title, title_spec, fitted_lines=fit_one_spectrum_pacs(source_name,'B','one',i,j)

				if fit_line!=0:
					Residual_B[i,j,:]=Residual
					param_names=fit_line.param_names
					fit_line_B[i,j,:]=fit_line.parameters
					fitted_lines_B=fitted_lines
				else:
					fit_line_B[i,j,:]=np.nan

		print(fit_line_B)

		Source_B={}
		Source_B['Object']=str(source_name)
		Source_B['WCS']=wcs_B
		Source_B['Wave']=Wave_B
		Source_B['Param_names']=param_names
		Source_B['Residual']=Residual_B
		Source_B['Fit']=fit_line_B
		

		a=0
		for line in fitted_lines_B:
			a=fitted_lines_B.index(line)
			Line_flux_B=np.zeros_like(Flux_image_B)
			for i in range(len(fit_line_B[:,0,0])):
				for j in range(len(fit_line_B[0,:,0])):
					if np.isnan(fit_line_B[i,j,0]):
						Line_flux_B[i,j]=0
					else:
						amplitude, mean, std=(fit_line_B[i,j,:][3*a+3]),(fit_line_B[i,j,:][3*a+4]),abs(fit_line_B[i,j,:][3*a+5])		#On retrouve les paramètres A et sigma pour le calcul du flux des lines
						
						Line_flux_B[i,j]=(amplitude*std*np.sqrt(2*np.pi))

			Source_B[str(line)]=Line_flux_B
		
		write_dict(Source_R,str(source_name)+'_R.npy')
		write_dict(Source_B,str(source_name)+'_B.npy')

		return  fit_line_R, fit_line_B, Source_R, Source_B

	if instru=='SPIRE_MAP':
		init_fit_SSW, Res_SSW=fit_one_spectrum_spire_map(source_name,'SSW','mean')[3],fit_one_spectrum_spire_map(source_name,'SSW','mean')[4]
		init_fit_SLW, Res_SLW=fit_one_spectrum_spire_map(source_name,'SLW','mean')[3],fit_one_spectrum_spire_map(source_name,'SLW','mean')[4]

		Flux_mean_HR_SSW_image,Flux_mean_HR_SSW_spectrum, Wave_HR_SSW, wcs_HR_SSW, Flux_SSW=plot_image_flux_SPIRE_Map(source_name,'HR','SSW')
		Flux_mean_HR_SLW_image,Flux_mean_HR_SLW_spectre, Wave_HR_SLW, wcs_HR_SLW, Flux_SLW=plot_image_flux_SPIRE_Map(source_name,'HR','SLW')
	
		fit_line_SSW=np.zeros((np.shape(Flux_mean_HR_SSW_image)[0],np.shape(Flux_mean_HR_SSW_image)[1],len(init_fit_SSW.parameters)))
		fit_line_SLW=np.zeros((np.shape(Flux_mean_HR_SLW_image)[0],np.shape(Flux_mean_HR_SLW_image)[1],len(init_fit_SLW.parameters)))
		Residual_SSW=np.zeros((np.shape(Flux_mean_HR_SSW_image)[0],np.shape(Flux_mean_HR_SSW_image)[1],len(Res_SSW)))
		Residual_SLW=np.zeros((np.shape(Flux_mean_HR_SLW_image)[0],np.shape(Flux_mean_HR_SLW_image)[1],len(Res_SLW)))
		

		#Retrieve all the fit parameters for the SSW-Band
		for i in range(np.shape(Flux_mean_HR_SSW_image)[0]):
			for j in range(np.shape(Flux_mean_HR_SSW_image)[1]):
				Flux,Wave_SSW, wcs_SSW, fit_line, Residual, label, title, title_spec, fitted_lines=fit_one_spectrum_spire_map(source_name,'SSW','one',i,j)
				if fit_line!=0:
					Residual_SSW[i,j,:]=Residual
					param_names=fit_line.param_names
					fit_line_SSW[i,j,:]=fit_line.parameters
					fitted_lines_SSW=fitted_lines
				else:
					fit_line_SSW[i,j,:]=np.nan

		Source_SSW={}
		Source_SSW['Object']=str(source_name)
		Source_SSW['WCS']=wcs_SSW
		Source_SSW['Wave']=Wave_SSW
		Source_SSW['Param_names']=param_names
		Source_SSW['Residual']=Residual_SSW
		Source_SSW['Fit']=fit_line_SSW

		a=0
		for line in fitted_lines_SSW:
			#print(line)
			a=fitted_lines_SSW.index(line)
			#print(a)
			Line_flux_SSW=np.zeros_like(Flux_mean_HR_SSW_image)
			#param_names=fit_line_SSW.param_names
			for i in range(len(fit_line_SSW[:,0,0])):
				for j in range(len(fit_line_SSW[0,:,0])):
					if np.isnan(fit_line_SSW[i,j,0]):
						Line_flux_SSW[i,j]=0
					else:
						amplitude, mean, std=(fit_line_SSW[i,j,:][3*a+8]),(fit_line_SSW[i,j,:][3*a+9]),abs(fit_line_SSW[i,j,:][3*a+10])		#Poly degre 7 #On retrouve les paramètres A et sigma pour le calcul du flux des lines
						Line_flux_SSW[i,j]=(amplitude*std*1E9*np.pi)
				
			
			Source_SSW[str(line)]=Line_flux_SSW


		for i in range(np.shape(Flux_mean_HR_SLW_image)[0]):
			for j in range(np.shape(Flux_mean_HR_SLW_image)[1]):
				Flux,Wave_SLW, wcs_SLW, fit_line, Residual, label, title, title_spec, fitted_lines=fit_one_spectrum_spire_map(source_name,'SLW','one',i,j)
				if fit_line!=0:
					Residual_SLW[i,j,:]=Residual
					param_names=fit_line.param_names
					fit_line_SLW[i,j,:]=fit_line.parameters
					fitted_lines_SLW=fitted_lines
				else:
					fit_line_SLW[i,j,:]=np.nan

		Source_SLW={}
		Source_SLW['Object']=str(source_name)
		Source_SLW['WCS']=wcs_SLW
		Source_SLW['Wave']=Wave_SLW
		Source_SLW['Param_names']=param_names
		Source_SLW['Residual']=Residual_SLW
		Source_SLW['Fit']=fit_line_SLW

		a=0
		for line in fitted_lines_SLW:
			#print(line)
			a=fitted_lines_SLW.index(line)
			#print(a)
			Line_flux_SLW=np.zeros_like(Flux_mean_HR_SLW_image)
			#param_names=fit_line_SSW.param_names
			for i in range(len(fit_line_SLW[:,0,0])):
				for j in range(len(fit_line_SLW[0,:,0])):
					if np.isnan(fit_line_SLW[i,j,0]):
						Line_flux_SLW[i,j]=0
					else:
						amplitude, mean, std=(fit_line_SLW[i,j,:][3*a+8]),(fit_line_SLW[i,j,:][3*a+9]),abs(fit_line_SLW[i,j,:][3*a+10])		#Poly degre 7 so we take the 8th parameter as the 1st amplitude, and we have 3 parameters so the step is *3 each time. #On retrouve les paramètres A et sigma pour le calcul du flux des lines
						Line_flux_SLW[i,j]=(amplitude*std*1E9*np.pi)
				
			
			Source_SLW[str(line)]=Line_flux_SLW

		write_dict(Source_SSW,str(source_name)+'_SSW.npy')
		write_dict(Source_SLW,str(source_name)+'_SLW.npy')

		return fit_line_SSW,fit_line_SLW, Source_SSW, Source_SLW

	if instru=='SPIRE_SPARSE':
		init_fit_SSW, Res_SSW=fit_one_spectrum_spire_sparse(source_name,'SSW')[2],fit_one_spectrum_spire_sparse(source_name,'SSW')[3]
		init_fit_SLW, Res_SLW=fit_one_spectrum_spire_sparse(source_name,'SLW')[2],fit_one_spectrum_spire_sparse(source_name,'SLW')[3]

		fit_line_SSW=np.zeros((len(detector_SSW),len(init_fit_SSW.parameters)))
		fit_line_SLW=np.zeros((len(detector_SLW),len(init_fit_SLW.parameters)))
		Residual_SSW=np.zeros((len(detector_SSW),len(Res_SSW)))
		Residual_SLW=np.zeros((len(detector_SLW),len(Res_SLW)))
		#Flux_SSW=np.zeros((len(detector_SSW),len(init_fit_SSW.parameters)))
		
		Source_SSW={}
		Source_SSW['Object']=str(source_name)
	
		
		for detector in detector_SSW:
			i=detector_SSW.index(detector)
			#print (i)
			#Wave_SSW,Flux_SSW= hdulist_SSW['SSWD4'].data['wave'], hdulist_SSW['SSWD4'].data['flux']
			Flux,Wave_SSW, fit_line, Residual, label, title, title_spec, fitted_lines=fit_one_spectrum_spire_sparse(source_name,'SSW',detector)
			param_names=fit_line.param_names
			#Residual_SSW.append(Residual)
			fitted_lines_SSW=fitted_lines
			if fit_line!=0:
				Residual_SSW[i,:]=Residual
				#param_names=fit_line.param_names
				fit_line_SSW[i,:]=fit_line.parameters
				#fitted_lines_SSW=fitted_lines
			else:
				fit_line_SSW[i,:]=np.nan
		
		print(fit_line_SSW)
		Source_SSW['Wave']=Wave_SSW
		Source_SSW['Param_names']=param_names
		Source_SSW['Detectors']=detector_SSW
		Source_SSW['Residual']=Residual_SSW
		Source_SSW['Fit']=fit_line_SSW

		a=0
		for line in fitted_lines_SSW:
			#print(line)
			a=fitted_lines_SSW.index(line)
			#print(a)
			Line_flux_SSW=[]
			#param_names=fit_line_SSW.param_names
			for i in range(len(fit_line_SSW[:,0])):
					if np.isnan(fit_line_SSW[i,0]):
						Line_flux_SSW.append(0)
					else:
						amplitude, mean, std=(fit_line_SSW[i,:][3*a+8]),(fit_line_SSW[i,:][3*a+9]),abs(fit_line_SSW[i,:][3*a+10])		#Poly degre 7 #On retrouve les paramètres A et sigma pour le calcul du flux des lines
						Line_flux_SSW.append(amplitude*std*1E9*np.pi)				#Ampli in W.m-2.Hz-1.sr-1 and std en GHz so : *1E9
				
			
			Source_SSW[str(line)]= Line_flux_SSW
		
		Source_SLW={}
		Source_SLW['Object']=str(source_name)
	
		
		for detector in detector_SLW:
			i=detector_SLW.index(detector)
			print (i)
			#Wave_SSW,Flux_SSW= hdulist_SSW['SSWD4'].data['wave'], hdulist_SSW['SSWD4'].data['flux']
			Flux,Wave_SLW, fit_line, Residual, label, title, title_spec, fitted_lines=fit_one_spectrum_spire_sparse(source_name,'SLW',detector)
			param_names=fit_line.param_names
			#Residual_SSW.append(Residual)
			fitted_lines_SLW=fitted_lines
			if fit_line!=0:
				Residual_SLW[i,:]=Residual
				#param_names=fit_line.param_names
				fit_line_SLW[i,:]=fit_line.parameters
				#fitted_lines_SSW=fitted_lines
			else:
				fit_line_SLW[i,:]=np.nan
		
		print(fit_line_SLW)
		Source_SLW['Wave']=Wave_SLW
		Source_SLW['Param_names']=param_names
		Source_SLW['Detectors']=detector_SLW
		Source_SLW['Residual']=Residual_SLW
		Source_SLW['Fit']=fit_line_SLW

		a=0
		for line in fitted_lines_SLW:
			#print(line)
			a=fitted_lines_SLW.index(line)
			#print(a)
			Line_flux_SLW=[]
			#param_names=fit_line_SSW.param_names
			for i in range(len(fit_line_SLW[:,0])):
					if np.isnan(fit_line_SLW[i,0]):
						Line_flux_SLW.append(0)
					else:
						amplitude, mean, std=(fit_line_SLW[i,:][3*a+8]),(fit_line_SLW[i,:][3*a+9]),abs(fit_line_SLW[i,:][3*a+10])		#Poly degre 7 #On retrouve les paramètres A et sigma pour le calcul du flux des lines
						Line_flux_SLW.append(amplitude*std*1E9*np.pi)
				
			
			Source_SLW[str(line)]= Line_flux_SLW

	
		write_dict(Source_SSW,str(source_name)+'_SSW.npy')
		write_dict(Source_SLW,str(source_name)+'_SLW.npy')

		return fit_line_SSW,fit_line_SLW, Source_SSW, Source_SLW 



############################################################  MAPS-PACS/SPIRE - BUBBLES   ########################################################
#################################################################################################################################################

def fit_Bubbles():
	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()

	for source in sources:
		
		#fit_map_spectrum(source,'PACS')

		'''
		if int(my_dict[str(source)]['SPIRE_SPARSE'])==1 :
			fit_map_spectrum(source,'PACS')
			fit_map_spectrum(source,'SPIRE_SPARSE')

		if int(my_dict[str(source)]['SPIRE_MAP'])==1:
			fit_map_spectrum(source,'SPIRE_MAP')
			fit_map_spectrum(source,'PACS')'''

		
		if int(my_dict[str(source)]['SPIRE_MAP'])==0 and int(my_dict[str(source)]['SPIRE_SPARSE'])==0:
			fit_map_spectrum(source,'PACS')











	#return
