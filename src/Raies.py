import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import  GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling import models, fitting
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
	l_init_continuum = models.Polynomial1D(degree=Deg)
	continuum_poly=fit_pol(l_init_continuum, Wave, Flux)
	y_continuum_poly= continuum_poly(Wave)

	return l_init_continuum, y_continuum_poly

def fit_lines_gauss(Wave,Flux,Flux_range,emission):		#Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	fit_g = fitting.LevMarLSQFitter()
	fit_pol=fitting.LinearLSQFitter()
	l_init_continuum = fit_continuum(Wave,Flux,2)[0]
	
	if emission=='Em':
		ampli=np.nanmax(Flux_range)
		try:
			mean= float(Wave[np.where(Flux==np.nanmax(Flux_range))])
		except:
			mean=ampli
	if emission=='Abs':
		ampli=-abs(np.nanmin(Flux_range))
		try:
			mean= float(Wave[np.where(Flux==np.nanmin(Flux_range))])
		except:
			mean=ampli

	g_init= models.Gaussian1D(amplitude=ampli, mean=mean, stddev=0.01, bounds={"mean": (mean-0.01*mean, mean+0.01*mean)})
	g_line= g_init+l_init_continuum

	fit_line=fit_g(g_line,Wave,Flux)
	y_line= fit_line(Wave)

	return y_line, g_init, fit_line

def fit_lines_lorentz(Wave,Flux,Flux_range,emission):		#Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	fit_g = fitting.LevMarLSQFitter()
	fit_pol=fitting.LinearLSQFitter()
	l_init_continuum = fit_continuum(Wave,Flux,2)[0]
	
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

	g_init= models.Lorentz1D(amplitude=ampli, x_0=mean, fwhm=0.01, bounds={"mean": (mean-0.01*mean, mean+0.01*mean)})
	g_line= g_init+l_init_continuum

	fit_line=fit_g(g_line,Wave,Flux)
	y_line= fit_line(Wave)

	return y_line, g_init, fit_line

#def fit_function_pacs(source_name, Wave_R,Wave_B,Flux_R,Flux_B):			#Calculate the fit function according to the lines for a given source  / instru= PACS/SPIRE_Map/SPIRE_Sparse
def fit_function_pacs(source_name,colour, Wave,Flux):			#Colour=R/B
	my_dict=csv_to_dict('Bulles.csv')

	if colour=='R':
		l_init_continuum=fit_continuum(Wave,Flux,2)[0]
		g_line=l_init_continuum
		label=''
		fitted_lines=[]

		if int(my_dict[source_name]['NII_122_em'])==1:
			NII= Flux[np.where((Wave>121.5) & (Wave<122.5))]
			g_line+=fit_lines_gauss(Wave,Flux,NII,'Em')[1]
			#g_line_R_l+=fit_lines_lorentz(Wave_R,Flux_R,NII,'Em')[1]
			label+=' + NII'
			fitted_lines.append('NII_122_em')

		if int(my_dict[source_name]['OH_119'])==1:
			OH_1= Flux[np.where((Wave>119.21) & (Wave<119.25))] 	#OH: 119.23 et 119.44
			OH_2= Flux[np.where((Wave>119.42) & (Wave<119.46))]

			g_init_OH_1, g_init_OH_2=fit_lines_gauss(Wave,Flux,OH_1,'Abs')[1],fit_lines_gauss(Wave,Flux,OH_2,'Abs')[1]
			g_line+=g_init_OH_1+g_init_OH_2
			label+=' + OH_119'
			fitted_lines.append('OH_119')

		if int(my_dict[source_name]['NII_122_abs'])==1:
			NII_abs= Flux[np.where((Wave>121) & (Wave<123))]
			g_line+=fit_lines_gauss(Wave,Flux,NII_abs,'Abs')[1]
			label+=' + NII_122'
			fitted_lines.append('NII_122_abs')

		if int(my_dict[source_name]['OI_145'])==1:
			OI_145=Flux[np.where((Wave>144.8) & (Wave<145.8))]
			g_line+=fit_lines_gauss(Wave,Flux,OI_145,'Em')[1]
			label+=' + OI'
			fitted_lines.append('OI_145')

		
		if int(my_dict[source_name]['Thing_123'])==1:
			Thing_123= Flux[np.where((Wave>122.5) & (Wave<123.5))]
			g_line+=fit_lines_gauss(Wave,Flux,Thing_123,'Em')[1]
			label+=' + Thing'
			fitted_lines.append('Thing_123')

	if colour=='B':
		l_init_continuum=fit_continuum(Wave,Flux,2)[0]
		g_line=l_init_continuum
		label=''	
		fitted_lines=[]	

		if int(my_dict[source_name]['NIII_57'])==1:
			NIII=Flux[np.where((Wave>55) & (Wave<59))] 

			g_line+=fit_lines_gauss(Wave,Flux,NIII,'Em')[1]
			label+=' + NIII'
			fitted_lines.append('NIII_57')

		if int(my_dict[source_name]['OI_63'])==1:
			OI_63=Flux[np.where((Wave>62) & (Wave<64))]
			g_line+=fit_lines_gauss(Wave,Flux,OI_63,'Em')[1]
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

			cond=np.where(np.isnan(Flux_red[:,row,column]))
			if np.shape(cond)[1]!=0:
				Flux_red[cond,row,column]=0
			
			#cond=np.where(np.isfinite(Flux_red[:,row,column]))
			#Wave, Flux=Wave_R[cond], Flux_red[cond,row,column][0]
			Wave, Flux=Wave_R, Flux_red[:,row,column]
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'

		#if np.shape(cond)[1]!=0:
		g_line, label, fitted_lines = fit_function_pacs(source_name,'R' ,Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		fit_line=fit_g(g_line,Wave,Flux)
		y_line= fit_line(Wave)
	
		'''
		else:
			g_line, label = 0,0
			fitted_lines=''
			fit_line=0
			y_line= 0'''

		#Residual flux
		Residual=Flux-y_line

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
			
			cond=np.where(np.isnan(Flux_blue[:,row,column]))
			if np.shape(cond)[1]!=0:					#	on met tout simplement les NaN à 0 pour pouvoir garder la même dimension partout 
				Flux_blue[cond,row,column]=0

			#cond=np.where(np.isfinite(Flux_blue[:,row,column]))
			#Wave, Flux=Wave_B[cond], Flux_blue[cond,row,column][0]
			Wave, Flux=Wave_B, Flux_blue[:,row,column]
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'
		
		#if np.shape(cond)[1]!=0:
		g_line, label, fitted_lines = fit_function_pacs(source_name,'B' ,Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		fit_line=fit_g(g_line,Wave,Flux)
		y_line= fit_line(Wave)
	
		'''
		else:
			g_line, label = 0,0
			fitted_lines=''
			fit_line=0
			y_line= 0'''

		#Residual flux
		Residual=Flux-y_line

	return Flux, Wave, wcs, fit_line, Residual, label, title, title_spec, fitted_lines

def plot_fit_one_spectrum_pacs(source_name,mode,row=None,column=None):
	Flux_R,Wave_R, wcs_R, fit_line_R, Residual_R, label_R, title, title_spec_R, fitted_lines_R=fit_one_spectrum_pacs(source_name,'R',mode,row=None,column=None)
	Flux_B,Wave_B, wcs_B, fit_line_B, Residual_B, label_B, title, title_spec_B, fitted_lines_B=fit_one_spectrum_pacs(source_name,'B',mode,row=None,column=None)
	y_line_R,y_line_B= fit_line_R(Wave_R),fit_line_B(Wave_B)
	
	#Plots
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(Wave_R, Flux_R, '-b')
	plt.plot(Wave_B, Flux_B, '-b', label='data')
	plt.plot(Wave_R,y_line_R,'-y',label=str(title_spec_R)+str(label_R[:]))
	plt.plot(Wave_B,y_line_B,'-r',label=str(title_spec_B)+ str(label_B[:]))
	plt.legend()
	plt.title(r'Fit des lines PACS '+str(title))
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($MJy.sr^{-1}$)')
	
	# Residual flux plot 
	plt.subplot(1,2,2)
	plt.plot(Wave_R,Residual_R,'-c', label='Gauss')
	plt.plot(Wave_B, Residual_B, '-c')
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($MJy.sr^{-1}$)')
	plt.legend()
	plt.title(r'Residu: Données-Fit des lines'+str(title))

	plt.suptitle(r'Fit des lines PACS pour '+str(source_name)+r' avec un model Gaussien $+$ Polynome deg 2 (continuum)')
	plt.show()

	return 

############################################################  SPIRE-lines   #################################################################
############################################################################################################################################


def sinc_function(x,amplitude=1.,x0=0.,sigma=1.):#,bounds=1):
	return amplitude*np.sinc((x-x0)/sigma)
	#return amplitude*(np.sin(sigma*np.pi*(x-x0))/(sigma*np.pi*(x-x0)))

def fit_lines_sinc(Wave,Flux,Flux_range,emission):		#Colour=R,B/ Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	fit_g = fitting.LevMarLSQFitter()
	fit_pol=fitting.LinearLSQFitter()
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
	
	try:
		sigma=(Wave[np.where(Flux==np.nanmax(Flux_range))[0][0]]-Wave[np.where(Flux==np.nanmin(Flux_range))[0][0]])*2./np.pi 	 #Calcul approximatif d'un sigma car très sensible aux paramètres initiaux.
	except:
		sigma=2.

	g_init= SincModel(amplitude=ampli, x0=x0, sigma=sigma, bounds={"x0": (x0-0.02*x0, x0+0.02*x0)})
	g_line=g_init+l_init_continuum

	fit_line=fit_g(g_line,Wave,Flux)
	y_line= fit_line(Wave)

	return y_line, g_init, fit_line

def fit_function_spire_map(source_name,colour, Wave, Flux):			#Calculate the fit function according to the lines for a given source  / colour=SSW/SLW

	my_dict=csv_to_dict('Bulles.csv')

	if colour=='SSW':
		l_init_continuum=fit_continuum(Wave,Flux,7)[0]
		g_line=l_init_continuum
		label=''
		fitted_lines=[] 

		if int(my_dict[source_name]['NII_1461'])==1:
			NII= Flux[np.where((Wave>1457) & (Wave<1465))]
			g_line+=fit_lines_sinc(Wave,Flux,NII,'Em')[1]
			label+=' + NII'
			fitted_lines.append('NII_1461')

		if int(my_dict[source_name]['OH_971'])==1:
			OH_971=Flux[np.where((Wave>970) & (Wave<974))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_971,'Abs')[1]
			label+=' + OH'
			fitted_lines.append('OH_971')

		if int(my_dict[source_name]['OH_1033'])==1:
			OH_1033=Flux[np.where((Wave>1031) & (Wave<1034))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_1033,'Abs')[1]
			label+=' + OH'
			fitted_lines.append('OH_1033')

		if int(my_dict[source_name]['CO_98'])==1:
			CO_98=Flux[np.where((Wave>1033) & (Wave<1037))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_98,'Em')[1]
			fitted_lines.append('CO_98')

		if int(my_dict[source_name]['CO_109'])==1:
			CO_109=Flux[np.where((Wave>1149) & (Wave<1153))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_109,'Em')[1]
			fitted_lines.append('CO_109')

		if int(my_dict[source_name]['CO_1110'])==1:
			CO_1110=Flux[np.where((Wave>1265) & (Wave<1269))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_1110,'Em')[1]
			fitted_lines.append('CO_1110')

		if int(my_dict[source_name]['HF_10'])==1:
			HF_10=Flux[np.where((Wave>1230) & (Wave<1234))]
			g_line+=fit_lines_sinc(Wave,Flux,HF_10,'Abs')[1]
			label+=' + HF'
			fitted_lines.append('HF_10')

		if int(my_dict[source_name]['H2O_1113'])==1:
			H2O_1113=Flux[np.where((Wave>1112) & (Wave<1114))]
			g_line+= fit_lines_sinc(Wave,Flux,H2O_1113,'Abs')[1]
			label+=' + H2O'
			fitted_lines.append('H2O_1113')

		if int(my_dict[source_name]['H2O_1115'])==1:
			H2O_1115=Flux[np.where((Wave>1114) & (Wave<1116))]
			g_line+=fit_lines_sinc(Wave,Flux,H2O_1115,'Abs')[1]
			fitted_lines.append('H2O_1115')
	
	if colour=='SLW':
		l_init_continuum=fit_continuum(Wave,Flux,7)[0]
		g_line=l_init_continuum
		label=''
		fitted_lines=[]

		if int(my_dict[source_name]['CI_10'])==1:
			CI_10=Flux[np.where((Wave>490) & (Wave<493.8))]
			g_line+=fit_lines_sinc(Wave,Flux,CI_10,'Em')[1]
			label+=' + CI'
			fitted_lines.append('CI_10')

		if int(my_dict[source_name]['CI_21'])==1:
			CI_21=Flux[np.where((Wave>808.25) & (Wave<810.75))]
			g_line+=fit_lines_sinc(Wave,Flux,CI_21,'Em')[1]
			fitted_lines.append('CI_21')
			
		if int(my_dict[source_name]['CO_43'])==1:
			CO_43=Flux[np.where((Wave>459) & (Wave<463))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_43,'Em')[1]
			label+=' + C0'
			fitted_lines.append('CO_43')

		if int(my_dict[source_name]['CO_54'])==1:
			CO_54=Flux[np.where((Wave>573) & (Wave<579))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_54,'Em')[1]
			fitted_lines.append('CO_54')

		if int(my_dict[source_name]['CO_65'])==1:
			CO_65=Flux[np.where((Wave>689) & (Wave<693))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_65,'Em')[1]
			fitted_lines.append('CO_65')

		if int(my_dict[source_name]['CO_76'])==1:
			CO_76=Flux[np.where((Wave>804) & (Wave<807))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_76,'Em')[1]
			fitted_lines.append('CO_76')
			
		if int(my_dict[source_name]['CO_87'])==1:
			CO_87=Flux[np.where((Wave>919) & (Wave<923))]
			g_line+=fit_lines_sinc(Wave,Flux,CO_87,'Em')[1]
			fitted_lines.append('CO_87')

		if int(my_dict[source_name]['OH_909'])==1:
			OH_909=Flux[np.where((Wave>907) & (Wave<911))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_909,'Abs')[1]
			label+=' + OH'
			fitted_lines.append('OH_909')

		if int(my_dict[source_name]['OH_971'])==1:
			OH_971=Flux[np.where((Wave>969) & (Wave<973))]
			g_line+=fit_lines_sinc(Wave,Flux,OH_971,'Abs')[1]
			label+=' + OH'
			fitted_lines.append('OH_971')

		if int(my_dict[source_name]['CH_835'])==1:
			CH_835=Flux[np.where((Wave>833) & (Wave<837))]
			g_line+=fit_lines_sinc(Wave,Flux,CH_835,'Abs')[1]
			label+=' + CH'
			fitted_lines.append('CH_835')

	return g_line,label, fitted_lines

def fit_one_spectrum_spire_map(source_name,colour,mode,row=None,column=None):

	if colour=='SSW':
		Flux_mean_HR_SSW_image,Flux_mean_HR_SSW_spectrum, Wave_HR_SSW, wcs_HR_SSW, Flux_SSW=plot_image_flux_SPIRE_Map(source_name,'HR','SSW')
		wcs=wcs_HR_SSW
		
		if mode=='mean':
			#cond=np.where(np.isfinite(Flux_mean_HR_SSW_spectrum))
			#Wave, Flux=Wave_HR_SSW[cond], Flux_mean_HR_SSW_spectrum[cond]
			Wave, Flux=Wave_HR_SSW, Flux_mean_HR_SSW_spectrum
			title=': mean spectrum'
			title_spec='Average on the total image'
		if mode=='one':
			if row==None and column==None:			#If row and column not specified, take a ramndom position (row and column)
				cond=np.where(np.isfinite(Flux_SSW))	#returns a 3D array
				row=random.choice(np.arange(np.min(cond[1]+1),np.max(cond[1])))	#1 is the row dimension and 2 the column dimension
				column=random.choice(np.arange(np.min(cond[2]+1),np.max(cond[2])))
			#print(row,column)
			#cond=np.where(np.isfinite(Flux_SSW[:,row,column]))
			
			cond=np.where(np.isnan(Flux_SSW[:,row,column]))
			
			if np.shape(cond)[1]!=0:
				Flux_SSW[cond,row,column]=0
			#Wave, Flux=Wave_HR_SSW[cond], Flux_SSW[cond,row,column][0]
			Wave, Flux=Wave_HR_SSW, Flux_SSW[:,row,column]
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'

		#if np.shape(cond)[1]!=0:
		g_line, label, fitted_lines =fit_function_spire_map(source_name,'SSW', Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		fit_line=fit_g(g_line,Wave,Flux)
		y_line= fit_line(Wave)
	
		'''
		else:
			g_line, label = 0,0
			fitted_lines=''
			fit_line=0
			y_line= 0'''

		Residual=Flux-y_line

	if colour=='SLW':
		Flux_mean_HR_SLW_image,Flux_mean_HR_SLW_spectre, Wave_HR_SLW, wcs_HR_SLW, Flux_SLW=plot_image_flux_SPIRE_Map(source_name,'HR','SLW')
		wcs=wcs_HR_SLW

		if mode=='mean':
			#cond=np.where(np.isfinite(Flux_mean_HR_SLW_spectre))
			#Wave, Flux=Wave_HR_SLW[cond], Flux_mean_HR_SLW_spectre[cond]
			Wave, Flux=Wave_HR_SLW, Flux_mean_HR_SLW_spectre
			title=': mean spectrum'
			title_spec='Average on the total image'
		if mode=='one':
			if row==None and column==None:			#If row and column not specified, take a ramndom position (row and column)
				cond=np.where(np.isfinite(Flux_SLW))	#returns a 3D array
				row=random.choice(np.arange(np.min(cond[1]+1),np.max(cond[1])))	#1 is the row dimension and 2 the column dimension
				column=random.choice(np.arange(np.min(cond[2]+1),np.max(cond[2])))
				#print(row,column)
			
			cond=np.where(np.isnan(Flux_SLW[:,row,column]))
			if np.shape(cond)[1]!=0:
				Flux_SLW[cond,row,column]=0

			#cond=np.where(np.isfinite(Flux_SLW[:,row,column]))
			#Wave, Flux=Wave_HR_SLW[cond], Flux_SLW[cond,row,column][0]
			Wave, Flux=Wave_HR_SLW, Flux_SLW[:,row,column]
			title=': one spectrum' 
			title_spec='pixel:('+str(row)+','+str(column)+')'

		#if np.shape(cond)[1]!=0:
		g_line, label, fitted_lines =fit_function_spire_map(source_name,'SLW', Wave, Flux)
		fit_g = fitting.LevMarLSQFitter()
		fit_line=fit_g(g_line,Wave,Flux)
		y_line= fit_line(Wave)
		
		'''
		else:
			g_line, label = 0,0
			fitted_lines=''
			fit_line=0
			y_line= 0'''

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

	return 


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
				
			
			Source_R[str(line)]=fit_R,Line_flux_R


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

		Source_B={}
		Source_B['Object']=str(source_name)
		Source_B['WCS']=wcs_B
		Source_B['Wave']=Wave_B
		Source_B['Param_names']=param_names
		Source_B['Residual']=Residual_B
		

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

			Source_B[str(line)]=fit_B,Line_flux_B
		
		write_dict(Source_R,str(source_name)+'_R.npy')
		write_dict(Source_B,str(source_name)+'_B.npy')

		return fit_line_R,fit_line_B, Source_R, Source_B

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
						Line_flux_SSW[i,j]=(amplitude*std*np.pi)
				
			
			Source_SSW[str(line)]=fit_line_SSW, Line_flux_SSW


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
						Line_flux_SLW[i,j]=(amplitude*std*np.pi)
				
			
			Source_SLW[str(line)]=fit_line_SLW, Line_flux_SLW

		write_dict(Source_SSW,str(source_name)+'_SSW.npy')
		write_dict(Source_SLW,str(source_name)+'_SLW.npy')

		return fit_line_SSW,fit_line_SLW, Source_SSW, Source_SLW


############################################################  MAPS-PACS/SPIRE - BUBBLES   ########################################################
#################################################################################################################################################

def fit_Bubbles():
	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()

	for source in sources:
		if int(my_dict[str(source)]['SPIRE_MAP'])==0 and int(my_dict[str(source)]['SPIRE_SPARSE'])==0 :
			fit_map_spectrum(source,'PACS')
			#fit_map_spectrum(source,'SPIRE_MAP')
	#if int(my_dict[str(source_name)]['SPIRE_MAP'])==1:
	#	fit_map_spectrum(source_name,'SPIRE_MAP')

	#return

def colorbar(mappable):
	ax = mappable.axes
	fig = ax.figure
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	return fig.colorbar(mappable, cax=cax)

def plot_map(source_name):
	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()

	Source_R=np.load(direc1+'MAPS/'+str(source_name)+'_R.npy',allow_pickle=True).item()
	Source_B=np.load(direc1+'MAPS/'+str(source_name)+'_B.npy',allow_pickle=True).item()
	WCS_R=Source_R['WCS']
	WCS_B=Source_B['WCS']
	lines_R=list(Source_R.keys())[5:]
	lines_B=list(Source_B.keys())[5:]


	num_name=source_name[4:8]
	hdu_MIPS24=fits.open(direc+'Bulles_Nicolas/MB'+str(num_name)+'/MIPS24_MB'+str(num_name)+'.fits')

	Flux_range_Mips=np.nanmax(hdu_MIPS24[0].data[100:150,100:150])-np.nanmin(hdu_MIPS24[0].data[100:150,100:150])
	#Flux_range_Mips=np.nanmax(hdu_MIPS24[0].data)-np.nanmin(hdu_MIPS24[0].data)
	Levels=list(np.arange(np.nanmin(hdu_MIPS24[0].data[100:150,100:150]),np.nanmax(hdu_MIPS24[0].data[100:150,100:150]),Flux_range_Mips/10.))

	plt.figure()
	plt.subplot(1,1,1,projection=WCS(hdu_MIPS24[0].header))
	plt.imshow(hdu_MIPS24[0].data)#[100:150,100:150])
	plt.colorbar(label="Flux (MJy/sr)")
	plt.contour(hdu_MIPS24[0].data, levels=Levels, colors='white',linewidths=0.5)#, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=[50,60,70,80,100], colors='white',linewidths=0.5)
	#plt.colorbar()
	plt.xlabel('RA (J2000)')
	plt.ylabel('DEC (J2000)')
	plt.title('Source '+str(source_name)+r' observed with Spitzer MIPS 24 $\mu$m', fontstyle='italic',fontsize=11)
	#plt.grid()
	plt.show()

	plt.figure()
		#gs = GridSpec(1, 2, hspace=0.3, wspace=0.3)
		#plt.suptitle('Line Flux maps for '+str(source_name))
	a=1
	for line in lines_R:
		#index=np.where(Source_R[line]>0.9*np.nanmax(Source_R[line]))
		#Source_R[line][index]=np.median(Source_R[line])
		if  line=='OH_119'  or line=='NII_122_abs' :
			type_line='Absorption'
		else:
			type_line='Emission'
		size=np.shape(Source_R[line][1])
		ax=plt.subplot(2,2,a,projection=WCS_R)
		im=plt.imshow(Source_R[line][1])#(np.where(Source_R[line]<0.8*np.nanmax(Source_R[line]))))
		ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
		#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
		ax.set_xlim(-0.5,np.shape(Source_R[line][1])[1]-0.5)
		ax.set_ylim(-0.5,np.shape(Source_R[line][1])[0]-0.5)
		#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
		#ax.set_xlim(0,np.shape(Source_R[line])[0]-1)
		#ax.set_ylim(0,np.shape(Source_R[line])[1]-1)
		ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
		ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
		ax.coords[0].set_ticks(exclude_overlapping=True)
		plt.title(str(type_line)+' line '+str(line),fontsize=10)
		plt.colorbar(im, ax=ax , label="Flux (MJy/sr)",aspect=20)
		a+=1
		#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in PACS R-Band  Image size: '+str(size), fontsize=11,fontstyle='italic')
	
	plt.show()

	a=1
	plt.figure()
	for line in lines_B:			#Que des Emission lines
		size=np.shape(Source_B[line][1])
		ax=plt.subplot(1,2,a,projection=WCS_B)
		im=plt.imshow(Source_B[line][1])
		ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
		#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
		ax.set_xlim(-0.5,np.shape(Source_B[line][1])[1]-0.5)
		ax.set_ylim(-0.5,np.shape(Source_B[line][1])[0]-0.5)
		#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
		#ax.set_xlim(0,np.shape(Source_B[line])[0]-1)
		#ax.set_ylim(0,np.shape(Source_B[line])[1]-1)
		ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
		ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
		ax.coords[0].set_ticks(exclude_overlapping=True)
		plt.title('Emission line '+str(line),fontsize=10)
		plt.colorbar(im, ax=ax, label="Flux (MJy/sr)")
		#plt.colorbar()
		a+=1
		#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in PACS B-Band Image size: '+str(size),fontsize=11,fontstyle='italic')
	plt.show()

	
	if int(my_dict[str(source_name)]['SPIRE_MAP'])==1 and int(my_dict[str(source_name)]['PACS'])==1:

		Source_SSW=np.load(direc1+'MAPS/'+str(source_name)+'_SSW.npy',allow_pickle=True).item()
		Source_SLW=np.load(direc1+'MAPS/'+str(source_name)+'_SLW.npy',allow_pickle=True).item()			#
		lines_SSW=list(Source_SSW.keys())[5:]
		lines_SLW=list(Source_SLW.keys())[5:]
		WCS_SSW=Source_SSW['WCS']
		WCS_SLW=Source_SLW['WCS']

		plt.figure()
		a=1
		for line in lines_SSW:			#More absorption lines
			if line=='NII_1461' or line=='CO_109' or line=='CO_1110':
				type_line='Emission'
			else:
				type_line='Absorption'

			size=np.shape(Source_SSW[line][1])
			ax=plt.subplot(2,3,a,projection=WCS_SSW)
			im=plt.imshow(Source_SSW[line][1])
			ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
				#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
			ax.set_xlim(-0.5,np.shape(Source_SSW[line][1])[1]-0.5)
			ax.set_ylim(-0.5,np.shape(Source_SSW[line][1])[0]-0.5)
				#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
				#ax.set_xlim(0,np.shape(Source_SSW[line])[0]-1)
				#ax.set_ylim(0,np.shape(Source_SSW[line])[1]-1)
			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')# fontsize=10)
				#divider = make_axes_locatable(ax)
				#cax2 = divider.append_axes("right", size="5%", pad=0.05)
				#colorbar(im)
				#plt.colorbar()
			a+=1
				#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in SPIRE-MAP SSW-Band Image size: '+str(size),fontsize=11, fontstyle='italic')
		plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.5)
		plt.show()
			
		plt.figure()
		a=1
		for line in lines_SLW[:6]:				#More emission lines
			if line=='OH_909' or line=='OH_971' or line=='CH_835':
				type_line='Absorption'
			else:
				type_line='Emission'
			size=np.shape(Source_SLW[line][1])
			ax=plt.subplot(2,3,a,projection=WCS_SLW)
			im=plt.imshow(Source_SLW[line][1])
			ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
				#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
			ax.set_xlim(-0.5,np.shape(Source_SLW[line][1])[1]-0.5)
			ax.set_ylim(-0.5,np.shape(Source_SLW[line][1])[0]-0.5)#-1)
				#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
				#ax.set_xlim(0,np.shape(Source_SLW[line])[0]-1)
				#ax.set_ylim(0,np.shape(Source_SLW[line])[1]-1)
			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
				#plt.colorbar()
			a+=1
				#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in SPIRE-MAP SLW-Band. Image size: '+str(size),fontsize=11,fontstyle='italic')
		plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.5)
		plt.show()

		plt.figure()
		a=1
		for line in lines_SLW[6:]:				#More emission lines
			if line=='OH_909' or line=='OH_971' or line=='CH_835':
				type_line='Absorption'
			else:
				type_line='Emission'
			size=np.shape(Source_SLW[line][1])
			ax=plt.subplot(2,2,a,projection=WCS_SLW)
			im=plt.imshow(Source_SLW[line][1])
			ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
				#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
			ax.set_xlim(-0.5,np.shape(Source_SLW[line][1])[1]-0.5)
			ax.set_ylim(-0.5,np.shape(Source_SLW[line][1])[0]-0.5)#-1)
				#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
				#ax.set_xlim(0,np.shape(Source_SLW[line])[0]-1)
				#ax.set_ylim(0,np.shape(Source_SLW[line])[1]-1)
			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
				#plt.colorbar()
			a+=1
				#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in SPIRE-MAP SLW-Band. Image size: '+str(size),fontsize=11,fontstyle='italic')
		plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.5)
		plt.show()

		#bounds=np.arange(np.nanmin(Source_R[line]),0.9*np.nanmax(Source_R[line]),1)
		#norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
		#pcm = ax.pcolormesh(Source_R[line], norm=norm)#, cmap='RdBu_r')


	return #Source_R, Source_B

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


def footprint(source_name):

	Detectors=np.load(direc1+'SPARSE_detectors.npy',allow_pickle=True).item()
	detector_SLW=list(Detectors[source_name].keys())[:19]
	detector_SSW=list(Detectors[source_name].keys())[19:]
	Beam_SSW=Angle('8"',unit=u.deg).degree					#Beam=16 but we want the radius so 8
	Beam_SLW=Angle('17"',unit=u.deg).degree
	#sep_beam_SSW=Angle('13.5"',unit=u.deg).degree
	#sep_beam_SLW=Angle('48"',unit=u.deg).degree
	FOV=Angle('1.3m',unit=u.deg).degree

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

	r = SphericalCircle((Detectors[source_name][central_detector_SLW][0]* u.deg, Detectors[source_name][central_detector_SLW][1] * u.deg), Beam_SLW * u.degree, edgecolor='yellow',label='SLW',facecolor='none', linewidth=1.2,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
	ax.add_patch(r)

	r = SphericalCircle((Detectors[source_name][central_detector_SSW][0]* u.deg, Detectors[source_name][central_detector_SSW][1] * u.deg), Beam_SSW * u.degree, edgecolor='red',label='SSW',facecolor='none',linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
	ax.add_patch(r)

	for detector in detector_SLW:
		if detector=='central_detector_SLW':
			continue
		r = SphericalCircle((Detectors[source_name][detector][0]* u.deg, Detectors[source_name][detector][1] * u.deg), Beam_SLW * u.degree, edgecolor='yellow',facecolor='none', linewidth=1.2,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
		ax.add_patch(r)
	#plt.legend('SLW')

	for detector in detector_SSW:
		if detector=='central_detector_SLW':
			continue
		r = SphericalCircle((Detectors[source_name][detector][0]* u.deg, Detectors[source_name][detector][1] * u.deg), Beam_SSW * u.degree, edgecolor='red',facecolor='none',linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
		ax.add_patch(r)
	
	#plt.legend('SSW')
	r = SphericalCircle((Detectors[source_name][central_detector_SLW][0]* u.deg, Detectors[source_name][central_detector_SLW][1] * u.deg), FOV * u.degree, edgecolor='green',linewidth=2,facecolor='none',transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
	ax.add_patch(r)
	plt.legend()
	plt.suptitle('Footprint of the SPIRE Sparse FTS bolometers for source '+ str(source_name))

	plt.show()

############################################################  CO-LINES Rotation Diagram   ########################################################
#################################################################################################################################################


def plot_co_rot_diagram():

	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()
	Energy_level=[55.32,82.97,116.16,154.87]			 #in Kelvin so divided by Kb
	J=[4,5,6,7]										#Quantum number to divide the flux by the statistical weight : 2*J+1
	Aul=[6.126E-6,1.221E-5,2.137E-5,3.422E-5]		#Einstein coefficients
	Freq=[461.04E9,576.27E9,691.47E9,806.65E9]
	k=1.38064852E-23								#Boltzmann constant
	h=6.62607015E-34								#Planck constant 
	c=3E8
	#Gamma=8*np.pi*k/(h*c*c*c)
	Gamma=8E-4*np.pi/h
	print(Gamma)
	fit=models.Linear1D(slope=-1,intercept=1)
	fitter=fitting.LinearLSQFitter()

	a=1
	for source in sources:
		#a=1
		if int(my_dict[str(source)]['SPIRE_MAP'])==1:
			type_source=my_dict[str(source)]['Morphologie']
			print(source)
			Source_SSW=np.load(direc1+'MAPS/'+str(source)+'_SSW.npy',allow_pickle=True).item()
			Source_SLW=np.load(direc1+'MAPS/'+str(source)+'_SLW.npy',allow_pickle=True).item()	

			lines_SSW=list(Source_SSW.keys())[5:]
			lines_SLW=list(Source_SLW.keys())[5:]

			CO=[]					#We'll create a list with a mean value of each CO line 
			CO_error=[]
			Energy=[]
			Nu=np.zeros_like(Source_SLW['CI_10'][1])

			#We take sources with at least 3 CO lines:
			
			for line in lines_SLW:
				#Nu=np.zeros_like(Source_SLW[line][1])
				if line=='CO_43':
					CO.append(np.log(10*np.nanmean(Source_SLW[line][1])*Gamma*Freq[0]**5/((2*J[0]+1)*Aul[0])))	#[1] IS THE 2D image with a value for each pixel
					CO_error.append(np.nanstd(Source_SLW[line][1])/np.nanmean(Source_SLW[line][1]))
					Energy.append(Energy_level[0])
				if line=='CO_54' :
					CO.append(np.log(10*np.nanmean(Source_SLW[line][1])*Gamma*Freq[1]**5/((2*J[1]+1)*Aul[1])))	#[1] IS THE 2D image with a value for each pixel
					CO_error.append(np.nanstd(Source_SLW[line][1])/np.nanmean(Source_SLW[line][1]))
					Energy.append(Energy_level[1])

				if line=='CO_65':
					CO.append(np.log(10*np.nanmean(Source_SLW[line][1])*Gamma*Freq[2]**5/((2*J[2]+1)*Aul[2])))	#[1] IS THE 2D image with a value for each pixel
					CO_error.append(np.nanstd(Source_SLW[line][1])/np.nanmean(Source_SLW[line][1]))
					Energy.append(Energy_level[2])
				if line=='CO_76':
					CO.append(np.log(10*np.nanmean(Source_SLW[line][1])*Gamma*Freq[3]**5/((2*J[3]+1)*Aul[3])))
					CO_error.append(np.nanstd(Source_SLW[line][1])/np.nanmean(Source_SLW[line][1]))
					Energy.append(Energy_level[3])

			print(CO)
			print(CO_error)

			if len(CO)>2:
				#a=1
				fit_CO=fitter(fit,Energy,CO)
				y_fit=fit_CO(Energy)
				Temp=int(-1/fit_CO.parameters[0])

				plt.subplot(2,3,a)
				plt.errorbar(Energy,CO,CO_error)# 'p', markerfacecolor='cyan', markeredgecolor='blue',label='Data')#label='Slope=%5.3f, '%tuple(popt))
				plt.plot(Energy,y_fit, '--g', label='Temp= '+str(Temp)+' K')
				plt.xlabel(r'$E_{u}/k_{b}$ (K)', fontsize=8)
				plt.ylabel(r'$ln(N_{u}/g_{u})$ ($cm^{-2}$)', fontsize=8)
				plt.title(str(source)+' ('+str(type_source)+' Morphology) ', fontsize=9)
				plt.legend(fontsize=8)
				a+=1
			#plt.ylim(5E-20,5E-19)

		
	plt.suptitle('CO rotation diagram from SPIRE MAP data ', fontsize=10, fontstyle='italic')
	plt.tight_layout()
	plt.show()

	return #fit_CO, CO



