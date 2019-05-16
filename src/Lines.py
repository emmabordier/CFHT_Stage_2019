import numpy as np
import matplotlib.pyplot as plt
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
import os as os
import csv
from Herschel import *

direc='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/'
direc1='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/Data/'


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
		mean= float(Wave[np.where(Flux==np.nanmax(Flux_range))])
	if emission=='Abs':
		ampli=-abs(np.nanmin(Flux_range))
		mean= float(Wave[np.where(Flux==np.nanmin(Flux_range))])

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
		mean= float(Wave[np.where(Flux==np.nanmax(Flux_range))])
	if emission=='Abs':
		ampli=-abs(np.nanmin(Flux_range))
		mean= float(Wave[np.where(Flux==np.nanmin(Flux_range))])

	g_init= models.Lorentz1D(amplitude=ampli, x_0=mean, fwhm=0.01, bounds={"mean": (mean-0.01*mean, mean+0.01*mean)})
	g_line= g_init+l_init_continuum

	fit_line=fit_g(g_line,Wave,Flux)
	y_line= fit_line(Wave)

	return y_line, g_init, fit_line

def fit_function(source_name, instru, Wave_R,Wave_B,Flux_R,Flux_B, line=None):			#Calculate the fit function according to the lines for a given source  / instru= PACS/SPIRE_Map/SPIRE_Sparse

	my_dict=csv_to_dict('Bulles.csv')


	if instru=='PACS':
		l_init_continuum_R, l_init_continuum_B = fit_continuum(Wave_R,Flux_R,2)[0], fit_continuum(Wave_B,Flux_B,2)[0]		#We use a 2-degree polynomial model for the continuum
		g_line_R_g=l_init_continuum_R			#Initialization of g_line (model function) R-range with Gaussian model
		g_line_B_g=l_init_continuum_B			#Same with B-range
		g_line_R_l=l_init_continuum_R			#Initialization of g_line R-range with Lorentzian model
		g_line_B_l=l_init_continuum_B			#Same with B-range
		label_R='continuum poly deg2'
		label_B='continuum poly deg2'

		if int(my_dict[source_name]['NII_122_em'])==1 or line=='NII_122_em':
			NII= Flux_R[np.where((Wave_R>121.5) & (Wave_R<122.5))]

			g_line_R_g+=fit_lines_gauss(Wave_R,Flux_R,NII,'Em')[1]
			g_line_R_l+=fit_lines_lorentz(Wave_R,Flux_R,NII,'Em')[1]
			label_R+=' + NII'

		if int(my_dict[source_name]['NIII_57'])==1:
			NIII=Flux_B[np.where((Wave_B>55) & (Wave_B<59))] 

			g_line_B_g+=fit_lines_gauss(Wave_B,Flux_B,NIII,'Em')[1]
			g_line_B_l+=fit_lines_lorentz(Wave_B,Flux_B,NIII,'Em')[1]
			label_B+=' + NIII'

		if int(my_dict[source_name]['OI_63'])==1:
			OI_63, OI_145=Flux_B[np.where((Wave_B>62) & (Wave_B<64))], Flux_R[np.where((Wave_R>144.8) & (Wave_R<145.8))]

			g_init_OI_63,g_init_OI_145=fit_lines_gauss(Wave_B,Flux_B,OI_63,'Em')[1], fit_lines_gauss(Wave_R,Flux_R,OI_145,'Em')[1]
			g_init_OI_63_l,g_init_OI_145_l= fit_lines_lorentz(Wave_B,Flux_B,OI_63,'Em')[1], fit_lines_lorentz(Wave_R,Flux_R,OI_145,'Em')[1]

			g_line_R_g+=g_init_OI_145
			g_line_B_g+=g_init_OI_63
			g_line_R_l+=g_init_OI_145_l
			g_line_B_l+=g_init_OI_63_l
			label_R+=' + OI_145'
			label_B+=' + OI_63'

		if int(my_dict[source_name]['OH_119'])==1:
			OH_1= Flux_R[np.where((Wave_R>119.21) & (Wave_R<119.25))] 	#OH: 119.23 et 119.44
			OH_2= Flux_R[np.where((Wave_R>119.42) & (Wave_R<119.46))]

			g_init_OH_1, g_init_OH_2=fit_lines_gauss(Wave_R,Flux_R,OH_1,'Abs')[1],fit_lines_gauss(Wave_R,Flux_R,OH_2,'Abs')[1]
			g_init_OH_1_l, g_init_OH_2_l=fit_lines_lorentz(Wave_R,Flux_R,OH_1,'Abs')[1],fit_lines_lorentz(Wave_R,Flux_R,OH_2,'Abs')[1]

			g_line_R_g+=g_init_OH_1+g_init_OH_2
			g_line_R_l+=g_init_OH_1_l+ g_init_OH_2_l
			label_R+=' + OH_119'


		if int(my_dict[source_name]['NII_122_abs'])==1:
			NII_abs= Flux_R[np.where((Wave_R>121) & (Wave_R<123))]

			g_line_R_g+=fit_lines_gauss(Wave_R,Flux_R,NII_abs,'Abs')[1]
			g_line_R_l+=fit_lines_lorentz(Wave_R,Flux_R,NII_abs,'Abs')[1]
			label_R+=' + NII_122'

		return g_line_R_g, g_line_B_g,g_line_R_l, g_line_B_l, label_R, label_B 
	
	if instru=='SPIRE_Map':
		l_init_continuum_SSW,l_init_continuum_SLW = fit_continuum(Wave_R,Flux_R,7)[0], fit_continuum(Wave_B,Flux_B,7)[0] 
		g_line_SSW=l_init_continuum_SSW
		g_line_SLW=l_init_continuum_SLW
		label_SSW='continuum poly deg6'
		label_SLW='continuum poly deg6'

		if int(my_dict[source_name]['NII_1461'])==1:
			NII= Flux_R[np.where((Wave_R>1450) & (Wave_R<1466))]
			g_line_SSW+=fit_lines_sinc(Wave_R,Flux_R,NII,'Em')[1]
			label_SSW+=' + NII'
		
		if int(my_dict[source_name]['CI_10'])==1:
			if source_name=='MGE_4121':

				CI_10,CI_21=Flux_B[np.where((Wave_B>490) & (Wave_B<493.8))],Flux_B[np.where((Wave_B>807) & (Wave_B<812))]

			else:
				CI_10,CI_21=Flux_B[np.where((Wave_B>490) & (Wave_B<493.8))],Flux_B[np.where((Wave_B>808.25) & (Wave_B<810.75))]
			
			g_init_CI_10,g_init_CI_21=fit_lines_sinc(Wave_B,Flux_B,CI_10,'Em')[1],fit_lines_sinc(Wave_B,Flux_B,CI_21,'Em')[1]
			g_line_SLW+=g_init_CI_10+g_init_CI_21
			label_SLW+=' + CI'
		
		if int(my_dict[source_name]['CO_43'])==1:
			CO_43,CO_54,CO_65=Flux_B[np.where((Wave_B>459) & (Wave_B<463))],Flux_B[np.where((Wave_B>573.5) & (Wave_B<579.5))],Flux_B[np.where((Wave_B>689) & (Wave_B<693))]

			g_init_CO_43,g_init_CO_54, g_init_CO_65=fit_lines_sinc(Wave_B,Flux_B,CO_43,'Em')[1],fit_lines_sinc(Wave_B,Flux_B,CO_54,'Em')[1],fit_lines_sinc(Wave_B,Flux_B,CO_65,'Em')[1]
			g_line_SLW+=g_init_CO_43+g_init_CO_54+g_init_CO_65
			label_SLW+=' + C0'

		if int(my_dict[source_name]['CO_76'])==1:
			CO_76=Flux_B[np.where((Wave_B>804) & (Wave_B<807))]
			g_init_CO_76=fit_lines_sinc(Wave_B,Flux_B,CO_76,'Em')[1]
			g_line_SLW+=g_init_CO_76
			#label_SLW+=' + C0'

		if int(my_dict[source_name]['OH_971'])==1:
			OH_909,OH_971_1,OH_971_2,OH_1033=Flux_B[np.where((Wave_B>907) & (Wave_B<911))],Flux_B[np.where((Wave_B>970) & (Wave_B<974))],Flux_R[np.where((Wave_R>969) & (Wave_R<973))], Flux_R[np.where((Wave_R>1031) & (Wave_R<1034))]

			g_init_OH_909,g_init_OH_971_1,g_init_OH_971_2 ,g_init_OH_1033=fit_lines_sinc(Wave_B,Flux_B,OH_909,'Abs')[1],fit_lines_sinc(Wave_B,Flux_B,OH_971_1,'Abs')[1],fit_lines_sinc(Wave_R,Flux_R,OH_971_2,'Abs')[1], fit_lines_sinc(Wave_R,Flux_R,OH_1033,'Abs')[1]
			g_line_SLW+=g_init_OH_909+g_init_OH_971_1
			g_line_SSW+=g_init_OH_1033+g_init_OH_971_2
			label_SLW+=' + OH'
			label_SSW+='+ OH'

		if int(my_dict[source_name]['CH_835'])==1:
			CH_835=Flux_B[np.where((Wave_B>834) & (Wave_B<837))]

			g_line_SLW+=fit_lines_sinc(Wave_B,Flux_B,CH_835,'Abs')[1]
			label_SLW+=' + CH'

		if int(my_dict[source_name]['HF_10'])==1:
			HF_10=Flux_R[np.where((Wave_R>1230) & (Wave_R<1234))]

			g_line_SSW+=fit_lines_sinc(Wave_R,Flux_R,HF_10,'Abs')[1]
			label_SSW+=' + CH'

		if int(my_dict[source_name]['H2O_1113'])==1:
			H2O_1113, H2O_1115=Flux_R[np.where((Wave_R>1112) & (Wave_R<1114))],Flux_R[np.where((Wave_R>1114) & (Wave_R<1116))]
			g_init_H2O_1113, g_init_H2O_1113= fit_lines_sinc(Wave_R,Flux_R,H2O_1113,'Abs')[1],fit_lines_sinc(Wave_R,Flux_R,H2O_1115,'Abs')[1]
			
			g_line_SSW+=g_init_H2O_1113+ g_init_H2O_1113
			label_SSW+=' + H2O'

		return g_line_SSW, g_line_SLW, label_SSW, label_SLW

def lines_PACS(source_name):
	Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R, Flux_red= flux_PACS(source_name,'R')
	Flux_image_B, Flux_spectrum_B, Wave_B, wcs_B, Flux_blue= flux_PACS(source_name,'B')

	cond_R, cond_B=np.where(np.isfinite(Flux_spectrum_R)), np.where(np.isfinite(Flux_spectrum_B))   #Pour ne pas compter les NaN
	Wave_R, Flux_R=Wave_R[cond_R], Flux_spectrum_R[cond_R]
	Wave_B, Flux_B=Wave_B[cond_B], Flux_spectrum_B[cond_B]

	#Total fit function for all the lines
	g_line_R_g, g_line_B_g, g_line_R_l, g_line_B_l, label_R, label_B = fit_function(source_name, 'PACS', Wave_R, Wave_B, Flux_R, Flux_B)

	fit_g = fitting.LevMarLSQFitter()
	fit_line_R_g=fit_g(g_line_R_g,Wave_R,Flux_R)
	fit_line_B_g=fit_g(g_line_B_g,Wave_B,Flux_B)

	fit_line_R_l=fit_g(g_line_R_l,Wave_R,Flux_R)
	fit_line_B_l=fit_g(g_line_B_l,Wave_B,Flux_B)

	y_line_R_g,y_line_B_g= fit_line_R_g(Wave_R),fit_line_B_g(Wave_B)
	y_line_R_l,y_line_B_l= fit_line_R_l(Wave_R),fit_line_B_l(Wave_B)

	#Residual flux
	Residu_R_g, Residu_B_g=Flux_R-y_line_R_g,Flux_B-y_line_B_g
	Residu_R_l, Residu_B_l=Flux_R-y_line_R_l,Flux_B-y_line_B_l

	#Plots
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(Wave_R, Flux_R, '-b')
	plt.plot(Wave_B, Flux_B, '-b', label='data')
	plt.plot(Wave_R,y_line_R_g,'-g',label='Gauss')
	plt.plot(Wave_B,y_line_B_g,'-g')#,label=str(label_R[:])+str(label_B[:]))
	plt.plot(Wave_R,y_line_R_l,'-r',label='Lorentz')
	plt.plot(Wave_B,y_line_B_l,'-r')#,label=str(label_B[:]))
	plt.legend()
	plt.title(r'Fit des lines PACS ')
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($MJy.sr^{-1}$)')
	
	# Residual flux plot 
	plt.subplot(1,2,2)
	plt.plot(Wave_R,Residu_R_g,'-c', label='Gauss')
	plt.plot(Wave_B, Residu_B_g, '-c')
	plt.plot(Wave_R,Residu_R_l+400,'-g', label='Lorentz')
	plt.plot(Wave_B, Residu_B_l+400, '-g')
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($MJy.sr^{-1}$)')
	plt.legend()
	plt.title(r'Residu: Données-Fit des lines')

	plt.suptitle(r'Fit des lines PACS pour '+str(source_name)+r' avec un model Gaussien $+$ Polynome deg 2 (continuum)')
	plt.show()

	return #fit_line_R, fit_line_B


############################################################  SPIRE-lines   #################################################################
############################################################################################################################################


def sinc_function(x,amplitude=1.,x0=0.,sigma=1.):#,bounds=1):
	return amplitude*np.sinc((x-x0)/sigma)
	#return amplitude*(np.sin(sigma*np.pi*(x-x0))/(sigma*np.pi*(x-x0)))

def fit_lines_sinc(Wave,Flux,Flux_range,emission):		#Colour=R,B/ Flux_range délimite la zone sur laquelle on veut fitter la line / Emission= Em/Abs
	fit_g = fitting.LevMarLSQFitter()
	fit_pol=fitting.LinearLSQFitter()
	l_init_continuum = models.Polynomial1D(degree=6)
	
	SincModel=custom_model(sinc_function)

	if emission=='Em':
		ampli=np.nanmax(Flux_range)
		x0= float(Wave[np.where(Flux==np.nanmax(Flux_range))])
	if emission=='Abs':
		ampli=-abs(np.nanmin(Flux_range))
		x0= float(Wave[np.where(Flux==np.nanmin(Flux_range))])

	#L1=np.
	#Wvl=Wave[np.where(Flux==np.nanmin(Flux_range))[0][0]:np.where(Flux==np.nanmax(Flux_range))[0][0]]
	#zeros=
	sigma=(Wave[np.where(Flux==np.nanmax(Flux_range))[0][0]]-Wave[np.where(Flux==np.nanmin(Flux_range))[0][0]])*2./np.pi 	 #Calcul approximatif d'un sigma car très sensible aux paramètres initiaux.
	g_init= SincModel(amplitude=ampli, x0=x0, sigma=sigma, bounds={"x0": (x0-0.02*x0, x0+0.02*x0)})
	#g_init= SincModel(amplitude=ampli, x0=x0,sigma=3.5,bounds={"x0": (x0-0.02*x0, x0+0.02*x0)})
	g_line=g_init+l_init_continuum

	fit_line=fit_g(g_line,Wave,Flux)
	y_line= fit_line(Wave)

	return y_line, g_init, fit_line

def lines_SPIRE_Map(source_name):#,mode):
	Flux_mean_HR_SLW_image,Flux_mean_HR_SLW_spectre, Wave_HR_SLW, wcs_HR_SLW, Flux_SLW=plot_image_flux_SPIRE_Map(source_name,'HR','SLW')
	Flux_mean_HR_SSW_image,Flux_mean_HR_SSW_spectre, Wave_HR_SSW, wcs_HR_SSW, Flux_SSW=plot_image_flux_SPIRE_Map(source_name,'HR','SSW')

	cond_SSW, cond_SLW=np.where(np.isfinite(Flux_mean_HR_SSW_spectre)), np.where(np.isfinite(Flux_mean_HR_SLW_spectre))   #Pour ne pas compter les NaN
	Wave_SLW, Flux_SLW=Wave_HR_SLW[cond_SLW], Flux_SLW[cond_SLW,5,5][0]#Flux_mean_HR_SLW_spectre[cond_SLW]#*1E17
	Wave_SSW, Flux_SSW=Wave_HR_SSW[cond_SSW], Flux_SSW[cond_SSW,5,5][0]#Flux_mean_HR_SSW_spectre[cond_SSW]#*1E17

	fit_g = fitting.LevMarLSQFitter()

	g_line_SSW, g_line_SLW, label_SSW, label_SLW= fit_function(source_name, 'SPIRE_Map', Wave_SSW, Wave_SLW, Flux_SSW, Flux_SLW)

	fit_g = fitting.LevMarLSQFitter()
	fit_line_SSW=fit_g(g_line_SSW,Wave_SSW,Flux_SSW)
	fit_line_SLW=fit_g(g_line_SLW,Wave_SLW,Flux_SLW)
	
	y_line_SSW=fit_line_SSW(Wave_SSW)
	y_line_SLW=fit_line_SLW(Wave_SLW)
	
	Residu_SLW=Flux_SLW-y_line_SLW
	Residu_SSW=Flux_SSW-y_line_SSW

	#Plots
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot(Wave_SLW, Flux_SLW, '-b', label='SLW')
	plt.plot(Wave_SSW, Flux_SSW, '-g', label='SSW')
	plt.plot(Wave_SSW,y_line_SSW,'c',label=str(label_SSW[:]))
	plt.plot(Wave_SLW,y_line_SLW,'r',label=str(label_SLW[:]))
	plt.legend()
	plt.title(r'Fit des lines PACS ')
	plt.xlabel(r'Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	
	# Plot du résidu (on soustrait toutes les lines présentes)
	plt.subplot(1,2,2)
	plt.plot(Wave_SSW,Residu_SSW)
	plt.plot(Wave_SLW, Residu_SLW)
	plt.xlabel(r'Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	plt.title(r'Residu: Données-Fit des lines')

	plt.suptitle(r'Fit des lines SPIRE pour '+str(source_name)+r' avec un model Sinc $+$ Polynome deg 2 (continuum)')
	plt.show()

	return fit_line_SSW,fit_line_SLW #fit_line_R, fit_line_B


############################################################  MAPS-PACS/SPIRE   #################################################################
############################################################################################################################################

'''
my_dict=csv_to_dict('Bulles.csv')
myKeys=my_dict.keys()
for keys in myKeys:
	if int(my_dict[keys]['SPIRE_MAP'])==1:
		lines_SPIRE_Map(keys)'''


def Flux_line(source_name,instru,line,row,column):		#row/colonne définit la position d'un pixel  instru=MAPS/SPIRE_Map
	fit_g = fitting.LevMarLSQFitter()
	
	if instru=='PACS':
		Wave_red,Flux_red= flux_PACS(source_name,'R')[2],flux_PACS(source_name,'R')[4]
		Wave_blue,Flux_blue= flux_PACS(source_name,'B')[2],flux_PACS(source_name,'B')[4]
		fit_g = fitting.LevMarLSQFitter()
		if line=='NII_122_em' or line=='OH_119' or line=='OI_145' or line=='NII_122_abs' or line=='Thing_123':
			cond=np.where(np.isfinite(Flux_red[:,row,column]))
			Wave_R, Flux_R=Wave_red[cond], Flux_red[:,row,column][cond]
		if line=='NIII_57' or line=='OI_63' :
			cond=np.where(np.isfinite(Flux_blue[:,row,column]))
			Wave_B, Flux_B=Wave_blue[cond], Flux_blue[:,row,column][cond]
			
		if np.shape(cond)[1]!=0 :

			if line=='NII_122_em':
			
				NII= Flux_R[np.where((Wave_R>121.5) & (Wave_R<122.5))]
				fit_line=fit_lines_gauss(Wave_R,Flux_R,NII,'Em')[2]

			if line=='NII_122_abs':
			
				NII= Flux_R[np.where((Wave_R>121.5) & (Wave_R<122.5))]
				fit_line=fit_lines_gauss(Wave_R,Flux_R,NII,'Abs')[2]
			
			if line=='Thing_123':
				Thing_123= Flux_R[np.where((Wave_R>122.5) & (Wave_R<123.5))]
				fit_line=fit_lines_gauss(Wave_R,Flux_R,Thing_123,'Em')[2]

			if line=='NIII_57':

				NIII=Flux_B[np.where((Wave_B>56) & (Wave_B<58))]
				fit_line=fit_lines_gauss(Wave_B,Flux_B,NIII,'Em')[2]
			
			if line=='OH_119':

				OH_1= Flux_R[np.where((Wave_R>119.21) & (Wave_R<119.25))]
				OH_2= Flux_R[np.where((Wave_R>119.42) & (Wave_R<119.46))]

				g_init_OH_1, g_init_OH_2=fit_lines_gauss(Wave_R,Flux_R,OH_1,'Abs')[1],fit_lines_gauss(Wave_R,Flux_R,OH_2,'Abs')[1]
				l_init_continuum = models.Polynomial1D(degree=2)

				g_line_OH=g_init_OH_1+g_init_OH_2+l_init_continuum
				fit_line=fit_g(g_line_OH,Wave_R,Flux_R)

			if line=='OI_63':

				OI_63=Flux_B[np.where((Wave_B>62) & (Wave_B<64))]
				fit_line=fit_lines_gauss(Wave_B,Flux_B,OI_63,'Em')[2]

			if line=='OI_145':

				 OI_145=Flux_R[np.where((Wave_R>144.8) & (Wave_R<145.8))]
				 fit_line=fit_lines_gauss(Wave_R,Flux_R,OI_145,'Em')[2]
		
			amplitude, std=abs(fit_line.amplitude_0[0]),fit_line.stddev_0[0]			#On retrouve les paramètres A et sigma pour le calcul du flux des lines
			Line_flux=amplitude*std*np.sqrt(2*np.pi)

		else:
			Line_flux=0

	if instru=='SPIRE_Map':
		Wave_red,Flux_red= plot_image_flux_SPIRE_Map(source_name,'HR','SSW')[2],plot_image_flux_SPIRE_Map(source_name,'HR','SSW')[4]
		Wave_blue,Flux_blue= plot_image_flux_SPIRE_Map(source_name,'HR','SLW')[2],plot_image_flux_SPIRE_Map(source_name,'HR','SLW')[4]
		fit_g = fitting.LevMarLSQFitter()
		if line=='NII_1461' or line=='OH_971' or line=='OH_1033' or line=='H2O_1113' or line=='H2O_1115':
			cond=np.where(np.isfinite(Flux_red[:,row,column]))
			Wave_R, Flux_R=Wave_red[cond], Flux_red[:,row,column][cond]
		if line=='CI_10' or line=='CI_21' or line=='CO_43' or line=='CO_54' or line=='CO_65' or line=='CO_76' or line=='CO_87' or line=='CH_835' or line=='OH_909':
			cond=np.where(np.isfinite(Flux_blue[:,row,column]))
			Wave_B, Flux_B=Wave_blue[cond], Flux_blue[:,row,column][cond]

		if np.shape(cond)[1]!=0 :
		#High frequencies
		
			if line=='NII_1461':
				NII= Flux_R[np.where((Wave_R>1450) & (Wave_R<1466))]
				fit_line=fit_lines_sinc(Wave_R,Flux_R,NII,'Em')[2]

			if line=='HF_10':
				HF_10=Flux_R[np.where((Wave_R>1230) & (Wave_R<1234))]
				fit_line=fit_lines_sinc(Wave_R,Flux_R,HF_10,'Abs')[2]

			if line=='H2O_1113':
				H2O_1113=Flux_R[np.where((Wave_R>1112) & (Wave_R<1114))]
				fit_line= fit_lines_sinc(Wave_R,Flux_R,H2O_1113,'Abs')[2]

			if line=='H2O_1115':
				H2O_1115=Flux_R[np.where((Wave_R>1114) & (Wave_R<1116))]
				fit_line=fit_lines_sinc(Wave_R,Flux_R,H2O_1115,'Abs')[2]

			if line=='OH_971':		
				OH_971=Flux_R[np.where((Wave_R>970) & (Wave_R<974))]
				fit_line=fit_lines_sinc(Wave_R,Flux_R,OH_971,'Abs')[2]

			if line=='OH_1033':	
				OH_1033=Flux_R[np.where((Wave_R>1031) & (Wave_R<1034))]
				fit_line=fit_lines_sinc(Wave_R,Flux_R,OH_1033,'Abs')[2]

			#Low Frequencies
			
			if line=='CI_10':
				CI_10=Flux_B[np.where((Wave_B>490) & (Wave_B<493.8))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,CI_10,'Em')[2]
		
			if line=='CI_21':
				#if source_name=='MGE_4121':
				CI_21= Flux_B[np.where((Wave_B>807) & (Wave_B<812))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,CI_21,'Em')[2]
			
			if line=='CO_43':
				CO_43=Flux_B[np.where((Wave_B>459) & (Wave_B<463))]

				fit_line=fit_lines_sinc(Wave_B,Flux_B,CO_43,'Em')[2]

			if line=='CO_54':
				CO_54=Flux_B[np.where((Wave_B>573.5) & (Wave_B<580))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,CO_54,'Em')[2]

			if line=='CO_65':
				CO_65=Flux_B[np.where((Wave_B>689) & (Wave_B<693))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,CO_65,'Em')[2]

			if line=='CO_76':
				CO_65=Flux_B[np.where((Wave_B>804) & (Wave_B<807))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,CO_76,'Em')[2]

			if line=='CO_87':
				CO_65=Flux_B[np.where((Wave_B>920) & (Wave_B<922))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,CO_87,'Em')[2]

			if line=='CH_835':
				CH_835=Flux_B[np.where((Wave_B>834) & (Wave_B<837))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,CH_835,'Abs')[2]


			if line=='OH_909':
				OH_909=Flux_B[np.where((Wave_B>907) & (Wave_B<911))]
				fit_line=fit_lines_sinc(Wave_B,Flux_B,OH_909,'Abs')[2]

			amplitude, std=abs(fit_line.amplitude_0[0]),abs(fit_line.sigma_0[0])		#On retrouve les paramètres A et sigma pour le calcul du flux des lines
			Line_flux=amplitude*std*np.pi
	

		else:
			Line_flux=0


	return Line_flux


def map_line(source_name,instru,line):
	if instru=='PACS':
		Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R, Flux_red= flux_PACS(source_name,'R')
		Flux_image_B, Flux_spectrum_B, Wave_B, wcs_B, Flux_blue= flux_PACS(source_name,'B')
		Flux_pixel_R=np.zeros((np.shape(Flux_image_R)[0],np.shape(Flux_image_R)[1]))			#Initialisation des tableaux: taille de l'image initiale. Généralement : 21*21 pixels (peut changer)
		Flux_pixel_B=np.zeros((np.shape(Flux_image_B)[0],np.shape(Flux_image_B)[1]))
		Wave, wcs=[], []
		if line=='NII_122_em' or line=='OH_119' or line=='OI_145' or line=='NII_122_abs' or line=='Thing_123':
			for i in range(np.shape(Flux_image_R)[0]-1):
				for j in range(np.shape(Flux_image_R)[1]-1):
					Flux_pixel_R[i,j]=Flux_line(source_name,'PACS',line,i,j)
			Flux_pixel=Flux_pixel_R
			Wave, wcs=Wave_R, wcs_R
		if line=='NIII_57' or line=='OI_63':
			for i in range(np.shape(Flux_image_B)[0]-1):
				for j in range(np.shape(Flux_image_B)[1]-1):
					Flux_pixel_B[i,j]=Flux_line(source_name,'PACS',line,i,j)
			Flux_pixel=Flux_pixel_B
			Wave, wcs=Wave_B, wcs_B	
		
	if instru=='SPIRE_Map':
		Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R, Flux_red= plot_image_flux_SPIRE_Map(source_name,'HR','SSW')
		Flux_image_B, Flux_spectrum_B, Wave_B, wcs_B, Flux_blue= plot_image_flux_SPIRE_Map(source_name,'HR','SLW')
		Flux_pixel_R=np.zeros((np.shape(Flux_image_R)[0],np.shape(Flux_image_R)[1]))			#Initialisation des tableaux: taille de l'image initiale. Généralement : 21*21 pixels (peut changer)
		Flux_pixel_B=np.zeros((np.shape(Flux_image_B)[0],np.shape(Flux_image_B)[1]))
		Wave, wcs=[], []
		if line=='NII_1461' or line=='OH_971' or line=='OH_1033' or line=='H2O_1113' or line=='H2O_1115':
			for i in range(np.shape(Flux_image_R)[0]-1):
				for j in range(np.shape(Flux_image_R)[1]-1):
					Flux_pixel_R[i,j]=Flux_line(source_name,'SPIRE_Map',line,i,j)
			Flux_pixel=Flux_pixel_R
			Wave, wcs=Wave_R, wcs_R

		if line=='CI_10' or line=='CI_21' or line=='CO_43' or line=='CO_54' or line=='CO_65' or line=='CO_76' or line=='CO_87' or line=='CH_835' or line=='OH_909':
			for i in range(np.shape(Flux_image_B)[0]-1):
				for j in range(np.shape(Flux_image_B)[1]-1):
					Flux_pixel_B[i,j]=Flux_line(source_name,'SPIRE_Map',line,i,j)
			Flux_pixel=Flux_pixel_B
			Wave, wcs=Wave_B, wcs_B	


	return Flux_pixel, Wave, wcs

def maps(source_name):				#Récupère toutes les lines dans le dictionnaire et stocke le flux de la line pour chaque pixel dans un dictionnaire
	my_dict=csv_to_dict('Bulles.csv')
	lines_PACS=list(my_dict[source_name].keys())[5:11]
	lines_SPIRE=list(my_dict[source_name].keys())[11:]
	Source={}
	Source['Object']=str(source_name)
	#Flux=[]
	for line in lines_PACS:
		if int(my_dict[source_name][line])==1:
			print(line)
			#Flux.append(carte_line(source_name,line))
			Source[str(line)]=map_line(source_name,'PACS',line)
			print (line,'done')
	for line in lines_SPIRE:
		if int(my_dict[source_name][line])==1:
			#print(line)
			Source[str(line)]=map_line(source_name,'SPIRE_Map',line)
			print(line,'done')

	write_dict(Source,str(source_name)+'_test.npy')
	return 				#L'objet de sortie est un dictionnaire où pour une source donnée, on a les lines, avec les cartes correspondantes

'''
image=MGE_4095['OI_63'][0]
wcs=MGE_4095['OI_63'][2]
fig=plt.figure()
gs = GridSpec(1, 1, hspace=0.3, wspace=0.3)
ax1 = plt.subplot(gs[0, 0], projection=wcs)
im1=plt.imshow(image,origin='lower')
fig.colorbar(im1, ax=ax1, label="Flux (Jy/pixel)", aspect=20)
ax1.coords[0].set_ticks(exclude_overlapping=True)
plt.xlabel('RA (J2000)')
plt.ylabel('DEC (J2000)')'''

#plt.subplot(2,2,1)
#plt.imshow(Flux[0],origin='lower')
#plt.subplot(2,2,2)
#plt.imshow(Flux[1],origin='lower')
#plt.subplot(2,2,3)
#plt.imshow(Flux[2],origin='lower')
#plt.show()'''


def plot_maps(source_name):
#Il faut utiliser la bibliothèque pickle pour pouvoir l'ouvrir. item() permet de ne pas l'ouvrir en tant que array mais en tant que dict
	Source=np.load(str(source_name)+'.npy',allow_pickle=True).item()			#
	lines=list(Source.keys())[1:]
	#On récupère la bulle en MIPS24
	num_name=source_name[4:8]
	hdu_MIPS24=fits.open(direc+'Bulles_Nicolas/MB'+str(num_name)+'/MIPS24_MB'+str(num_name)+'.fits')
	wcs=Source[lines[0]][2] # On prend le WCS de n'importe quelle raie. L'objet WCS est le même pour une source donnée, quelle que soit la raie
	#New_MIPS, footprint = reproject_interp(hdu_MIPS24, wcs, shape_out=np.shape(Source[lines[1]][0]))

	plt.figure()
	a=1
	#plt.figure()
	gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
	plt.suptitle('Line Flux maps for '+str(source_name))
	for line in lines:
		ax=plt.subplot(2,2,a,projection=wcs)
		im=plt.imshow(Source[line][0])
		ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)), colors='white',linewidths=0.5)
		#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
		ax.set_xlim(0,len(Source[line][0])-1)
		ax.set_ylim(0,len(Source[line][0])-1)
		#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
		ax.set_xlim(0,np.shape(Source[line][0])[0]-1)
		ax.set_ylim(0,np.shape(Source[line][0])[1]-1)
		ax.set_xlabel('RA (J2000)')
		ax.set_ylabel('DEC (J2000)')
		ax.coords[0].set_ticks(exclude_overlapping=True)
		plt.title('Map emission line '+str(line))
		plt.colorbar(im, ax=ax, label="Flux (MJy/sr)", aspect=20)
		a+=1
	#plt.tight_layout()
	plt.show()
	return


def comparison(source_name):
	num_name=source_name[4:8]
	hdu_MIPS24=fits.open(direc+'Bulles_Nicolas/MB'+str(num_name)+'/MIPS24_MB'+str(num_name)+'.fits')
	Source=np.load(str(source_name)+'.npy',allow_pickle=True).item()
	#test=Source['NII_122_em']
	test=Source['OH_119']
	wcs_MIPS=WCS(hdu_MIPS24[0].header)
	WCS_Source=test[2]
	New_MIPS, footprint = reproject_interp(hdu_MIPS24, WCS_Source, shape_out=np.shape(test[0]))
	
	fig = plt.figure(0) 
	fig.suptitle('Reprojection MIPS24 on Herschel')

	gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
	ax1 = plt.subplot(gs[0, 0], projection=wcs_MIPS)
	ax2 = plt.subplot(gs[1, 0], projection=WCS_Source)
	ax3 = plt.subplot(gs[0, 1], projection=WCS_Source)
	ax4 = plt.subplot(gs[1, 1], projection=WCS_Source)

	ax1.imshow(hdu_MIPS24[0].data)
	ax1.set_title('MIPS24', fontsize=11)
	ax1.coords[0].set_ticks(exclude_overlapping=True)
	ax1.set_xlabel('RA (J2000)')
	ax1.set_ylabel('DEC (J2000)')
	ax1.grid()
	
	ax2.imshow(test[0])
	ax2.set_title('Herschel NII', fontsize=11)
	ax2.coords[0].set_ticks(exclude_overlapping=True)
	ax2.set_xlabel('RA (J2000)')
	ax2.set_ylabel('DEC (J2000)')
	ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
	ax2.set_xlim(0,len(test[0])-1)
	ax2.set_ylim(0,len(test[0])-1)
	#ax2.imshow(hdu_MIPS24[0].data, alpha=0.5)
	ax2.grid()

	ax3.imshow(New_MIPS)
	ax3.set_title('MIPS24 on Herschel NII', fontsize=11)
	ax3.coords[0].set_ticks(exclude_overlapping=True)
	ax3.set_xlabel('RA (J2000)')
	ax3.set_ylabel('DEC (J2000)')

	ax4.imshow(test[0])
	ax4.contour(New_MIPS,colors='white',linewidths=0.5)
	ax4.coords[0].set_ticks(exclude_overlapping=True)
	ax4.set_title('Overlap', fontsize=11)
	ax4.set_xlabel('RA (J2000)')
	ax4.set_ylabel('DEC (J2000)')

	plt.show()
	
	return hdu_MIPS24 



