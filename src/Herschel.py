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
import glob
import csv

direc='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/'
direc1='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/Data/'

#Ces listes ont été trouvées avec le ipython Notebook CFHT_Nom.ipynb. Il existe des fichiers .txt et .csv répertoriant ces données pour chaque instrument
OBS_ID_PACS=[1342226189, 1342231302, 1342231303, 1342231738, 1342231739, 1342231740, 1342231741, 1342231742, 1342231743, 1342231744, 1342231745, 1342231746, 1342231747, 1342238730, 1342239689, 1342239741, 1342239757, 1342240165, 1342240166, 1342240167, 1342241267, 1342242444, 1342242445, 1342242446, 1342243105, 1342243106, 1342243107, 1342243108, 1342243505, 1342243506, 1342243507, 1342243513, 1342243901, 1342250917, 1342252270]
OBS_ID_SPIRE_MAP=[1342262927, 1342254039, 1342254040, 1342254041, 1342254042, 1342262919, 1342262924, 1342262926, 1342265807]
OBS_ID_SPIRE_SPARSE=[1342253970, 1342253971, 1342262922, 1342262923, 1342262925, 1342265810, 1342268284, 1342268285, 1342268286]
OBJECT_PACS=['MGE_4384', 'MGE_3438', 'MGE_3448', 'MGE_3269', 'MGE_3280', 'MGE_3739', 'MGE_3736', 'MGE_3719', 'MGE_3222', 'MGE_3354', 'MGE_3360', 'MGE_3681', 'MGE_3670', 'MGE_4048', 'MGE_4191', 'MGE_4206', 'MGE_4134', 'MGE_4121', 'MGE_4095', 'MGE_4218', 'MGE_3899', 'MGE_4552', 'MGE_4436', 'MGE_4486', 'MGE_4485', 'MGE_4111', 'MGE_4110', 'MGE_4204', 'MGE_3149', 'MGE_4602', 'MGE_4473', 'MGE_4524', 'MGE_3834', 'MGE_4239', 'MGE_4167']
OBJECT_SPIRE_MAP=['MGE_4121', 'MGE_3269', 'MGE_3280', 'MGE_3739', 'MGE_3736', 'MGE_4384', 'MGE_4204', 'MGE_4111', 'MGE_4485']
OBJECT_SPIRE_SPARSE=['MGE_3681', 'MGE_3448', 'MGE_4048', 'MGE_4206', 'MGE_4095', 'MGE_4134', 'MGE_3149', 'MGE_4602', 'MGE_4524']

#########################################################OUTILS####################################################################################
###################################################################################################################################################

def writefits(array, fname, overwrite=False):    #Fonction permettant d'écrire un fichier Fits. Attention au type de Header (PrimaryHDU, ImageHDU, BinTableHDU)

  if (os.path.isfile(fname)):
    if (overwrite == False):
      print("File already exists!, skiping")
      return
    else:
      print("File already exists!, overwriting")
      os.remove(fname)
  try: 
    hdu = fits.PrimaryHDU(array)
    hdu.writeto(fname)
  except:
    print('b1') 
    hdu = pf.PrimaryHDU(fname)
    print('b2') 
    hdu.writeto(array)

  print("File: %s saved!" % fname)

  return

def write_dict(array,fname,overwrite=False):
	if (os.path.isfile(fname)):
		if (overwrite == False):
			print("File already exists!, skiping")
			return

	np.save(fname,array,allow_pickle=True)
	print("File: %s saved!" % fname)

	return

def recup_long_number_Map(instru,resol=None,WVL=None,Couleur=None):			#Fichier=PACS,SPIRE_MAPPING,SPIRE_SPARSE. Type=Map, Sparse ou PACS. If PACS : Color:B/R
	if instru=='Map':
		path='SPIRE_MAPPING/'+str(resol)+'_'+str(WVL)+'/'
	if instru=='PACS':
		path='PACS/HPS3DEQI'+str(Couleur)+'/'
	if instru=='Sparse':
		path='SPIRE_SPARSE/'+str(resol)+'_spectrum_ext/'
	files=os.listdir(direc1+path)
	fits_files = []
	for names in files:
		if names.endswith(".fits"):
			fits_files.append(names)
	long_number=[]
	id_number=[]
	for names in fits_files:
		if instru=='Map':
			id_number.append(names[18:28])
			long_number.append(names[46:59])
		if instru=='Sparse':
			id_number.append(names[18:28])
			long_number.append(names[51:64])
		if instru=='PACS':
			id_number.append(names[5:15])
			long_number.append(names[32:45])
		
	return long_number, id_number

def reorganiser_long_number(instru,resol=None,WVL=None,Couleur=None):   					#Cette fonction permet d'avoir la liste des long_number ordonnée suivant le OBS_ID et le OBJECT car os.listdir() désordonne le tout
	long_number=recup_long_number_Map(instru,resol,WVL,Couleur)[0]
	id_number=recup_long_number_Map(instru,resol,WVL,Couleur)[1]
	long_number2=[]
	index_id_number=[]
	if instru=='Map':
		for obs_id in OBS_ID_SPIRE_MAP:
			index_id_number.append(id_number.index(str(obs_id)))
	if instru=='Sparse':
		for obs_id in OBS_ID_SPIRE_SPARSE:
			index_id_number.append(id_number.index(str(obs_id)))
	if instru=='PACS':
		for obs_id in OBS_ID_PACS:
			index_id_number.append(id_number.index(str(obs_id)))
	for i in index_id_number:
		long_number2.append(long_number[i])
	
	return index_id_number, long_number2


def read_data(image):     #Fonction permettant de lire un fichier FITS et retourner la liste des headers
	hdulist=fits.open(direc1+image)
	return hdulist

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


def csv_to_dict(file):			#Fonction qui convertit le fichier csv en dictionnaire
	reader = csv.DictReader(open(direc1+file), delimiter=';', quoting=csv.QUOTE_NONE)
	result = {}
	for row in reader:
		key = row.pop('Object')
		if key in result:
			pass
		result[key] = row
	
	return result

######################################################INFO/DATA#######################################################################
######################################################################################################################################

def info_PACS(image):								# Retourne un tableau (voir éléments dans le return)
	hdulist=read_data(image)
	Object=hdulist[0].header['OBJECT']						
	Pixel_size_deg=abs(hdulist[1].header['CDELT1']*hdulist[1].header['CDELT2'])	#en deg^2
	#Pixel_size_sr=Pixel_size_deg*(3600*3600)/(4.25E10)			#1deg^2=3600*3600 arcesec^2 and 1sr=1rad^2=4.25E10 arcsec2
	#Pixel_size_sr=(Pixel_size_deg*u.deg*u.deg).to(u.sr)			
	beam=Pixel_size_deg*u.deg*u.deg
	Flux_Jy=(hdulist[1].data*u.Jy/beam).to(u.Jy/u.sr)				# En Jy/px > Jy/sr= W.m-2.Hz-1.sr-1
	Flux_Jy=Flux_Jy.value
	c=3E8*1E6 														#To be in microns.s-1 because wave is in microns

	#Flux=Flux/1E6 												# Jy/sr > MJy/sr	
	
	wvl=np.arange((hdulist[1].header['NAXIS3']))	# En microns
	wave=hdulist[1].header['CRVAL3']+hdulist[1].header['CDELT3']*wvl				
	
	# Jy/sr= W.m-2.Hz-1.sr-1 >  W.m-2.um-1.sr-1
	Flux=np.zeros_like(Flux_Jy)
	for i in range(np.shape(Flux)[1]):
		for j in range(np.shape(Flux)[2]):
			Flux[:,i,j]=Flux_Jy[:,i,j]*1E-26*c/(wave**2)					#I(freq) > I(lambda): W.m-2.sr-1.um-1

	Flux_mean_pixel=np.nanmean(Flux,axis=(0))		#On a une image moyennée sur 5*5 pixels (moyenne des 1900 points dans chaque pixel)
	Flux_mean_spectrum=np.nanmean(Flux,axis=(1,2))		# On a un spectre moyenné sur 1900 points(moyenne des 25 pixels pour chaque point)
	wcs=WCS(hdulist[1].header,naxis=2)
	return Object, Flux, Flux_mean_pixel, Flux_mean_spectrum, wave, wcs

def info_SPIRE_MAP(image):
	hdulist=read_data(image)
	Objet=hdulist[0].header['OBJECT']
	Flux=hdulist[1].data
	wvl=np.arange((hdulist[1].header['NAXIS3']))
	wave=hdulist[1].header['CRVAL3']+hdulist[1].header['CDELT3']*wvl
	Flux_mean_pixel=np.nanmean(Flux,axis=(0))		
	Flux_mean_spectrum=np.nanmean(Flux,axis=(1,2))	
	wcs=WCS(hdulist[1].header,naxis=2)
	return Objet, Flux, Flux_mean_pixel, Flux_mean_spectrum, wave, wcs


def info_ID(type):						#type=PACS/Sparse/Map
	if type=='Map':
		objet,obs_id,HR_SLW,HR_SSW,LR_SLW,LR_SSW=[],[],[],[],[],[]
		with open (direc1+'/SPIRE_MAPPING/SPIRE_MAP.txt', "r") as file:
			for line in file:
				line=line.strip()
				if line:
					obj, obs,hr_slw,hr_ssw,lr_slw,lr_ssw = [elt for elt in line.split("\t")]
					objet.append(obj), obs_id.append(obs)
					HR_SLW.append(hr_slw), HR_SSW.append(hr_ssw) ,LR_SLW.append(lr_slw), LR_SSW.append(lr_ssw)

		return objet,obs_id,HR_SLW,HR_SSW,LR_SLW,LR_SSW
	if type=='Sparse':
		objet,obs_id,HR,LR=[],[],[],[]
		with open (direc1+'/SPIRE_SPARSE/SPIRE_SPARSE.txt', "r") as file:
			for line in file:
				line=line.strip()
				if line:
					obj, obs,hr,lr = [elt for elt in line.split("\t")]
					objet.append(obj), obs_id.append(obs)
					HR.append(hr), LR.append(lr)

		return objet,obs_id,HR,LR
	if type=='PACS':
		objet,obs_id,B,R=[],[],[],[]
		with open (direc1+'/PACS/PACS.txt', "r") as file:
			for line in file:
				line=line.strip()
				if line:
					obj, obs,b,r = [elt for elt in line.split("\t")]
					objet.append(obj), obs_id.append(obs)
					B.append(b), R.append(r)

		return objet,obs_id,B,R


def SPARSE_spectrum(source_name):					#Cette fonction récupère le spectre du détecteur central, fait une moyenne de tous les autres en tant que "Background" puis calcule un spectre final 
	hdulist_SSW=open_source(source_name,'Sparse','SSW')
	hdulist_SLW=open_source(source_name,'Sparse','SLW')
	SLW_central_flux,SLW_central_wave=hdulist_SLW['SLWC3'].data['flux'],hdulist_SLW['SLWC3'].data['wave']		#11: Correspondant au détecteur SLWC3
	SSW_central_flux,SSW_central_wave=hdulist_SSW['SSWD4'].data['flux'],hdulist_SSW['SSWD4'].data['wave']		#39: Correspondant au détecteur SSWD4
	Bkg_SLW_flux=[]
	Bkg_SSW_flux=[]
	for slw in range (2,21):
		if slw==11:												#On exclut le détecteur central dans le calcul du Background
			continue 
		Bkg_SLW_flux.append(hdulist_SLW[slw].data['flux'])
	for ssw in range (21,56):
		if slw==39:
			continue 
		Bkg_SSW_flux.append(hdulist_SSW[ssw].data['flux'])

	Bkg_SLW_flux=np.mean(np.array(Bkg_SLW_flux),axis=0)			#Calcul du Background en faisant la moyenne de tous les autres détecteurs
	Bkg_SSW_flux=np.mean(np.array(Bkg_SSW_flux),axis=0)

	SLW_spectrum=SLW_central_flux-Bkg_SLW_flux					#Spectre final sans le background
	SSW_spectrum=SSW_central_flux-Bkg_SSW_flux

	return SLW_central_flux, SLW_spectrum, Bkg_SLW_flux, SLW_central_wave, SSW_central_flux, SSW_spectrum, Bkg_SSW_flux, SSW_central_wave


##############################################PLOTS TOUTES LES SOURCES/TOUS DÉTECTEURS###############################################################################
######################################################################################################################################

def plot_tot_SPIRE_Map(resol,WVL,type):			#Resol=HR/LR, WVL= SSW/SLW, type=image ou spectre
	path='SPIRE_MAPPING/'+str(resol)+'_'+str(WVL)+'/'
	files=os.listdir(direc1+path)
	fits_files = []
	for names in files:
		if names.endswith(".fits"):
			fits_files.append(names)
	position=1
	for file in fits_files:
		Flux=info_SPIRE_MAP(path+file)[1]
		Object=info_SPIRE_MAP(path+file)[0]
		if type=='image':
			Flux_mean=info_SPIRE_MAP(path+file)[2]
			plt.subplot(3,3,position)
			plt.imshow(Flux_mean,origin='lower')
			plt.title(str(Object))
			plt.colorbar()
		else:
			Flux_mean=info_SPIRE_MAP(path+file)[3]
			Wave=info_SPIRE_MAP(path+file)[4]
			plt.subplot(3,3,position)
			plt.plot(Wave,Flux_mean)
			plt.ylim(0,1E-16)							#On les met toutes à la même échelle pour avoir une idée des sources très brillantes
			plt.xlabel('Frequency (GHz)')
			plt.ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
			plt.title(str(Object))
		position+=1
	plt.suptitle('Plots de toutes les sources (moyennées sur '+str(len(Flux))+' points) observé avec SPIRE en '+str(resol)+'_'+str(WVL))	
	plt.tight_layout()
	plt.show()
	
	return 
	
def plot_SPIRE_SPARSE(source_name,resol,WVL,type='spectre'):
	path='SPIRE_SPARSE/'+str(resol)+'_spectrum_ext/'
	index_obj=info_ID('Sparse')[0].index(source_name)
	obs_id=info_ID('Sparse')[1]
	long_number=[info_ID('Sparse')[2],info_ID('Sparse')[3]]
	long_number_final=[]
	a=0
	if resol=='HR':
		long_number_final=long_number[0]
		a=1
	if resol=='LR':
		long_number_final=long_number[1]
		a=2
	file='hspirespectrometer'+str(obs_id[index_obj])+'_a106000'+str(a)+'_spg_'+str(resol)+'_20sds_'+str(long_number_final[index_obj])+'.fits'
	hdulist=read_data(path+file)
	
	if type=='spectre':
		plt.figure()
		for i in range (2,21):     #2-21: correspond aux headers des détecteurs SLW = 19 détecteurs
			Flux=hdulist[i].data['flux']
			Wave=hdulist[i].data['wave']
			Detector=hdulist[i].header['CHNLNAME']
			plt.subplot(5,4,i-1)
			plt.plot(Wave,Flux)
			#plt.ylim(np.min(hdulist[11].data['flux'])-1E-18,np.max(hdulist[11].data['flux'])+1E-18)   #On met tous les graphes à la même échelle de flux en prenant pour base le détecteur central
			plt.title(str(Detector))
		plt.suptitle('Spectres correspondant aux 19 détecteurs SLW de la source '+str(source_name))
		plt.tight_layout(pad=0.95, w_pad=0.02, h_pad=0.02)
		plt.show()
		
		plt.figure()
		for i in range (30,50):     #2-21: correspond aux headers des détecteurs SLW = 19 détecteurs
			Flux=hdulist[i].data['flux']
			Wave=hdulist[i].data['wave']			
			Detector=hdulist[i].header['CHNLNAME']
			plt.subplot(5,4,i-29)
			plt.plot(Wave,Flux)
			#plt.ylim(np.min(hdulist[39].data['flux'])-0.5E-17,np.max(hdulist[39].data['flux'])+0.5E-17)  #On met tous les graphes à la même échelle de flux en prenant pour base le détecteur central
			plt.title(str(Detector))
		plt.suptitle('Spectres correspondant aux 20 détecteurs (parmi les 37) SSW de la source '+str(source_name))
		plt.tight_layout(pad=0.95, w_pad=0.02, h_pad=0.02)
		plt.show()

	return 



##############################################PLOTS INDIVIDUELS SPIRE_MAP (par source)########################################################################
######################################################################################################################################



def plot_image_flux_SPIRE_Map(source_name,resol,WVL,type=None): 			#On obtient pour une source, une résolution, un range (WVL=SSW ou SLW) donnés, un plot de l'image ou du spectre 
	index_obj=info_ID('Map')[0].index(source_name)
	obs_id=info_ID('Map')[1]
	path='SPIRE_MAPPING/'+str(resol)+'_'+str(WVL)+'/'         #On sous-entend direc1 (qui est déjà présent dans le read_data)
	long_number=[info_ID('Map')[2],info_ID('Map')[3],info_ID('Map')[4],info_ID('Map')[5]]
	long_number_final=[]
	if resol=='HR' and WVL=='SLW':
		long_number_final=long_number[0]
	if resol=='HR' and WVL=='SSW':
		long_number_final=long_number[1]
	if resol=='LR' and WVL=='SLW':
		long_number_final=long_number[2]
	if resol=='LR' and WVL=='SSW':
		long_number_final=long_number[3]
	file='hspirespectrometer'+str(obs_id[index_obj])+'_spg_'+str(WVL)+'_'+str(resol)+'_20ssc_'+str(long_number_final[index_obj])+'.fits'
	Objet,Flux, Flux_mean_image,Flux_mean_spectre,Wave,wcs =info_SPIRE_MAP(path+file)   
	#Wave=3E8/(Wave*1E3)
	if type=='image':
		fig=plt.figure()
		fig.add_subplot(111,projection=wcs)
		plt.imshow(Flux_mean_image,origin='lower',cmap=plt.cm.viridis)
		plt.xlabel('RA')
		plt.ylabel('Dec')
		plt.title(str(source_name))
		plt.colorbar()
	if type=='spectre':
		plt.plot(Flux_mean_spectre)
		plt.xlabel('Arbitrary')
		plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}$)')
		plt.title(str(source_name))
	#plt.suptitle('Plots de toutes les sources (moyennées sur '+str(len(Flux))+' points) observé avec SPIRE en '+str(resol)+'_'+str(WVL))	
	#plt.tight_layout()
	plt.show()
	
	return Flux_mean_image , Flux_mean_spectre, Wave, wcs, Flux


def plot_SPIRE_MAP(source_name):				#On obtient 4 plots : les images et les spectres, en HR et en LR (SSW et SLW "recollés")
	Flux_mean_HR_SLW_image,Flux_mean_HR_SLW_spectre, Wave_HR_SLW, wcs_HR_SLW, Flux_HR_SLW=plot_image_flux_SPIRE_Map(source_name,'HR','SLW')
	Flux_mean_HR_SSW_image,Flux_mean_HR_SSW_spectre, Wave_HR_SSW, wcs_HR_SSW, Flux_HR_SSW=plot_image_flux_SPIRE_Map(source_name,'HR','SSW')
	Flux_mean_LR_SLW_image,Flux_mean_LR_SLW_spectre, Wave_LR_SLW, wcs_LR_SLW, Flux_LR_SLW=plot_image_flux_SPIRE_Map(source_name,'LR','SLW')
	Flux_mean_LR_SSW_image,Flux_mean_LR_SSW_spectre, Wave_LR_SSW, wcs_LR_SSW, Flux_LR_SSW=plot_image_flux_SPIRE_Map(source_name,'LR','SSW')
	fig=plt.figure()
	ax1=fig.add_subplot(221,projection=wcs_HR_SSW,aspect='equal')
	plt.imshow(Flux_mean_HR_SSW_image,origin='lower')
	ax1.coords[0].set_ticks(exclude_overlapping=True)
	plt.colorbar(label='Flux')
	plt.xlabel('RA (J2000)')
	plt.ylabel('Dec (J2000)')
	plt.title('High Resolution SSW ', fontsize=10)
	fig.add_subplot(222)
	plt.plot(Wave_HR_SSW,Flux_mean_HR_SSW_spectre,label='SSW')
	plt.plot(Wave_HR_SLW,Flux_mean_HR_SLW_spectre,label='SLW')
	plt.xlabel('Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	plt.legend()
	plt.title('High Resolution total spectrum(SSW+SLW)',fontsize=10)
	ax2=fig.add_subplot(223,projection=wcs_HR_SLW)
	ax2.coords[0].set_ticks(exclude_overlapping=True)
	plt.imshow(Flux_mean_HR_SLW_image,origin='lower')
	plt.xlabel('RA (J2000)')
	plt.ylabel('Dec (J2000)')
	plt.colorbar(label='Flux')
	plt.title('High Resolution SLW ',fontsize=10)
	fig.add_subplot(224)
	plt.plot(Wave_LR_SSW,Flux_mean_LR_SSW_spectre,label='SSW')
	plt.plot(Wave_LR_SLW,Flux_mean_LR_SLW_spectre,label='SLW')
	plt.xlabel('Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	plt.title('Low Resolution total spectrum(SSW+SLW)',fontsize=10)
	plt.suptitle('Object:'+str(source_name)+' observed with SPIRE-FTS spectrometer')
	plt.legend()
	plt.tight_layout()
	#plt.tight_layout(pad=0.90, w_pad=0.03, h_pad=0.01)
	plt.show()

	return wcs_HR_SSW

#def SPIRE_MAP(source_name,resol,WVL):

####################################################SPIRE_SPARSE############################################################################
############################################################################################################################################


def plot_final_SPIRE_SPARSE(source_name):     #On trace le spectre des détecteurs centraux, une moyenne des autres détecteurs (Background) et le spectre final : central - background
	#path='SPIRE_SPARSE/'+str(resol)+'_spectrum_ext/'
	path='SPIRE_SPARSE/HR_spectrum_ext/'
	index_obj=info_ID('Sparse')[0].index(source_name)
	obs_id=info_ID('Sparse')[1]
	long_number=[info_ID('Sparse')[2],info_ID('Sparse')[3]]
	long_number_final=[]
	a=1
	long_number_final=long_number[0]
	'''
	if resol=='HR':
		long_number_final=long_number[0]
		a=1
	if resol=='LR':
		long_number_final=long_number[1]
		a=2'''
	file='hspirespectrometer'+str(obs_id[index_obj])+'_a106000'+str(a)+'_spg_HR_20sds_'+str(long_number_final[index_obj])+'.fits'
	SLW_central_flux, SLW_spectrum, Bkg_SLW_flux, SLW_central_wave, SSW_central_flux, SSW_spectrum, Bkg_SSW_flux, SSW_central_wave=SPARSE_spectrum(source_name)
	
	#SLW_central_wave=3E8/(SLW_central_wave*1E3)
	#SSW_central_wave=3E8/(SSW_central_wave*1E3)

	fig = plt.figure(0) 
	fig.suptitle('Source: '+str(source_name)+' observed with SPIRE in SPARSE Mode and High Resolution')

	gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
	ax1 = plt.subplot(gs[0, 0])
	ax2 = plt.subplot(gs[1, 0])
	ax3 = plt.subplot(gs[0:, 1])

	ax1.plot(SLW_central_wave,SLW_central_flux,label='SLWC3')
	ax1.plot(SSW_central_wave,SSW_central_flux,label='SSWD4')
	ax1.set_title('Central detectors SLWC3 + SLWD4', fontsize=11)
	ax1.set_xlabel('Frequency (GHz)')
	ax1.set_ylim(np.min(SLW_central_flux)-1E-17,np.max(SSW_central_flux)+0.5E-17)  #On met les mêmes échelles de flux en prenant le min des SLW et le max des SSW.
	ax1.set_ylabel(r'Flux($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	ax1.legend()


	ax2.plot(SLW_central_wave,Bkg_SLW_flux,label='SLW')
	ax2.plot(SSW_central_wave,Bkg_SSW_flux,label='SSW')
	ax2.set_title('Mean background (other detectors)',fontsize=11)
	ax2.set_xlabel('Frequency (GHz)')
	ax2.set_ylim(np.min(SLW_central_flux)-1E-17,np.max(SSW_central_flux)+0.5E-17)
	ax2.set_ylabel(r'Flux($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	ax2.legend()

	ax3.plot(SLW_central_wave,SLW_spectrum,label='SLW')
	ax3.plot(SSW_central_wave,SSW_spectrum,label='SSW')
	ax3.set_title('Central Detector - Mean Background',fontsize=11)
	ax3.set_xlabel('Frequency (GHz)')
	ax3.set_ylim(np.min(SLW_central_flux)-1E-17,np.max(SSW_central_flux)+0.5E-17)
	ax3.set_ylabel(r'Flux($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
	ax3.legend()

	plt.show()
	
	return




############################################################   PACS   ######################################################################
############################################################################################################################################

def flux_PACS(source_name,color):		#On trace le spectre des détecteurs centraux, une moyenne des autres détecteurs (Background) et le spectre final : central - background
										#color=B/R									
	path='PACS/HPS3DEQI'+str(color)+'/'
	index_obj=info_ID('PACS')[0].index(source_name)
	obs_id=info_ID('PACS')[1]
	long_number=[info_ID('PACS')[2],info_ID('PACS')[3]]
	long_number_final=[]
	couleur=0
	if color=='B':
		long_number_final=long_number[0]
		couleur='b'
	if color=='R':
		long_number_final=long_number[1]
		couleur='r'
	file='hpacs'+str(obs_id[index_obj])+'_20hps3deqi'+str(couleur)+'s_00_'+str(long_number_final[index_obj])+'.fits'
	Objet,Flux, Flux_mean_image,Flux_mean_spectre,Wave,wcs =info_PACS(path+file)

	return Flux_mean_image,Flux_mean_spectre,Wave,wcs,Flux

	
def plot_PACS(source_name):
	Flux_image_B, Flux_spectrum_B, Wave_B, wcs_B, Flux_blue= flux_PACS(source_name,'B')
	Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R, Flux_red= flux_PACS(source_name,'R')

	fig = plt.figure(0) 
	fig.suptitle('Object '+str(source_name)+' observed with PACS (Spectrometer)')

	gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
	ax1 = plt.subplot(gs[0, 0], projection=wcs_B)
	ax2 = plt.subplot(gs[0, 1], projection=wcs_R)
	ax3 = plt.subplot(gs[1, 0:])

	#ax1=fig.add_subplot(221,projection=wcs_HR_SSW,aspect='equal')
	#plt.imshow(Flux_mean_HR_SSW_image,origin='lower')

	im1=ax1.imshow(Flux_image_B,origin='lower')
	ax1.coords[0].set_ticks(exclude_overlapping=True)
	fig.colorbar(im1, ax=ax1, label="Flux (MJy/sr)", aspect=20)
	ax1.set_xlabel('RA (J2000)')
	ax1.set_ylabel('Dec (J2000)')
	ax1.set_title(r'B2A Band image : 55-72 $\mu$m')

	im2=ax2.imshow(Flux_image_R,origin='lower')
	ax2.coords[0].set_ticks(exclude_overlapping=True)
	fig.colorbar(im2, ax=ax2, label="Flux (MJy/sr)", aspect=20)
	ax2.set_xlabel('RA (J2000)')
	ax2.set_ylabel('Dec (J2000)')
	ax2.set_title(r'R1 Band image : 102-146 $\mu$m')
	
	ax3.plot(Wave_B,Flux_spectrum_B,label='B2A band')#, format='%.2e')
	ax3.plot(Wave_R,Flux_spectrum_R,label='R1 band')#,format='%.2e')
	#ax3.xaxis.set_ticks(range(5))
	ax3.set_title(r'Mean spectrum B2A $+$ R1 bands')
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax3.set_xlabel(r'Wavelength ($\mu$m)')
	ax3.set_ylabel(r'Flux ($W.m^{-2}.sr^{-1}.$$\mu$$m^{-1}$)')
	ax3.legend()
	#ax3.grid()

	#plt.savefig(str(source_name)+'_PACS.pdf',dpi=150)
	#plt.show()

	plt.figure()
	plt.plot(Wave_B,Flux_spectrum_B,label='B2A band')#, format='%.2e')
	plt.plot(Wave_R,Flux_spectrum_R,label='R1 band')
	plt.title(r'Mean spectrum B2A $+$ R1 bands')
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.$$\mu$$m^{-1}$)')
	plt.legend()

	plt.show()

	
	return




