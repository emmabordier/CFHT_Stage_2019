import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import  GridSpec
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
import os as os
import csv

direc1='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/Data/'
OBS_ID_PACS=[1342226189, 1342231302, 1342231303, 1342231738, 1342231739, 1342231740, 1342231741, 1342231742, 1342231743, 1342231744, 1342231745, 1342231746, 1342231747, 1342238730, 1342239689, 1342239741, 1342239757, 1342240165, 1342240166, 1342240167, 1342241267, 1342242444, 1342242445, 1342242446, 1342243105, 1342243106, 1342243107, 1342243108, 1342243505, 1342243506, 1342243507, 1342243513, 1342243901, 1342250917, 1342252270]
OBS_ID_SPIRE_MAP=[1342262927, 1342254039, 1342254040, 1342254041, 1342254041, 1342262919, 1342262924, 1342262926, 1342265807]
OBS_ID_SPIRE_SPARSE=[1342253970, 1342253971, 1342262922, 1342262923, 1342262925, 1342265810, 1342268284, 1342268285, 1342268286]
OBJECT_PACS=['MGE_4384', 'MGE_3438', 'MGE_3448', 'MGE_3269', 'MGE_3280', 'MGE_3739', 'MGE_3736', 'MGE_3719', 'MGE_3222', 'MGE_3354', 'MGE_3360', 'MGE_3681', 'MGE_3670', 'MGE_4048', 'MGE_4191', 'MGE_4206', 'MGE_4134', 'MGE_4121', 'MGE_4095', 'MGE_4218', 'MGE_3899', 'MGE_4552', 'MGE_4436', 'MGE_4486', 'MGE_4485', 'MGE_4111', 'MGE_4110', 'MGE_4204', 'MGE_3149', 'MGE_4602', 'MGE_4473', 'MGE_4524', 'MGE_3834', 'MGE_4239', 'MGE_4167']
OBJECT_SPIRE_MAP=['MGE_4121', 'MGE_3269', 'MGE_3280', 'MGE_3739', 'MGE_3739', 'MGE_4384', 'MGE_4204', 'MGE_4111', 'MGE_4485']
OBJECT_SPIRE_SPARSE=['MGE_3681', 'MGE_3448', 'MGE_4048', 'MGE_4206', 'MGE_4095', 'MGE_4134', 'MGE_3149', 'MGE_4602', 'MGE_4524']


def writefits(array, fname, overwrite=False):    #Fonction permettant d'écrire un fichier Fits. Attention au type de Header (PrimaryHDU, ImageHDU, BinTableHDU)

  if (os.path.isfile(fname)):
    if (overwrite == False):
      print("File already exists!, skiping")
      return
    else:
      print("File already exists!, overwriting")
      os.remove(fname)
  try: 
    hdu = fits.ImageHDU(array)
    hdu.writeto(fname)
  except:
    print('b1') 
    hdu = pf.ImageHDU(fname)
    print('b2') 
    hdu.writeto(array)

  print("File: %s saved!" % fname)

  return


def read_data(image):     #Fonction permettant de lire un fichier FITS et retourner la liste des headers
	hdulist=fits.open(direc1+image)
	return hdulist


##############################################INFO/DATA###############################################################################
######################################################################################################################################

def info_PACS(image):								# Retourne un tableau (voir éléments dans le return)
	hdulist=read_data(image)
	Objet=hdulist[0].header['OBJECT']
	Flux=hdulist[1].data 							# En Jy/pixel
	Wvl=hdulist[2].data 							# En microns						
	Flux_mean_pixel=np.nanmean(Flux,axis=(0))		#On a une image moyennée sur 5*5 pixels (moyenne des 1900 points dans chaque pixel)
	Flux_mean_point=np.nanmean(Flux,axis=(1,2))		# On a un spectre moyenné sur 1900 points(moyenne des 25 pixels pour chaque point)
	return Objet, Flux, Wvl, Flux_mean_pixel, Flux_mean_point

def info_SPIRE_MAP(image):
	hdulist=read_data(image)
	Objet=hdulist[0].header['OBJECT']
	Flux=hdulist[1].data
	wvl=np.arange((hdulist[1].header['NAXIS3']))
	wave=hdulist[1].header['CRVAL3']+hdulist[1].header['CDELT3']*wvl
	Flux_mean_pixel=np.nanmean(Flux,axis=(0))		
	Flux_mean_point=np.nanmean(Flux,axis=(1,2))	
	wcs=WCS(hdulist[1].header,naxis=2)
	return Objet, Flux, Flux_mean_pixel, Flux_mean_point, wave, wcs

#def info_SPIRE_SPARSE(image):
#	hdulist=read_data(image)

def info_ID_SPIRE_MAP():
	objet,obs_id,HR_SLW,HR_SSW,LR_SLW,LR_SSW=[],[],[],[],[],[]
	with open (direc1+'/SPIRE_MAPPING/SPIRE_MAP.txt', "r") as file:
		for line in file:
			line=line.strip()
			if line:
				obj, obs,hr_slw,hr_ssw,lr_slw,lr_ssw = [elt for elt in line.split("\t")]
				objet.append(obj), obs_id.append(obs)
				HR_SLW.append(hr_slw), HR_SSW.append(hr_ssw) ,LR_SLW.append(lr_slw), LR_SSW.append(lr_ssw)

	return objet,obs_id,HR_SLW,HR_SSW,LR_SLW,LR_SSW

def info_ID_SPIRE_SPARSE():
	objet=[]
	obs_id=[]
	HR=[]
	LR=[]
	with open (direc1+'/SPIRE_SPARSE/SPIRE_SPARSE.txt', "r") as file:
		for line in file:
			line=line.strip()
			if line:
				obj, obs,hr,lr = [elt for elt in line.split("\t")]
				objet.append(obj), obs_id.append(obs)
				HR.append(hr), LR.append(lr)

	return objet,obs_id,HR,LR

def SPARSE_spectrum(image):
	hdulist=read_data(image)
	SLW_central_flux,SLW_central_wave=hdulist[11].data['flux'],hdulist[11].data['wave']		#Correspondant au détecteur SLWC3
	SSW_central_flux,SSW_central_wave=hdulist[39].data['flux'],hdulist[39].data['wave']		#Correspondant au détecteur SSWD4
	Bkg_SLW_flux=[]
	Bkg_SSW_flux=[]
	for slw in range (2,21):
		if slw==11:
			continue 
		Bkg_SLW_flux.append(hdulist[slw].data['flux'])
	for ssw in range (21,56):
		if slw==39:
			continue 
		Bkg_SSW_flux.append(hdulist[ssw].data['flux'])

	Bkg_SLW_flux=np.mean(np.array(Bkg_SLW_flux),axis=0)
	Bkg_SSW_flux=np.mean(np.array(Bkg_SSW_flux),axis=0)

	SLW_spectrum=SLW_central_flux-Bkg_SLW_flux
	SSW_spectrum=SSW_central_flux-Bkg_SSW_flux

	return SLW_central_flux, SLW_spectrum, Bkg_SLW_flux, SLW_central_wave, SSW_central_flux, SSW_spectrum, Bkg_SSW_flux, SSW_central_wave



def recup_long_number_Map(resol,WVL,SPIRE_type):			#Fichier=PACS,SPIRE_MAPPING,SPIRE_SPARSE. SPIRE_TYPE=Map ou Sparse
	if SPIRE_type=='Map':
		path='SPIRE_MAPPING/'+str(resol)+'_'+str(WVL)+'/'
	else:
		path='SPIRE_SPARSE/'+str(resol)+'_spectrum_ext/'
	files=os.listdir(direc1+path)
	fits_files = []
	for names in files:
		if names.endswith(".fits"):
			fits_files.append(names)
	long_number=[]
	id_number=[]
	for names in fits_files:
		id_number.append(names[18:28])
		if SPIRE_type=='Map':
			long_number.append(names[46:59])
		else:
			long_number.append(names[51:64])
		
	return long_number, id_number

def reorganiser_long_number(resol,WVL,SPIRE_type):   					#Cette fonction permet d'avoir la liste des long_number ordonnée suivant le OBS_ID et le OBJECT car os.listdir() désordonne le tout
	long_number=recup_long_number_Map(resol,WVL,SPIRE_type)[0]
	id_number=recup_long_number_Map(resol,WVL,SPIRE_type)[1]
	long_number2=[]
	index_id_number=[]
	if SPIRE_type=='Map':
		for obs_id in OBS_ID_SPIRE_MAP:
			index_id_number.append(id_number.index(str(obs_id)))
	else:
		for obs_id in OBS_ID_SPIRE_SPARSE:
			index_id_number.append(id_number.index(str(obs_id)))
	for i in index_id_number:
		long_number2.append(long_number[i])
	
	return index_id_number, long_number2


##############################################PLOTS TOUTES LES SOURCES###############################################################################
######################################################################################################################################


def plot_image_PACS(WVL):			#Mettre R ou B (petite ou grande longueur d'onde)
	position=1
	for obs_id in OBS_ID_PACS:
		Flux=info_PACS('PACS/HPS3D'+str(WVL)+'/hpacs'+str(obs_id)+'_20hps3dbs_00_.fits')[3]
		Object=info_PACS('PACS/HPS3D'+str(WVL)+'/hpacs'+str(obs_id)+'_20hps3dbs_00_.fits')[0]
		plt.subplot(9,4,position)
		plt.imshow(Flux,origin='lower')
		plt.title(str('Object'))
		position+=1
	plt.tight_layout()
	plt.show()



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
		#Wave=info_SPIRE_MAP(path+file)[4]
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
			plt.xlabel('Frequency (GHz)')
			plt.ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
			plt.title(str(Object))
		position+=1
	plt.suptitle('Plots de toutes les sources (moyennées sur '+str(len(Flux))+' points) observé avec SPIRE en '+str(resol)+'_'+str(WVL))	
	plt.tight_layout()
	plt.show()
	
	return 
	
	#for file in fits_files:
	#	Flux=info_PACS(path+'file')[3]

def plot_SPIRE_SPARSE(nom_source,resol,WVL,type='spectre'):
	path='SPIRE_SPARSE/'+str(resol)+'_spectrum_ext/'
	index_obj=info_ID_SPIRE_SPARSE()[0].index(nom_source)
	obs_id=info_ID_SPIRE_SPARSE()[1]
	long_number=[info_ID_SPIRE_SPARSE()[2],info_ID_SPIRE_SPARSE()[3]]
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
			#plt.xlabel('Frequency (GHz)')
			#plt.ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
			plt.title(str(Detector))
		plt.suptitle('Spectres correspondant aux 19 détecteurs SLW de la source '+str(nom_source))
			#plt.tight_layout()
		plt.show()
		
		plt.figure()
		for i in range (30,50):     #2-21: correspond aux headers des détecteurs SLW = 19 détecteurs
			Flux=hdulist[i].data['flux']
			Wave=hdulist[i].data['wave']			
			Detector=hdulist[i].header['CHNLNAME']
			plt.subplot(5,4,i-29)
			plt.plot(Wave,Flux)
			#plt.xlabel('Frequency (GHz)')
			#plt.ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
			plt.title(str(Detector))
		plt.suptitle('Spectres correspondant aux 20 détecteurs (parmi les 37) SLW de la source '+str(nom_source))
			#plt.tight_layout()
		plt.show()

	return #hdulist.info()



##############################################PLOTS INDIVIDUELS SPIRE_MAP (par source)########################################################################
######################################################################################################################################

#Faire fonction qui affiche IMAGE+spectre en HR et LR + Faire la correspondance unités en abscisse avec les données du header 

#Arguments : nom de la source + 

#Ecrire un fichier csv avec toutes les sources ET PACS/SPIRE



def plot_image_flux_SPIRE_Map(nom_source,resol,WVL,type=None): 			#On obtient pour une source, une résolution, un range (WVL=SSW ou SLW) donnés, un plot de l'image ou du spectre 
	index_obj=info_ID_SPIRE_MAP()[0].index(nom_source)
	obs_id=info_ID_SPIRE_MAP()[1]
	path='SPIRE_MAPPING/'+str(resol)+'_'+str(WVL)+'/'         #On sous-entend direc1 (qui est déjà présent dans le read_data)
	long_number=[info_ID_SPIRE_MAP()[2],info_ID_SPIRE_MAP()[3],info_ID_SPIRE_MAP()[4],info_ID_SPIRE_MAP()[5]]
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
	#Flux=info_SPIRE_MAP(path+file)[1]
	Flux_mean_image=info_SPIRE_MAP(path+file)[2]
	Flux_mean_spectre=info_SPIRE_MAP(path+file)[3]
	Wave=info_SPIRE_MAP(path+file)[4]
	wcs=info_SPIRE_MAP(path+file)[5]
	if type=='image':
		fig=plt.figure()
		fig.add_subplot(111,projection=wcs)
		plt.imshow(Flux_mean_image,origin='lower',cmap=plt.cm.viridis)
		plt.xlabel('RA')
		plt.ylabel('Dec')
		plt.title(str(nom_source))
		plt.colorbar()
	if type=='spectre':
		#Flux_mean=info_SPIRE_MAP(path+file)[3]
		#plt.subplot(3,3,position)
		plt.plot(Flux_mean_spectre)
		plt.xlabel('Arbitrary')
		plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}$)')
		plt.title(str(nom_source))
	#plt.suptitle('Plots de toutes les sources (moyennées sur '+str(len(Flux))+' points) observé avec SPIRE en '+str(resol)+'_'+str(WVL))	
	#plt.tight_layout()
	plt.show()
	
	return Flux_mean_image , Flux_mean_spectre, Wave, wcs


def plot_SPIRE_MAP(nom_source):				#On obtient 4 plots : les images et les spectres, en HR et en LR (SSW et SLW "recollés")
	Flux_mean_HR_SLW_image,Flux_mean_HR_SLW_spectre, Wave_HR_SLW, wcs_HR_SLW=plot_image_flux_SPIRE_Map(nom_source,'HR','SLW')
	Flux_mean_HR_SSW_image,Flux_mean_HR_SSW_spectre, Wave_HR_SSW, wcs_HR_SSW=plot_image_flux_SPIRE_Map(nom_source,'HR','SSW')
	Flux_mean_LR_SLW_image,Flux_mean_LR_SLW_spectre, Wave_LR_SLW, wcs_LR_SLW=plot_image_flux_SPIRE_Map(nom_source,'LR','SLW')
	Flux_mean_LR_SSW_image,Flux_mean_LR_SSW_spectre, Wave_LR_SSW, wcs_LR_SSW=plot_image_flux_SPIRE_Map(nom_source,'LR','SSW')
	fig=plt.figure()
	fig.add_subplot(221,projection=wcs_HR_SSW)
	plt.imshow(Flux_mean_HR_SSW_image,origin='lower')
	plt.colorbar()
	plt.xlabel('RA')
	plt.ylabel('Dec')
	plt.title(str(nom_source)+' en HR SSW')
	fig.add_subplot(222)
	plt.plot(Wave_HR_SSW,Flux_mean_HR_SSW_spectre,label='SSW')
	plt.plot(Wave_HR_SLW,Flux_mean_HR_SLW_spectre,label='SLW')
	plt.xlabel('Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}$)')
	plt.legend()
	plt.title(str(nom_source)+' en HR')
	fig.add_subplot(223,projection=wcs_HR_SLW)
	plt.imshow(Flux_mean_HR_SLW_image,origin='lower')
	plt.xlabel('RA')
	plt.ylabel('Dec')
	plt.colorbar()
	plt.title(str(nom_source)+' en HR SLW')
	fig.add_subplot(224)
	plt.plot(Wave_LR_SSW,Flux_mean_LR_SSW_spectre,label='SSW')
	plt.plot(Wave_LR_SLW,Flux_mean_LR_SLW_spectre,label='SLW')
	plt.xlabel('Frequency (GHz)')
	plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}$)')
	plt.title(str(nom_source)+' en LR')
	plt.suptitle('Source:'+str(nom_source))
	plt.legend()
	plt.tight_layout()
	plt.show()

#def SPIRE_MAP(nom_source,resol,WVL):

####################################################SPIRE_SPARSE##########################################################
##########################################################################################################################


def plot_final_SPIRE_SPARSE(nom_source,resol):
	path='SPIRE_SPARSE/'+str(resol)+'_spectrum_ext/'
	index_obj=info_ID_SPIRE_SPARSE()[0].index(nom_source)
	obs_id=info_ID_SPIRE_SPARSE()[1]
	long_number=[info_ID_SPIRE_SPARSE()[2],info_ID_SPIRE_SPARSE()[3]]
	long_number_final=[]
	a=0
	if resol=='HR':
		long_number_final=long_number[0]
		a=1
	if resol=='LR':
		long_number_final=long_number[1]
		a=2
	file='hspirespectrometer'+str(obs_id[index_obj])+'_a106000'+str(a)+'_spg_'+str(resol)+'_20sds_'+str(long_number_final[index_obj])+'.fits'
	SLW_central_flux, SLW_spectrum, Bkg_SLW_flux, SLW_central_wave, SSW_central_flux, SSW_spectrum, Bkg_SSW_flux, SSW_central_wave=SPARSE_spectrum(path+file)
	
	fig = plt.figure(0) 
	fig.suptitle('Source '+str(nom_source)+' observee en '+str(resol)+ ' avec SPIRE en mode SPARSE')

	gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
	ax1 = plt.subplot(gs[0, 0])
	ax2 = plt.subplot(gs[1, 0])
	ax3 = plt.subplot(gs[0:, 1])

	ax1.plot(SLW_central_wave,SLW_central_flux,label='SLWC3')
	ax1.plot(SSW_central_wave,SSW_central_flux,label='SSWD4')
	ax1.set_title('Detecteur central')
	ax1.set_xlabel('Frequency (GHz)')
	ax1.set_ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
	ax1.legend()


	ax2.plot(SLW_central_wave,Bkg_SLW_flux,label='SLW')
	ax2.plot(SSW_central_wave,Bkg_SSW_flux,label='SSW')
	ax2.set_title('Moyenne Background')
	ax2.set_xlabel('Frequency (GHz)')
	ax2.set_ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
	ax2.legend()

	ax3.plot(SLW_central_wave,SLW_spectrum,label='SLW')
	ax3.plot(SSW_central_wave,SSW_spectrum,label='SSW')
	ax3.set_title('Detecteur central - Moyenne background')
	ax3.set_xlabel('Frequency (GHz)')
	ax3.set_ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
	ax3.legend()

	plt.show()
	#return




def plot_tot_SPIRE_Map2(resol,WVL,type):			#Resol=HR/LR, WVL= SSW/SLW, type=image ou spectre
	path='SPIRE_MAPPING/'+str(resol)+'_'+str(WVL)+'/'
	files=os.listdir(direc1+path)
	fits_files = []
	for names in files:
		if names.endswith(".fits"):
			fits_files.append(names)
	position=1
	Flux=info_SPIRE_MAP(path+fits_files[0])[1]
	fig1= plt.figure()#constrained_layout=True)
	grid = GridSpec(nrows=3, ncols=3,left=0.1, bottom=0.15, right=0.8, top=0.8, wspace=0.2, hspace=0.2)
	#for file in fits_files:
		#Flux=info_SPIRE_MAP(path+file)[1]
		#Wave=info_SPIRE_MAP(path+file)[1]
		#Object=info_SPIRE_MAP(path+file)[0]
	if type=='image':     #Attention, on n'arrive pas à changer la source ici car c'est inclut dans la boucle!
		for row in range(3):
			for col in range(3):
				#Wave=info_SPIRE_MAP(path+file)[1]
				Object=info_SPIRE_MAP(path+fits_files[row+col])[0]
				ax = fig1.add_subplot(grid[row, col])
				ax.imshow(info_SPIRE_MAP(path+fits_files[row+col])[2],origin='lower')
	'''else:
		Flux_mean=info_SPIRE_MAP(path+fits_file)[3]
		Wave=info_SPIRE_MAP(path+fits_file)[4]
		plt.subplot(3,3,position)
		plt.plot(Wave,Flux_mean)
		plt.xlabel('Frequency (GHz)')
		plt.ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
		plt.title(str(Object))'''
	#position+=1
	plt.suptitle('Plots de toutes les sources (moyennées sur '+str(len(Flux))+' points) observé avec SPIRE en '+str(resol)+'_'+str(WVL))	
	plt.tight_layout()
	plt.show()
	
	return fits_files

def plot_SPIRE_SPARSE2(nom_source,resol,WVL):
	path='SPIRE_SPARSE/'+str(resol)+'_spectrum_ext/'
	index_obj=info_ID_SPIRE_SPARSE()[0].index(nom_source)
	obs_id=info_ID_SPIRE_SPARSE()[1]
	long_number=[info_ID_SPIRE_SPARSE()[2],info_ID_SPIRE_SPARSE()[3]]
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
	fig1= plt.figure()#constrained_layout=True)
	grid = GridSpec(nrows=4, ncols=5,left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
	if WVL=='SLW':
		for row in range(4):
			for col in range(5):
				Flux=hdulist[2+row+col].data['flux']
				Wave=hdulist[2+row+col].data['wave']
				Detector=hdulist[2+row+col].header['CHNLNAME']
				ax = fig1.add_subplot(grid[row, col])
				ax.plot(Wave,Flux)
				#ax.set_xlabel('Frequency (GHz)')
				#ax.set_ylabel(r'Flux($W.m^{-2}.sr^{-1}$)')
				plt.title(str(Detector))
	#if WVL=='SSW':

	plt.suptitle('Spectres correspondant aux 19 détecteurs SLW de la source '+str(nom_source))
	plt.show()

	return #hdulist.info()













