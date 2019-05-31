import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import  GridSpec
from collections import namedtuple
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
from Lines_Herschel import *

direc1='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/Data/'
direc='/Users/bordieremma/Documents/Magistere_3/STAGE_CFHT/'

detector_SLW=['SLWA1','SLWA2','SLWA3','SLWB1','SLWB2','SLWB3','SLWB4','SLWC1','SLWC2','SLWC3','SLWC4','SLWC5','SLWD1','SLWD2','SLWD3','SLWD4','SLWE1','SLWE2','SLWE3']
detector_SSW=['SSWA1','SSWA2','SSWA3','SSWA4','SSWB1','SSWB2','SSWB3','SSWB4','SSWB5','SSWC1','SSWC2','SSWC3','SSWC4','SSWC5','SSWC6','SSWD1','SSWD2','SSWD3','SSWD4','SSWD6','SSWD7','SSWE1','SSWE2','SSWE3','SSWE4','SSWE5','SSWE6','SSWF1','SSWF2','SSWF3','SSWF5','SSWG1','SSWG2','SSWG3','SSWG4']
central_detector_SLW='SLWC3'
central_detector_SSW='SSWD4'

def plot_map(source_name, Det_SSW=None, Det_SLW=None):
	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()

	Source_R=np.load(direc1+'MAPS3/'+str(source_name)+'_R.npy',allow_pickle=True).item()
	Source_B=np.load(direc1+'MAPS3/'+str(source_name)+'_B.npy',allow_pickle=True).item()
	WCS_R=Source_R['WCS']
	WCS_B=Source_B['WCS']
	lines_R=list(Source_R.keys())[6:]
	lines_B=list(Source_B.keys())[6:]


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
		size=np.shape(Source_R[line])
		ax=plt.subplot(2,2,a,projection=WCS_R)
		im=plt.imshow(Source_R[line])#(np.where(Source_R[line]<0.8*np.nanmax(Source_R[line]))))
		ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
		#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
		ax.set_xlim(-0.5,np.shape(Source_R[line])[1]-0.5)
		ax.set_ylim(-0.5,np.shape(Source_R[line])[0]-0.5)
		#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
		#ax.set_xlim(0,np.shape(Source_R[line])[0]-1)
		#ax.set_ylim(0,np.shape(Source_R[line])[1]-1)
		ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
		ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
		ax.coords[0].set_ticks(exclude_overlapping=True)
		plt.title(str(type_line)+' line '+str(line),fontsize=10)
		plt.colorbar(im, ax=ax , label=r"Flux ($W.m^{-2}.sr^{-1}$)",aspect=20)#,format='%.e')
		a+=1
		#plt.tight_layout()
	plt.suptitle('Line Flux maps for '+str(source_name)+' in PACS R-Band  Image size: '+str(size), fontsize=11,fontstyle='italic')
	#plt.tight_layout()
	plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=2.5)
	plt.show()

	a=1
	plt.figure()
	for line in lines_B:			#Que des Emission lines
		size=np.shape(Source_B[line])
		ax=plt.subplot(2,2,a,projection=WCS_B)
		im=plt.imshow(Source_B[line])
		ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
		#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
		ax.set_xlim(-0.5,np.shape(Source_B[line])[1]-0.5)
		ax.set_ylim(-0.5,np.shape(Source_B[line])[0]-0.5)
		#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
		#ax.set_xlim(0,np.shape(Source_B[line])[0]-1)
		#ax.set_ylim(0,np.shape(Source_B[line])[1]-1)
		ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
		ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
		ax.coords[0].set_ticks(exclude_overlapping=True)
		plt.title('Emission line '+str(line),fontsize=10)
		plt.colorbar(im, ax=ax, label=r"Flux ($W.m^{-2}.sr^{-1}$)")#format='%.e')
		#plt.colorbar()
		a+=1
		#plt.tight_layout()
	plt.suptitle('Line Flux maps for '+str(source_name)+' in PACS B-Band Image size: '+str(size),fontsize=11,fontstyle='italic')
	plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=2.5)
	plt.show()

	
	if int(my_dict[str(source_name)]['SPIRE_MAP'])==1:

		Source_SSW=np.load(direc1+'MAPS3/'+str(source_name)+'_SSW.npy',allow_pickle=True).item()
		Source_SLW=np.load(direc1+'MAPS3/'+str(source_name)+'_SLW.npy',allow_pickle=True).item()			#
		lines_SSW=list(Source_SSW.keys())[6:]
		lines_SLW=list(Source_SLW.keys())[6:]
		WCS_SSW=Source_SSW['WCS']
		WCS_SLW=Source_SLW['WCS']

		plt.figure()
		a=1
		for line in lines_SSW:			#More absorption lines
			if line=='NII_1461' or line=='CO_109' or line=='CO_1110':
				type_line='Emission'
			else:
				type_line='Absorption'

			size=np.shape(Source_SSW[line])
			ax=plt.subplot(2,3,a,projection=WCS_SSW)
			im=plt.imshow(Source_SSW[line])
			ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
				#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
			ax.set_xlim(-0.5,np.shape(Source_SSW[line])[1]-0.5)
			ax.set_ylim(-0.5,np.shape(Source_SSW[line])[0]-0.5)
				#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
				#ax.set_xlim(0,np.shape(Source_SSW[line])[0]-1)
				#ax.set_ylim(0,np.shape(Source_SSW[line])[1]-1)
			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}$)')# fontsize=10)
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
			size=np.shape(Source_SLW[line])
			ax=plt.subplot(2,3,a,projection=WCS_SLW)
			im=plt.imshow(Source_SLW[line])
			ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
				#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
			ax.set_xlim(-0.5,np.shape(Source_SLW[line])[1]-0.5)
			ax.set_ylim(-0.5,np.shape(Source_SLW[line])[0]-0.5)#-1)
				#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
				#ax.set_xlim(0,np.shape(Source_SLW[line])[0]-1)
				#ax.set_ylim(0,np.shape(Source_SLW[line])[1]-1)
			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}$)')
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
			im=plt.imshow(Source_SLW[line])
			ax.contour(hdu_MIPS24[0].data, transform=ax.get_transform(WCS(hdu_MIPS24[0].header)),levels=Levels, colors='white',linewidths=0.5)
				#ax2.contour(hdu_MIPS24[0].data, transform=ax2.get_transform(wcs_MIPS), colors='white',linewidths=0.5)
			ax.set_xlim(-0.5,np.shape(Source_SLW[line])[1]-0.5)
			ax.set_ylim(-0.5,np.shape(Source_SLW[line])[0]-0.5)#-1)
				#ax.contour(New_MIPS,colors="white",linewidths=0.5,alpha=0.8)#linestyles='dashed')
				#ax.set_xlim(0,np.shape(Source_SLW[line])[0]-1)
				#ax.set_ylim(0,np.shape(Source_SLW[line])[1]-1)
			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}$)')
				#plt.colorbar()
			a+=1
				#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in SPIRE-MAP SLW-Band. Image size: '+str(size),fontsize=11,fontstyle='italic')
		plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.5)
		plt.show()

		#bounds=np.arange(np.nanmin(Source_R[line]),0.9*np.nanmax(Source_R[line]),1)
		#norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
		#pcm = ax.pcolormesh(Source_R[line], norm=norm)#, cmap='RdBu_r')

	if int(my_dict[str(source_name)]['SPIRE_SPARSE'])==1:
		Detectors=np.load(direc1+'SPARSE_detectors.npy',allow_pickle=True).item()
		Beam_SLW, Beam_SSW, FOV, detector_SLW, detector_SSW = info_detectors(source_name)

		Source_SSW=np.load(direc1+'MAPS3/'+str(source_name)+'_SSW.npy',allow_pickle=True).item()
		Source_SLW=np.load(direc1+'MAPS3/'+str(source_name)+'_SLW.npy',allow_pickle=True).item()			#
		lines_SSW=list(Source_SSW.keys())[6:]
		lines_SLW=list(Source_SLW.keys())[6:]

		Flux_SSW ,Wave_SSW ,fit_line_SSW, Residual_SSW, label_SSW, title, title_spec_SSW, fitted_lines =fit_one_spectrum_spire_sparse(source_name,'SSW', Det_SSW)#,Detector_SLW=None)
		Flux_SLW ,Wave_SLW , fit_line_SLW, Residual_SLW, label_SLW, title, title_spec_SLW, fitted_lines =fit_one_spectrum_spire_sparse(source_name,'SLW',Det_SLW)
		print(title_spec_SSW)

		y_line_SSW,y_line_SLW= fit_line_SSW(Wave_SSW),fit_line_SLW(Wave_SLW)

		central_detector_SLW='SLWC3'
		central_detector_SSW='SSWD4'

		if Det_SLW is None and Det_SSW is None:
			print(central_detector_SSW)
			Detector_SSW=central_detector_SSW
			Detector_SLW=central_detector_SLW
		else:
			Detector_SSW=Det_SSW
			Detector_SLW=Det_SLW
		#else:
		facecolor_SLW,facecolor_SSW='mediumaquamarine','tomato'

		gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
		ax1 = plt.subplot(gs[0:, 0], projection=WCS(hdu_MIPS24[0].header))
		ax2 = plt.subplot(gs[0, 1])
		ax3 = plt.subplot(gs[1, 1])

		#plt.figure()
		#ax=plt.subplot(2,2,1,projection=WCS(hdu_MIPS24[0].header))
		ax1.imshow(hdu_MIPS24[0].data, vmin=np.nanmin(hdu_MIPS24[0].data[100:150,100:150]) , vmax=np.nanmax(hdu_MIPS24[0].data[100:150,100:150]))#[100:150,100:150])
		ax1.set_xlabel('RA (J2000)')
		ax1.set_ylabel('DEC (J2000)')
		ax1.set_title('Source '+str(source_name)+r' observed with Spitzer MIPS 24 $\mu$m', fontstyle='italic',fontsize=11)

		r = SphericalCircle((Detectors[source_name][Detector_SLW][0]* u.deg, Detectors[source_name][Detector_SLW][1] * u.deg), Beam_SLW * u.degree, edgecolor='mediumaquamarine',label='SLW',facecolor=facecolor_SLW, alpha=0.7,linewidth=1.5,transform=ax1.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
		plt.text(Detectors[source_name][Detector_SLW][0]-0.007, Detectors[source_name][Detector_SLW][1]-0.001, str(Detector_SLW[3:]),color='mediumaquamarine',alpha=0.8,transform=ax1.get_transform('fk5'))
		ax1.add_patch(r)

		r = SphericalCircle((Detectors[source_name][Detector_SSW][0]* u.deg, Detectors[source_name][Detector_SSW][1] * u.deg), Beam_SSW * u.degree, edgecolor='tomato',label='SSW',facecolor=facecolor_SSW, alpha=0.7,linewidth=1.5,transform=ax1.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
		plt.text(Detectors[source_name][Detector_SSW][0], Detectors[source_name][Detector_SSW][1], str(Detector_SSW[3:]),color='yellow',alpha=0.8,transform=ax1.get_transform('fk5'))
		ax1.add_patch(r)

		for detector in detector_SLW:
			if detector==str(Detector_SLW):
				continue
			#print(detector)
			r = SphericalCircle((Detectors[source_name][detector][0]* u.deg, Detectors[source_name][detector][1] * u.deg), Beam_SLW * u.degree, edgecolor='mediumaquamarine',facecolor='none', linewidth=1.5,transform=ax1.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
			ax1.add_patch(r)
			#plt.text(Detectors[source_name][detector][0]+0.001, Detectors[source_name][detector][1]-0.001, str(detector[3:]), color='yellow',transform=ax.get_transform('fk5'))
			plt.text(Detectors[source_name][detector][0], Detectors[source_name][detector][1], str(detector[3:]), color='mediumaquamarine',horizontalalignment='center',verticalalignment='center',transform=ax1.get_transform('fk5'))
		#plt.legend('SLW')

		for detector in detector_SSW:
			if detector==str(Detector_SSW):
				continue
			r = SphericalCircle((Detectors[source_name][detector][0]* u.deg, Detectors[source_name][detector][1] * u.deg), Beam_SSW * u.degree, edgecolor='tomato',facecolor='none',linewidth=1.5,transform=ax1.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
			ax1.add_patch(r)
			plt.text(Detectors[source_name][detector][0], Detectors[source_name][detector][1], str(detector[3:]), color='tomato',horizontalalignment='center',verticalalignment='center',transform=ax1.get_transform('fk5'))
		
		r = SphericalCircle((Detectors[source_name][central_detector_SLW][0]* u.deg, Detectors[source_name][central_detector_SLW][1] * u.deg), FOV * u.degree, label='FOV',edgecolor='seagreen',linewidth=2,facecolor='none',transform=ax1.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
		ax1.add_patch(r)
		ax1.legend()
		#plt.colorbar(label="Flux (MJy/sr)")
		

		#plt.subplot(2,2,2)
		ax2.plot(Wave_SSW, Flux_SSW, '-b')
		ax2.plot(Wave_SLW, Flux_SLW, '-b', label='data')
		ax2.plot(Wave_SSW,y_line_SSW,'-',color='tomato',label=str(Detector_SSW)+' '+ str(label_SSW[:]))
		ax2.plot(Wave_SLW,y_line_SLW,'-',color='mediumaquamarine',label=str(Detector_SLW)+' '+ str(label_SLW[:]))
		ax2.legend()
		ax2.set_title(r'Lines fit SPIRE_MAP '+str(title), fontsize=10)
		ax2.tick_params(axis='both', which='major', labelsize=8)
		ax2.set_xlabel(r'Frequency (GHz)', fontsize=8)
		ax2.set_ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)', fontsize=8)

		'''
		#plt.subplot(2,2,3)
		ax3.plot(Wave_SSW,Residual_SSW,'-',color='orangered', label='Data')
		plt.plot(Wave_SLW, Residual_SLW, '-',color='orangered')
		plt.xlabel(r'Frequency (GHz)')
		plt.ylabel(r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
		plt.legend()
		plt.title(r'Residu: Données-Fit des lines'+str(title))'''

		#plt.subplot(2,2,4)
		detectors=[]
		for i in range (len(Source_SSW['Detectors'])):
			detectors.append(Source_SSW['Detectors'][i][3:])
		
		Detectors_FOV=['B2','B3','B4','C2','C3','C4','C5','D2','D3','D4','D6','F2','F3','E2','E3','E4','E5']#,'SLWB2','SLWB5','SLWC2','SLWC3','SLWC4','SLWD2','SLWD3']

		Flux_Lines=Source_SSW['NII_1461']
		Max_Flux=np.nanmax(Flux_Lines)
		det_max=detectors[Flux_Lines.index(Max_Flux)]
		central_detector_SSW=Source_SSW['Detectors'][Source_SSW['Detectors'].index('SSWD4')][3:]
		coord_central_det=Source_SSW['Detectors'].index('SSWD4')
		coord_max=Flux_Lines.index(Max_Flux)
		coord=np.arange(len(Flux_Lines))

		ax3.bar(coord,Flux_Lines)
		index=[]
		for det in Detectors_FOV:
			index.append(detectors.index(det))

		for i in index:
			ax3.bar(coord[i],Flux_Lines[i],color='seagreen')#, label='above FOV')

		if central_detector_SSW==det_max:
			ax3.bar(coord_max,Max_Flux,color='red', label='SSW central detector and max flux')
		else:
			ax3.bar(coord_central_det,Flux_Lines[coord_central_det],color='pink',label='SSW central detector')
			ax3.bar(coord_max,Max_Flux,color='red', label='max flux')
		#ax3.bar()
		plt.xticks(coord, detectors, rotation=90, fontsize=8)
		ax3.set_ylim(0,Max_Flux*1.1)
		ax3.tick_params(axis='both', which='major', labelsize=8)
		ax3.set_xlabel('SSW Detectors', fontsize=8)
		ax3.set_ylabel(r'Flux ($W.m^{-2}.sr^{-1}$)', fontsize=8)
		ax3.set_title('Flux for the emission line NII at 1461 GHz for each SSW Detector ', fontsize=10)
		ax3.legend(fontsize=8)

		plt.suptitle(r'Footprint on MIPS 24 $\mu$m and single spectrum for 2 given detectors: SSW and SLW for '+ str(source_name), fontsize='12')

		#plt.show()

		plt.figure()
		a=1
		for line in lines_SLW[:6]:				#More emission lines
			if line=='OH_909' or line=='OH_971' or line=='CH_835':
				type_line='Absorption'
			else:
				type_line='Emission'

			Integ_flux=Source_SLW[line]
			max_flux=np.max(Integ_flux)
			ax=plt.subplot(2,3,a,projection=WCS(hdu_MIPS24[0].header))
			ax.imshow(hdu_MIPS24[0].data, vmin=np.nanmin(hdu_MIPS24[0].data[100:150,100:150]) , vmax=np.nanmax(hdu_MIPS24[0].data[100:150,100:150]))#[100:150,100:150])
			ax.set_xlabel('RA (J2000)')
			ax.set_ylabel('DEC (J2000)')

			#for i in range (len(Source_SLW['Detectors'])):
			for Detect in Source_SLW['Detectors']:
				i=Source_SLW['Detectors'].index(Detect)
				alpha=abs(Integ_flux[i]/max_flux)
				#Detect=str(Source_SLW['Detectors'][i])
				RA,DEC=Detectors[source_name][str(Detect)]
				#DEC=Detectors[source_name][str(Detect)][1] * u.deg
				r = SphericalCircle((RA* u.deg, DEC *u.deg), Beam_SLW * u.degree, edgecolor='mediumaquamarine',label='SLW',facecolor='mediumaquamarine',alpha=alpha,linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
				#plt.text(Detectors[source_name][Detect][0], Detectors[source_name][Detect][1], str(Detect[3:]),color='white',alpha=1,transform=ax.get_transform('fk5'))
				ax.add_patch(r)


			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			#plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
				#plt.colorbar()
			a+=1
				#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in SPIRE-SPARSE SLW-Band. ',fontsize=11,fontstyle='italic')
		#plt.tight_layout()
		plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=2.5)

		plt.figure()
		a=1
		for line in lines_SLW[6:]:				#More emission lines
			if line=='OH_909' or line=='OH_971' or line=='CH_835':
				type_line='Absorption'
			else:
				type_line='Emission'

			Integ_flux=Source_SLW[line]
			max_flux=np.max(Integ_flux)
			ax=plt.subplot(2,3,a,projection=WCS(hdu_MIPS24[0].header))
			ax.imshow(hdu_MIPS24[0].data, vmin=np.nanmin(hdu_MIPS24[0].data[100:150,100:150]) , vmax=np.nanmax(hdu_MIPS24[0].data[100:150,100:150]))#[100:150,100:150])
			ax.set_xlabel('RA (J2000)')
			ax.set_ylabel('DEC (J2000)')

			#for i in range (len(Source_SLW['Detectors'])):
			for Detect in Source_SLW['Detectors']:
				i=Source_SLW['Detectors'].index(Detect)
				alpha=abs(Integ_flux[i]/max_flux)
				#Detect=str(Source_SLW['Detectors'][i])
				RA,DEC=Detectors[source_name][str(Detect)]
				#DEC=Detectors[source_name][str(Detect)][1] * u.deg
				r = SphericalCircle((RA* u.deg, DEC *u.deg), Beam_SLW * u.degree, edgecolor='mediumaquamarine',label='SLW',facecolor='mediumaquamarine',alpha=alpha,linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
				#plt.text(Detectors[source_name][Detect][0], Detectors[source_name][Detect][1], str(Detect[3:]),color='white',alpha=1,transform=ax.get_transform('fk5'))
				ax.add_patch(r)


			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=9)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=9)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			ax.tick_params(axis='both', which='major', labelsize=8)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			#plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
				#plt.colorbar()
			a+=1
				#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in SPIRE-SPARSE SLW-Band. ',fontsize=11,fontstyle='italic')
		#plt.tight_layout()
		plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=2.5)
		
		plt.figure()
		a=1
		for line in lines_SSW:				#More emission lines
			if line=='NII_1461' or line=='CO_109' or line=='CO_1110' or line=='CO_98':
				type_line='Emission'
			else:
				type_line='Absorption'

			Integ_flux=Source_SSW[line]
			max_flux=np.max(Integ_flux)
			ax=plt.subplot(2,3,a,projection=WCS(hdu_MIPS24[0].header))
			ax.imshow(hdu_MIPS24[0].data, vmin=np.nanmin(hdu_MIPS24[0].data[100:150,100:150]) , vmax=np.nanmax(hdu_MIPS24[0].data[100:150,100:150]))#[100:150,100:150])
			ax.set_xlabel('RA (J2000)')
			ax.set_ylabel('DEC (J2000)')

			#for i in range (len(Source_SLW['Detectors'])):
			for Detect in Source_SSW['Detectors']:
				i=Source_SSW['Detectors'].index(Detect)
				alpha=abs(Integ_flux[i]/max_flux)
				#Detect=str(Source_SLW['Detectors'][i])
				RA,DEC=Detectors[source_name][str(Detect)]
				#DEC=Detectors[source_name][str(Detect)][1] * u.deg
				r = SphericalCircle((RA* u.deg, DEC *u.deg), Beam_SSW * u.degree, edgecolor='tomato',alpha=alpha,label='SLW',facecolor='tomato',linewidth=1.5,transform=ax.get_transform('fk5'))#transform=ax.get_transform(WCS(hdu_MIPS24[0].header)))
				#plt.text(Detectors[source_name][Detect][0], Detectors[source_name][Detect][1], str(Detect[3:]),color='white',alpha=1,fontsize=8,transform=ax.get_transform('fk5'))
				ax.add_patch(r)


			ax.set_xlabel('RA (J2000)',labelpad=0.5,fontsize=8)
			ax.set_ylabel('DEC (J2000)',labelpad=0.,fontsize=8)
			ax.coords[0].set_ticks(exclude_overlapping=True)
			ax.tick_params(axis='both', which='major', labelsize=8)
			plt.title(str(type_line)+' line '+str(line),fontsize=10)
			#plt.colorbar(im, ax=ax, label=r'Flux ($W.m^{-2}.sr^{-1}.Hz^{-1}$)')
				#plt.colorbar()
			a+=1
				#plt.tight_layout()
		plt.suptitle('Line Flux maps for '+str(source_name)+' in SPIRE-SPARSE SSW-Band. Image size: '+str(size),fontsize=11,fontstyle='italic')
		#plt.tight_layout()
		plt.tight_layout(pad=3.0, w_pad=0.8, h_pad=2.5)



		plt.show()

	return #Source_R, Source_B

############################################################  CO-LINES Rotation Diagram   ########################################################
#################################################################################################################################################

def temperature(source_name,instru):
	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()
	my_CO=csv_to_dict('CO_transitions.csv')					#There is a dictionnary with all the parameters
	transitions=my_CO.keys()

	Detectors=np.load(direc1+'SPARSE_detectors.npy',allow_pickle=True).item()

	CO_SLW=['CO_43','CO_54','CO_65','CO_76','CO_87']
	CO_SSW=['CO_98','CO_109','CO_1110']

	Source_Sparse=[]
	Source_Map=[]
	for source in sources:
		if int(my_dict[str(source)]['SPIRE_MAP'])==1:
			Source_Map.append(source)
		if int(my_dict[str(source)]['SPIRE_SPARSE'])==1:
			Source_Sparse.append(source)

	k=1.38064852E-23								#Boltzmann constant
	h=6.62607015E-34								#Planck constant 
	c=3E8

	Gamma=8E-4*np.pi/h

	#Initialisation of the fitting model
	fit=models.Linear1D(slope=-1,intercept=1)
	fitter=fitting.LinearLSQFitter()

	if instru=='SPIRE_MAP':
		a=1
		#for source in Source_Map:
		type_source=my_dict[str(source_name)]['Morphologie']
		Source_SLW=np.load(direc1+'MAPS3/'+str(source_name)+'_SLW.npy',allow_pickle=True).item()		#Only SLW CO for Sparse Map
		lines=list(Source_SLW.keys())[6:]
		CO_lines=[]
		for line in lines:
			if line in CO_SLW:
				CO_lines.append(line)
		co_slw=np.zeros((np.shape(Source_SLW[CO_lines[0]])[0],np.shape(Source_SLW[CO_lines[0]])[0]))#len(lines)))
		co_slw_error=np.zeros((np.shape(Source_SLW[CO_lines[0]])[0],np.shape(Source_SLW[CO_lines[0]])[0]))#len(lines)))
		energy=np.zeros((np.shape(Source_SLW[CO_lines[0]])[0],np.shape(Source_SLW[CO_lines[0]])[0]))#len(lines)))
		Nu_lines=[]
		energy_K=[]
		for i in range(np.shape(Source_SLW[lines[0]])[0]):
			for j in range(np.shape(Source_SLW[lines[0]])[1]):
				for line in CO_lines:
					if Source_SLW[line][i,j]!=0:
						Nu_lines.append(np.log(Source_SLW[line][i,j]*Gamma/(float(my_CO[str(line)]['Freq'])*1E9*(2*float(my_CO[str(line)]['Ju'])+1)*float(my_CO[str(line)]['Aul']))))
						#energy_K.append(float(my_CO[str(line)]['Eu']))


					else:
						Nu_lines.append(0)
					
					energy_K.append(float(my_CO[str(line)]['Eu']))


				#print(Nu_lines)
				#print(energy_K)
				#if Nu_lines!='-inf':
				fit_CO=fitter(fit,energy_K,Nu_lines)
				y_fit=fit_CO(energy_K)
				#if fit_CO.parameters[0]!=0 :#or fit_CO.parameters[0]!=-inf:
				try:
					Temp=int(-1/fit_CO.parameters[0])
				except:
					Temp=0
				
				co_slw[i,j]=Temp

		print(co_slw)


		plt.subplot(1,2,1)
		plt.imshow(co_slw,origin='lower')
		plt.colorbar()

		plt.subplot(1,2,2)
		plt.imshow(Source_SLW['CO_43'],origin='lower')

		plt.show()

		return co_slw

def plot_co_rot_diagram():

	my_dict=csv_to_dict('Bulles.csv')
	sources=my_dict.keys()
	my_CO=csv_to_dict('CO_transitions.csv')					#There is a dictionnary with all the parameters
	transitions=my_CO.keys()

	Detectors=np.load(direc1+'SPARSE_detectors.npy',allow_pickle=True).item()
	
	CO_SLW=['CO_43','CO_54','CO_65','CO_76','CO_87']
	CO_SSW=['CO_98','CO_109','CO_1110']

	Source_Sparse=[]
	Source_Map=[]
	for source in sources:
		if int(my_dict[str(source)]['SPIRE_MAP'])==1:
			Source_Map.append(source)
		if int(my_dict[str(source)]['SPIRE_SPARSE'])==1:
			Source_Sparse.append(source)

	#Needed values for the calculations
	k=1.38064852E-23								#Boltzmann constant
	h=6.62607015E-34								#Planck constant 
	c=3E8
	#Gamma=8*np.pi*k/(h*c*c*c)
	Gamma=8E-4*np.pi/h
	#Beta=4*np.pi/(h*c)
	#Beta=8*np.pi/h
	#print(Gamma)
	fit=models.Linear1D(slope=-1,intercept=1)
	fitter=fitting.LinearLSQFitter()

	plt.figure()
	a=1
	for source in Source_Map:
		type_source=my_dict[str(source)]['Morphologie']
		Source_SLW=np.load(direc1+'MAPS/'+str(source)+'_SLW.npy',allow_pickle=True).item() #Only SLW CO for Sparse Map
		co_slw=[]					#We'll create a list with a mean value of each CO line 
		co_slw_error=[]
		energy=[]
		for line in CO_SLW:
			if line in list(Source_SLW.keys())[5:]:
				#index=CO_SLW.index(line)
				#Aul: Einstein coefficients, Ju: Quantum number, Eu: Energy of the upper level (Kelvin)
				co_slw.append(np.log(Source_SLW[line][1][4,3]*Gamma/(float(my_CO[str(line)]['Freq'])*(2*float(my_CO[str(line)]['Ju'])+1)*float(my_CO[str(line)]['Aul']))))		
				co_slw_error.append(np.nanstd(Source_SLW[line][1])/np.nanmean(Source_SLW[line][1]))
				#energy.append(Energy_level[index])
				energy.append(float(my_CO[str(line)]['Eu']))
				#print(
				Flux=(Source_SLW[line][1][4,3]*u.W/u.m/u.m/u.Hz/u.sr).to(u.MJy/u.sr)
				print(Flux)

		#print(co_slw)

		if len(co_slw)>2:
			#a=1
			fit_CO=fitter(fit,energy,co_slw)
			y_fit=fit_CO(energy)
			Temp=int(-1/fit_CO.parameters[0])
			#plt.figure()
			plt.subplot(2,3,a)
			plt.errorbar(energy,co_slw,co_slw_error,color='cyan',markeredgecolor='blue',fmt='p',label='Data')
			#plt.plot(energy,co_slw, 'p', markerfacecolor='cyan', markeredgecolor='blue',label='Data')#label='Slope=%5.3f, '%tuple(popt))
			plt.plot(energy,y_fit, '--g', label='Temp= '+str(Temp)+' K')
			plt.xlabel(r'$E_{u}/k_{b}$ (K)', fontsize=8)
			plt.ylabel(r'$ln(\frac{\beta I_{\nu}}{A_{ul} \nu_{ul} g_{ul}})$ ($cm^{-2}$)', fontsize=8)
			plt.title(str(source)+' ('+str(type_source)+' Morphology) ', fontsize=9)
			plt.legend(fontsize=8)
			a+=1
	plt.suptitle('CO rotation diagram from SPIRE MAP data ', fontsize=12, fontstyle='italic')
	plt.tight_layout()

	'''
	plt.figure()
	a=1
	for source in Source_Sparse:
		type_source=my_dict[str(source)]['Morphologie']
		Source_SSW=np.load(direc1+'MAPS/'+str(source)+'_SSW.npy',allow_pickle=True).item()
		Source_SLW=np.load(direc1+'MAPS/'+str(source)+'_SLW.npy',allow_pickle=True).item()	
		co_slw=[]					#We'll create a list with a mean value of each CO line 
		co_ssw=[]
		co_slw_error=[]
		co_ssw_error=[]
		energy_ssw,energy_slw=[],[]
		for line in CO_SLW:
			if line in list(Source_SLW.keys())[5:]:
				index=list(Source_SLW['Detectors']).index(central_detector_SLW)
				#co_slw.append(np.log(np.nanmean(Source_SLW[line][1])*Beta/(Freq[index]*(2*J[index]+1)*Aul[index])))	#[1] IS THE 2D image with a value for each pixel
				co_slw.append(np.log(Source_SLW[line][1][index]*Gamma/(float(my_CO[str(line)]['Freq'])*(2*float(my_CO[str(line)]['Ju'])+1)*float(my_CO[str(line)]['Aul']))))
				co_slw_error.append(np.nanstd(Source_SLW[line][1])/np.nanmean(Source_SLW[line][1]))
				#energy_slw.append(Energy_level[index])
				energy_slw.append(float(float(my_CO[str(line)]['Eu'])))

		for line in CO_SSW:
			if line in list(Source_SSW.keys())[5:]:
				index=list(Source_SSW['Detectors']).index(central_detector_SSW)
				#index=CO_SSW.index(line)+5
				#co_ssw.append(np.log(np.nanmean(Source_SSW[line][1])*Beta/(Freq[index]*(2*J[index]+1)*Aul[index])))	#[1] IS THE 2D image with a value for each pixel
				co_ssw.append(np.log(Source_SSW[line][1][index]*Gamma/(float(my_CO[str(line)]['Freq'])*(2*float(my_CO[str(line)]['Ju'])+1)*float(my_CO[str(line)]['Aul']))))
				co_ssw_error.append(np.nanstd(Source_SSW[line][1])/np.nanmean(Source_SSW[line][1]))
				#energy_ssw.append(Energy_level[index])
				energy_ssw.append(float(my_CO[str(line)]['Eu']))

		CO=co_slw+co_ssw
		Energy=energy_slw+energy_ssw
		error=co_slw_error+co_ssw_error

		#print(CO)
		#print(Energy)

		if len(CO)>2:
			#a=1
			fit_CO=fitter(fit,Energy,CO)
			y_fit=fit_CO(Energy)
			Temp=int(-1/fit_CO.parameters[0])
			plt.subplot(2,3,a)
			plt.errorbar(Energy,CO,error,color='cyan',markeredgecolor='blue',fmt='p',label='Data')
			#plt.plot(Energy,CO, 'p', markerfacecolor='cyan', markeredgecolor='blue',label='Data')#label='Slope=%5.3f, '%tuple(popt))
			plt.plot(Energy,y_fit, '--g', label='Temp= '+str(Temp)+' K')
			plt.xlabel(r'$E_{u}/k_{b}$ (K)', fontsize=8)
			plt.ylabel(r'$ln(\frac{\beta I_{\nu}}{A_{ul} \nu_{ul} g_{ul}})$ ($cm^{-2}$)', fontsize=8)
			plt.title(str(source)+' ('+str(type_source)+' Morphology) ', fontsize=9)
			plt.legend(fontsize=8)
			a+=1
	plt.suptitle('CO rotation diagram from SPIRE SPARSE data ', fontsize=12, fontstyle='italic')
	plt.tight_layout()'''

	plt.show()

	return #fit_CO, CO

############################################################  Continuum - PACS and SPIRE   ########################################################
#################################################################################################################################################

def dust(wvl,amplitude=100,beta=2):
	#freq=freq*10**9
	#freq0=100e9
  	#beta=1.8
  	T=18
  	k=1.38e-23
  	c=3E8
  	h=6.63e-34
  	return amplitude*(wvl*1E-6)**(-beta-2)/(np.exp(h*c/(wvl*1E-6*k*T))-1)

def spectrum_pacs_spire(source_name):

	my_dict=csv_to_dict('Bulles.csv')
	Detectors=np.load(direc1+'SPARSE_detectors.npy',allow_pickle=True).item()

	'''
	#PACS
	hdulist_R=open_source(source_name,'PACS','R')
	hdulist_B=open_source(source_name,'PACS','B')

	wcs_R=WCS(hdulist_R[1].header,naxis=2)
	wcs_B=WCS(hdulist_B[1].header,naxis=2)

	wvl_R=np.arange((hdulist_R[1].header['NAXIS3']))	# En microns
	Wave_R=hdulist_R[1].header['CRVAL3']+hdulist_R[1].header['CDELT3']*wvl_R	

	wvl_B=np.arange((hdulist_B[1].header['NAXIS3']))	# En microns
	Wave_B=hdulist_B[1].header['CRVAL3']+hdulist_B[1].header['CDELT3']*wvl_B

	Pixel_size_deg_R=abs(hdulist_R[1].header['CDELT1']*hdulist_R[1].header['CDELT2'])
	beam=Pixel_size_deg_R*u.deg*u.deg
	Flux_red=(hdulist_R[1].data*u.Jy/beam).to(u.MJy/u.sr)

	Pixel_size_deg_B=abs(hdulist_B[1].header['CDELT1']*hdulist_B[1].header['CDELT2'])
	beam=Pixel_size_deg_B*u.deg*u.deg
	Flux_blue=(hdulist_B[1].data*u.Jy/beam).to(u.MJy/u.sr)'''

	
	#PACS
	Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R, Flux_red= flux_PACS(source_name,'R')
	Flux_image_B, Flux_spectrum_B, Wave_B, wcs_B, Flux_blue= flux_PACS(source_name,'B')

	#SPIRE_MAP
	if int(my_dict[str(source_name)]['SPIRE_MAP'])==1:
		Flux_mean_HR_SSW_image,Flux_mean_HR_SSW_spectrum, Wave_SSW, wcs_HR_SSW, Flux_SSW=plot_image_flux_SPIRE_Map(source_name,'HR','SSW')
		Flux_mean_HR_SLW_image,Flux_mean_HR_SLW_spectre, Wave_SLW, wcs_HR_SLW, Flux_SLW=plot_image_flux_SPIRE_Map(source_name,'HR','SLW')

		#Choose PACS pixel
		pix_x_R,pix_y_R=int(np.shape(Flux_image_R)[0]/2),int(np.shape(Flux_image_R)[1]/2)
		pix_x_B,pix_y_B=int(np.shape(Flux_image_B)[0]/2),int(np.shape(Flux_image_B)[1]/2)

		#Find its coordinates
		pos_x_R,pos_y_R=wcs_R.all_pix2world(pix_x_R,pix_y_R,1)
		pos_x_B,pos_y_B=wcs_B.all_pix2world(pix_x_B,pix_y_B,1)

		#Find SPIRE pixel with thoose coordinates
		pix_x_pixel_SSW,pix_y_pixel_SSW=wcs_HR_SSW.all_world2pix(pos_x_R,pos_y_R,1)
		pix_x_pixel_SLW,pix_y_pixel_SLW=wcs_HR_SLW.all_world2pix(pos_x_B,pos_y_B,1)

		#Take the flux of this given pixel
		Flux_SSW=Flux_SSW[:,pix_x_pixel_SSW,pix_y_pixel_SSW]
		Flux_SLW=Flux_SLW[:,pix_x_pixel_SLW,pix_y_pixel_SLW]

		'''
		pix_x_pixel_SSW,pix_y_pixel_SSW=int(np.shape(Flux_mean_HR_SSW_image)[0]/2),int(np.shape(Flux_mean_HR_SSW_image)[1]/2)
		pix_x_pixel_SLW,pix_y_pixel_SLW=int(np.shape(Flux_mean_HR_SLW_image)[0]/2),int(np.shape(Flux_mean_HR_SLW_image)[1]/2)

		pos_x_ra_SSW,pos_y_dec_SSW=wcs_HR_SSW.all_pix2world(pix_x_pixel_SSW,pix_y_pixel_SSW,1)
		pos_x_ra_SLW,pos_y_dec_SLW=wcs_HR_SLW.all_pix2world(pix_x_pixel_SLW,pix_y_pixel_SLW,1)

		#Equivalent PACS pixel:
		pix_x_R,pix_y_R=wcs_R.all_world2pix(pos_x_ra_SSW,pos_y_dec_SSW,1)
		pix_x_B,pix_y_B=wcs_B.all_world2pix(pos_x_ra_SLW,pos_y_dec_SLW,1)

		Flux_SSW=Flux_SSW[:,pix_x_pixel_SSW,pix_y_pixel_SSW]
		Flux_SLW=Flux_SLW[:,pix_x_pixel_SLW,pix_y_pixel_SLW]'''

		print(pix_x_pixel_SSW,pix_y_pixel_SSW)
		print(pix_x_pixel_SLW,pix_y_pixel_SLW)

	if int(my_dict[str(source_name)]['SPIRE_SPARSE'])==1:


		hdulist_SSW, hdulist_SLW= open_source(source_name,'Sparse','SSW'),open_source(source_name,'Sparse','SLW')
		
		#Flux and Wave for the central detector
		Flux_SSW,Wave_SSW= hdulist_SSW[str(central_detector_SSW)].data['flux'],hdulist_SSW[str(central_detector_SSW)].data['wave']
		Flux_SLW,Wave_SLW= hdulist_SLW[str(central_detector_SLW)].data['flux'],hdulist_SLW[str(central_detector_SLW)].data['wave']

		#RA/DEC positions
		pos_x_ra_SSW,pos_y_dec_SSW= Detectors[str(source_name)][str(central_detector_SSW)]
		pos_x_ra_SLW,pos_y_dec_SLW= Detectors[str(source_name)][str(central_detector_SLW)]
		#Equivalent PACS pixel:
		pix_x_R,pix_y_R=wcs_R.all_world2pix(pos_x_ra_SSW,pos_y_dec_SSW,1)
		pix_x_B,pix_y_B=wcs_B.all_world2pix(pos_x_ra_SLW,pos_y_dec_SLW,1)

		print(int(pix_x_R),int(pix_y_R))

	'''
	#FIT CONTINUUM
	Source_R=np.load(direc1+'MAPS/'+str(source_name)+'_R.npy',allow_pickle=True).item()
	Source_B=np.load(direc1+'MAPS/'+str(source_name)+'_B.npy',allow_pickle=True).item()

	param_fit_R=Source_R['Fit'][pix_x_R,pix_y_R,:]
	param_fit_B=Source_R['Fit'][pix_x_B,pix_y_B,:]

	Residual_R=Source_R['Residual']
	Residual_B=Source_B['Residual']

	Continuum_R=Residual_R+'''


	#Unit conversion PACS: Jy/sr and microns and SPIRE: W.m-2.sr-1.Hz-1 and GHz
	#Flux_SSW=(Flux_SSW*u.W/u.m/u.m/u.Hz/u.sr).to(u.MJy/u.sr)#.values
	#Flux_SLW=(Flux_SLW*u.W/u.m/u.m/u.Hz/u.sr).to(u.MJy/u.sr)#.values
	Flux_SSW=Flux_SSW*(Wave_SSW*1E9)**2/3E14
	Flux_SLW=Flux_SLW*(Wave_SLW*1E9)**2/3E14
	Wave_SSW=(Wave_SSW*u.GHz).to(u.um,equivalencies=u.spectral())#.values
	Wave_SLW=(Wave_SLW*u.GHz).to(u.um,equivalencies=u.spectral())#.values


	'''
	Flux_SSW=Flux_SSW*3E14/(Wave_SSW**2)
	Flux_SLW=Flux_SLW*3E14/(Wave_SLW**2)'''

	pos_x_ra_R,pos_y_dec_R=wcs_R.all_pix2world(pix_x_R,pix_y_R,1)
	pos_x_ra_B,pos_y_dec_B=wcs_B.all_pix2world(pix_x_B,pix_y_B,1)

	'''
	#Wave=list(Wave_R)+list(Wave_B)
	Flux_R=Flux_red[:,int(pix_x_R),int(pix_y_R)]#)+list(Flux_blue[:,int(pix_x_B),int(pix_y_B)])
	DustModel=custom_model(dust)
	dust_init=DustModel(amplitude=5E-10,beta=2)
	fitter= fitting.LevMarLSQFitter()
	fit_R=fitter(dust_init,Wave_R,Flux_R)

	fit_tot=dust_init(Wave_R)'''


	print(int(pix_x_R) , int(pix_y_R))

	plt.figure()
	plt.plot(Wave_R,Flux_red[:,int(pix_x_R),int(pix_y_R)])#label='RA = ' +str(pos_x_ra_R)+ ' DEC= ' + str(pos_y_dec_R))
	plt.plot(Wave_B,Flux_blue[:,int(pix_x_B),int(pix_y_B)],label='RA = ' + str(pos_x_ra_B)+' DEC= ' + str(pos_y_dec_B))
	#plt.plot(Wave_SSW,Flux_SSW,label='RA = ' + str(pos_x_ra_SSW)+' DEC= ' +str(pos_y_dec_SSW))
	#plt.plot(Wave_SLW,Flux_SLW)#label='RA = ' + str(pos_x_ra_SLW)+' DEC= ' +str(pos_y_dec_SLW))
	#plt.plot(Wave_R,fit_tot)

	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($MJy.sr^{-1}$)')
	plt.title('Total spectrum for source '+str(source_name)+' PACS and SPIRE bands')
	plt.legend()
	plt.show()






