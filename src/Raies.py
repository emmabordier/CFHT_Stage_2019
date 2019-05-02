def raies_NII(nom_source):	
	Flux_image_R, Flux_spectrum_R, Wave_R, wcs_R= flux_PACS(nom_source,'R')

	cond_NII=np.where((Wave_R>120) & (Wave_R<125))
	Wave_NII=Wave_R[cond_NII]
	Flux_NII=Flux_spectrum_R[cond_NII]

	ampli=np.nanmax(Flux_NII)
	meanNII=Wave_NII[np.where(Flux_NII==np.nanmax(Flux_NII))]


	spectrum = Spectrum1D(flux=Flux_NII*u.Jy, spectral_axis=Wave_NII*u.um)
	#continuum2=models.Linear1D(slope=-0.0012*u.Jy/u.um,intercept=0.195*u.Jy)
	g_init_2 = models.Gaussian1D(amplitude=ampli, mean=meanNII, stddev=0.10) 
	l_init_cont=models.Linear1D(slope=-0.0011,intercept=0.183)
	#p_init = models.Polynomial1D(degree=2)#,c0=0.228, c1=-0.19E-2, c2=3.6E-6)
	g_line_linear=g_init_2+l_init_cont
	#g_line_poly=g_init_2+p_init

	fit_g = fitting.LevMarLSQFitter()
	fit_pol=fitting.LinearLSQFitter()
	g = fit_g(g_init_2, Wave_NII, Flux_NII)
	cont_lin=fit_g(l_init_cont, Wave_R, Flux_spectrum_R)
	#cont_pol=fit_g(p_init,Wave_R,Flux_spectrum_R)
	fit_line_linear=fit_g(g_line_linear,Wave_NII,Flux_NII)
	#fit_line_poly= fit_g(g_line_poly,Wave_NII,Flux_NII)

	#g_fit_2 = fit_lines(spectrum, g_init_2)
	y_fit_2 = g(Wave_NII)
	y_cont_lin = cont_lin(Wave_R)
	#y_cont_pol= cont_pol(Wave_R)
	y_line_linear = fit_line_linear(Wave_NII)
	#y_line_poly = fit_line_poly(Wave_NII)

	plt.plot(Wave_R, Flux_spectrum_R, label='data')
	plt.plot(Wave_R,y_cont_lin, label='Fit cont: Linear 1D')
	#plt.plot(Wave_R,y_cont_pol, label='Fit cont: Poly 1D (deg2)')
	plt.plot(Wave_NII, y_line_linear, label='Fit raie : Gauss+Linear')
	#plt.plot(Wave_NII, y_line_poly, label='Fit raie : Gauss+Poly1D (deg2)')
	plt.legend()
	plt.title(r'Fit de la raie NII à 122 $\mu$m')
	plt.xlabel(r'Wavelength ($\mu$m)')
	plt.ylabel(r'Flux ($Jy.pixel^{-1}$)')
	plt.show()


	return fit_line_linear