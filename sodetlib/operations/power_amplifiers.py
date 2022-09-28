import numpy as np
import matplotlib.pyplot as plt
import sodetlib as sdl

sdl.set_action()
def fullbandresponse_metric(S, band):
	"""
	LNA check metric using S.full_band_resp

	Args
	----
    S : SmurfControl
        Pysmurf instance
	bands : iterable
		Band to use.

    Returns
    ----------
    metric: flot
    	Median value of the fft returned by full_band_resp
    data: array
		(f, resp) -- Raw data returned by full_band_resp
	"""
	f, resp = S.full_band_resp(band=b)
	# Or do we need median of some frequency range???
    return np.nanmedian(np.real(resp)), (f,resp) 

sdl.set_action()
def rawadc_metric(S, band):
	"""
	LNA check metric using S.read_adc_data

	Args
	----
    S : SmurfControl
        Pysmurf instance
	bands : iterable
		Band to use.

    Returns
    ----------
    metric: flot
    	Median value of the adc data
    data: array
		Raw data returned by read_adc_data
	"""
	data = S.read_adc_data( band=b)
    return np.nanmedian(data), data

sdl.set_action()
def check_LNAs(S, metric=fullbandresponse_metric, bands=range(8), 
				display=True, threshold=5e-1):
	"""
	Checks whether LNAs controlled by slot S are functinoing, using specified metric.  
	First all LNAs are turned on for a reference measurement, then measures 
	what happens when each stages LNAs are turned off (and then back on).

	Args
	----
    S : SmurfControl
        Pysmurf instance
	metric : function
	 	Function used to measure LNAs functionality.
	 	Must return (scalar metric, raw data).
	bands : iterable
		bands to consider.  Defaults to all.
		There are two LNAs per slot, corresponding to bands 0-3 and 4-7.
	display : bool
		When True, displays results in color-coded table,
		using threshold for coloring
	threshold: float
	 	Maximum ratio of OFF/ON to be considered 'good' for color coding
	 	when display is True. If None, display will use default log colorbar

    Returns
    ----------
    metrics: dict
    	Key is all ON ref or which stage had LNAs OFF, value is list of
    	metrics corresponding to each band in bands.
    arrays: dict
    	Key is all ON ref or which stage had LNAs OFF, value is list of
    	raw data from metric fn corresponding to each band in bands.
    bands: list
    	The bands tested against -- corresponds to the values returned dics
	"""

	results={
		'ref':[],
		'300K':[],
		'40K':[],
		'4K':[]
	}

	# All on [300K, 40K, 4K]
	S.C.write_ps_en(0b1111) # Turn on cryogenic LNAs
	S.C.write_optical(0b11) # Turn on 300K LNAs

	# Get all on reference
	for b in bands:
		results["ref"].append(metric(S,b)) 

	# 300K OFF
	S.C.write_optical(0b00)
	for b in bands:
		results["300K"].append(metric(S,b)) 
	# 300K ON
	S.C.write_optical(0b11)

	# 40K OFF
	S.C.write_ps_en(0b0101)
	for b in bands:
		results["40K"].append(metric(S,b)) 
	# 40K ON
	S.C.write_ps_en(0b1111)

	# 4K OFF
	S.C.write_ps_en(0b1010)
	for b in bands:
		results["4K"].append(metric(S,b)) 
	# 4K ON
	S.C.write_ps_en(0b1111)

	meds = {}
	arrays ={}
	for lna, pairs in results.items():
		meds[lna] = [i[0] for i in pairs]
		arrays[lna] = [i[1] for i in pairs]


	if display:
		# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
		cols = [i for i in meds if i!='ref']
		data = np.array([meds[c] for c in cols]).T
		ref = np.array(meds['ref'])
		data = data / ref[:, np.newaxis]

		fig, ax = plt.subplots()
		if threshold is not None:
			mask = data<threshold
			# make empty array of rgb tuples
			colors = np.zeros(list(data.shape)+[3])
			colors[mask]= (0,255,0)
			colors[~mask]= (255,0,0)
			im = ax.imshow(colors)
		else:
			im = ax.imshow(data, norm='log')

		# Show all ticks and label them with the respective list entries
		ax.set_xticks(np.arange(len(cols)))
		ax.set_xticklabels(cols)
		ax.set_yticks(np.arange(len(bands)))
		ax.set_yticklabels(bands)
		ax.set_xlabel("LNA")
		ax.set_ylabel("Band")
		ax.set_title("LNA Off / All On")

		# Loop over data dimensions and create text annotations.
		for i in range(len(bands)):
		    for j in range(len(cols)):
		        text = ax.text(j, i, f"{data[i, j]:.0e}",
		                       ha="center", va="center", color="w")

	return meds, arrays, list(bands)
