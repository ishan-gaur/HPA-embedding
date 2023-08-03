"""
channels
The names of the channels as they will appear in the dataset directory folders ie {channel}.png
This will be saved after you set it for the first time, so you only need to set it once.

cmaps
The colormaps to use for each channel. These are the names of the colormaps in the microfilm package.
Normally pure_{color} works fine.
"""
channels=["nuclei", "microtubule", "Geminin", "CDT1"]
cmaps=["pure_blue", "pure_yellow", "pure_green", "pure_red"]
"""
norm_strategy
min_max: normalize to [0, 1] using image min and max
rescale: rescale to [0, 1] using datatype min and max
threshold: threshold to [0, 1] by clipping everything outside the thresholds and rescaling
    must set norm_min and norm_max (None by default)
percentile: threshold to [0, 1] by clipping everything outside the percentiles and rescaling
"""
# norm_strategy = 'min_max'
# norm_min, norm_max = None, None
# norm_strategy = 'threshold'
# norm_min, norm_max = 500, 65535
norm_strategy = 'percentile'
norm_min, norm_max = 20, 99