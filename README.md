# HPA-Embedding

Please lmk if y'all know how to get rid of the `please compile abn` warning when using the segmentation model. Think it has to do with the inplace activated batch-norm library, but I'm not sure why it isn't compiling. Do we need to explicitly install nvcc before install HPA-Cell-Segmentation?

Running ``main.py`` requires using the `data-prep` conda environment to use HPA-Cell-Segmentation. This can be set up from the `HPA-Cell-Segmentation` submodule of this repo. The reason for having a separate conda environment is due to the saved segmentation model's use of an old version of pytorch and some deprecated changes it uses from previous versions of numpy.

Once in the `HPA-Cell-Segmentation` directory, run `conda env create -f environment.yml`, `conda activate data-prep`, and then `sh install.sh`. It's important to not try to install each of the dependencies individually as it's very easy for the wrong versions of packages to get installed when added on sequentially. The resulting `data-prep` environment is only required for the mask generation step in main. (`--image_mask_cache`)

This repo aggregates image/mask caches at the level of the paths given in `data-folder.txt` in the working directory from where main is run. If you want to aggregate these caches at a level other that the first level of the data directory file tree you will have to provide a custom `data-folder.txt` there are various methods you can look up online for doing this to tailor your needs. You can maybe find inspiration as well by checking out the source for doing this in `pipeline.py` > `create_image_paths_file`.

`main.py` provides a command line interface for running any of the steps of the pipeline and the necessary parameters for the preprocessing steps can be set in `config.py`. Don't worry about making multiple versions, the source is saved with the dataset using the `--name` parameter in main.

You can do an interactive trial run on a dev dataset to play with these parameters using the `data-process.ipynb` notebook. You can also get more information and do dry runs of the normalization, as an example, using the `--stats` option in the pipeline.

`main.py` primarily calls functions from `pipeline.py` and `stats.py`. These two use `utils.py` and `data_viz.py` for other functions.
