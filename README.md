# HPA-Embedding

Please lmk if y'all know how to get rid of the `please compile abn` warning when using the segmentation model. Think it has to do with the inplace activated batch-norm library, but I'm not sure why it isn't compiling. Do we need to explicitly install nvcc before install HPA-Cell-Segmentation?

Running ``insert proper file here`` requires using the `data-prep` conda environment to use HPA-Cell-Segmentation. This can be setup from the `HPA-Cell-Segmentation` submodule of this repo. The reason for having a separate conda environment is due to the saved segmentation model's use of an old version of pytorch and some deprecated changes it uses from previous versions of numpy.

Once in the `HPA-Cell-Segmentation` directory, run `conda env create -f environment.yml`, `conda activate data-prep`, and then `sh install.sh`. It's important to not try to install each of the dependencies individually as it's very easy for the wrong versions of packages to get installed when added on sequentially. The resulting `data-prep` environment is only required for generating the `mask.png` and `com.npy` files for the dataset preprocessing.

``find /data/ishang/CCNB1-dataset/ -type d > data-folder.txt``
And delete the top entry, which will be the parent directory
