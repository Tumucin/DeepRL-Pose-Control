# Install panda_gym package
pip3 install -e .

# Add submodules
git submodule init
git submodule update

# Install PyKDL module
conda install -c conda-forge python-orocos-kdl

# Download the pre-trained models for testing
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1DePp2fjsBwHaF2CqFKZMpiV_z-36OoNz?usp=sharing



