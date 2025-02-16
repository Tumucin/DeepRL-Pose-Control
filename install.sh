# Install panda_gym package
pip3 install -e .

# Add submodules
git submodule init
git submodule update

mkdir failedSamples
# Install PyKDL module
conda install -c conda-forge python-orocos-kdl

cd panda_gym/envs/kdl_parser/kdl_parser_py/
pip3 install .
# Download the pre-trained models for testing
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1DePp2fjsBwHaF2CqFKZMpiV_z-36OoNz?usp=sharing



