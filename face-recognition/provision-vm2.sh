#!/bin/bash

################################################
# Provisions the 2nd of 2 VMs used for the face-recognition mini project.
# This script is based on https://github.com/nirmalyaghosh/deep-learning-vm
################################################

function mssg {
    now=$(date +"%T")
    echo "[$now] $1"
    shift
}

mssg "Provisioning the 2nd of 2 VMs used for the face-recognition mini project ..."

mssg "Updating the package index files. Usually takes ~ 6 minutes, depending on the speed of your network ..."
apt-get -y update >/dev/null 2>&1

################################################
# apt-fast
mssg "Installing apt-fast to try speed things up ..."
apt-get install -y aria2 --no-install-recommends >/dev/null 2>&1
filetowget=apt-fast
if [[ ! -f $filetowget ]]; then
   wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast >/dev/null 2>&1
   cp apt-fast /usr/bin/
   chmod +x /usr/bin/apt-fast
fi
filetowget=apt-fast.conf
if [[ ! -f $filetowget ]]; then
   wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast.conf >/dev/null 2>&1
   cp apt-fast.conf /etc
fi

mssg "Installing pip ..."
apt-fast -y install python-pip >/dev/null 2>&1

################################################
# Miniconda
mssg "Downloading & Installing Miniconda ..."
miniconda=Miniconda3-4.0.5-Linux-x86_64.sh
if [[ ! -f $miniconda ]]; then
    wget --quiet http://repo.continuum.io/miniconda/$miniconda
    chmod +x $miniconda
    ./$miniconda -b -p /home/vagrant/miniconda
    echo 'export PATH="/home/vagrant/miniconda/bin:$PATH"' >> /home/vagrant/.bashrc
    source /home/vagrant/.bashrc
    chown -R vagrant:vagrant /home/vagrant/miniconda
    /home/vagrant/miniconda/bin/conda install conda-build anaconda-client anaconda-build -y -q
fi

################################################
# Essential Python packages : pandas, scikit-learn
mssg "Installing pandas, scikit-learn"
/home/vagrant/miniconda/bin/conda install "pandas==0.18.1" -y -q
/home/vagrant/miniconda/bin/conda install "scikit-learn==0.18" -y -q

################################################
# Theano, H5py, Keras
mssg "Installing Theano dependencies"
apt-fast install -y python3-numpy python3-scipy python3-dev python3-pip python3-nose g++ libopenblas-dev git >/dev/null 2>&1
/home/vagrant/miniconda/bin/pip install nose
/home/vagrant/miniconda/bin/pip install nose_parameterized
mssg "Installing Theano"
/home/vagrant/miniconda/bin/pip install "Theano==0.8.2"

mssg "Installing H5py"
apt-fast install -y libhdf5-dev >/dev/null 2>&1
/home/vagrant/miniconda/bin/pip install "h5py==2.6.0" >/dev/null 2>&1

mssg "Installing Keras"
/home/vagrant/miniconda/bin/pip install "keras==1.1.1" >/dev/null 2>&1

mssg "Installing Tensorflow"
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl
/home/vagrant/miniconda/bin/pip install $TF_BINARY_URL >/dev/null 2>&1

mssg "Installing PIL"
apt-get build-dep python-imaging
apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
/home/vagrant/miniconda/bin/pip install Pillow

mssg "Installing matplotlib"
/home/vagrant/miniconda/bin/conda install matplotlib -y

################################################
mssg "Installing IPython Notebook server"
mkdir -p /home/vagrant/notebooks
chown -R vagrant:vagrant /home/vagrant/notebooks
/home/vagrant/miniconda/bin/pip install ipython[notebook]

echo ""
mssg "List of installed packages"
/home/vagrant/miniconda/bin/pip list

mssg "Set the timezone"
echo 'Asia/Singapore' | sudo tee /etc/timezone
sudo dpkg-reconfigure -f noninteractive tzdata

mssg "Done!"