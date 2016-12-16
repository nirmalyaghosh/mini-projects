#!/bin/bash

################################################
# Provisions the 1st of 2 VMs used for the face-recognition mini project.
# This script is based on https://github.com/nirmalyaghosh/deep-learning-vm
################################################

function mssg {
    now=$(date +"%T")
    echo "[$now] $1"
    shift
}

mssg "Provisioning the 1st of 2 VMs used for the face-recognition mini project ..."

mssg "Updating the package index files. Usually takes ~ 6 minutes, depending on the speed of your network ..."
apt-get -y update >/dev/null 2>&1

################################################
# apt-fast
mssg "Installing apt-fast to try speed things up ..."
apt-get install -y aria2 --no-install-recommends >/dev/null 2>&1
aptfast=apt-fast
if [[ ! -f $aptfast ]]; then
    wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast >/dev/null 2>&1
    wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast.conf >/dev/null 2>&1
    cp apt-fast /usr/bin/
    chmod +x /usr/bin/apt-fast
    cp apt-fast.conf /etc
fi

################################################
# Miniconda
mssg "Downloading & Installing Miniconda ..."
miniconda=Miniconda2-latest-Linux-x86_64.sh
if [[ ! -f $miniconda ]]; then
    wget --quiet http://repo.continuum.io/miniconda/$miniconda
    chmod +x $miniconda
    ./$miniconda -b -p /home/vagrant/miniconda
    echo 'export PATH="/home/vagrant/miniconda/bin:$PATH"' >> /home/vagrant/.bashrc
    source /home/vagrant/.bashrc
    chown -R vagrant:vagrant /home/vagrant/miniconda
    /home/vagrant/miniconda/bin/conda install conda-build anaconda-client anaconda-build -y -q
fi

# ################################################
# Installing OpenCV
mssg "Installing OpenCV 2.4.10 ..."
/home/vagrant/miniconda/bin/conda install -c menpo opencv=2.4.10 -y

# ################################################
# Installing imageio, matplotlib, numpy, scikit-video, etc.
mssg "Installing imageio ..."
/home/vagrant/miniconda/bin/pip install imageio
mssg "Installing matplotlib ..."
/home/vagrant/miniconda/bin/conda install matplotlib -y
mssg "Installing visvis ..."
/home/vagrant/miniconda/bin/pip install visvis
mssg "Installing scikit-video ..."
/home/vagrant/miniconda/bin/pip install -i https://pypi.anaconda.org/pypi/simple scikit-video
mssg "Installing numpy ..."
/home/vagrant/miniconda/bin/conda install numpy -y -q

################################################
mssg "Installing IPython Notebook server"
mkdir -p /home/vagrant/notebooks
chown -R vagrant:vagrant /home/vagrant/notebooks
chmod 766 /home/vagrant/data/face_videos
/home/vagrant/miniconda/bin/pip install ipython[notebook]
