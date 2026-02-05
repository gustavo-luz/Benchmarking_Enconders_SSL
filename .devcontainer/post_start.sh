#!/bin/bash


pip install --upgrade pip
pip install --upgrade setuptools
echo "🎈 Installing Minerva-Dev"
(cd Minerva-Dev && pip install -e .)

pip install pandasql
pip install graphviz
sudo pip install ray[default]==2.46.0
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch
pip install sqlitedict
pip3 install opencv-python-headless
# pip install jsonargparse==4.37.0
pip install --upgrade numba
pip uninstall numpy -y
pip install numpy==1.26.4
pip uninstall kaleido -y
pip install kaleido==0.2.1


echo "🎈 Everything installed"

# ------ Adding useful options --------
echo "🎈 Adding useful options...."

# Add tmux options
# Syncronize panes using Control+b e
# Disable syncronize panes using Control+b E
# Color support
# Mouse support
echo -e "set -g default-terminal \"screen-256color\"\nset -g mouse on\nbind e setw synchronize-panes on\nbind E setw synchronize-panes off" >> ~/.tmux.conf

# remove full path from prompt
sed -i '/^\s*PS1.*\\w/s/\\w/\\W/g' ~/.bashrc

# Setup the links to 
if [ -d "/workspaces/shared/data" ]; then
    ln -s "/workspaces/shared/data" shared_data
fi
if [ -d "/workspaces/shared/runs" ]; then
    ln -s "/workspaces/shared/runs" shared_runs
fi

echo "Done 🎉 🎈"
exit 0
