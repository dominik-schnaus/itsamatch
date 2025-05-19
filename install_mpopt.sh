CURRENT_ENV=$(dirname $(dirname $(which python)))

TMP_PATH=${1:-/tmp/mpopt}

mkdir -p $TMP_PATH
cd $TMP_PATH

# Download the latest version of mpopt
git clone https://github.com/vislearn/libmpopt.git
cd libmpopt

# Create a virtual environment
VENV_PATH=$TMP_PATH/venv
python3 -m venv $VENV_PATH
source $VENV_PATH/bin/activate

# if the python version is later than 3.11, change the MESON_VERSION=0.64.1 in ./scripts/install-into-venv to MESON_VERSION=1.8.0
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1-2)
if [[ $PYTHON_VERSION > "3.11" ]]; then
    sed -i 's/MESON_VERSION=0.64.1/MESON_VERSION=1.8.0/g' ./scripts/install-into-venv
fi

# run ./scripts/install-into-venv
./scripts/install-into-venv

# copy the include and lib files to the current environment
cp -r $VENV_PATH/include/mpopt $CURRENT_ENV/include/
cp -r $VENV_PATH/include/qpbo $CURRENT_ENV/include/
cp -r $VENV_PATH/lib/python${PYTHON_VERSION}/site-packages/mpopt/ $CURRENT_ENV/lib/python${PYTHON_VERSION}/site-packages/

deactivate

# Clean up
cd ..
rm -rf $TMP_PATH
echo "Cleaning up temporary files..."

echo "mpopt installed successfully in ${CURRENT_ENV}"