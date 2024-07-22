# Function to initialize Conda in current shell
initialize_conda() {
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
}

# Step 1: Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Proceeding to download Miniconda."

    # Step 2 & 3: Download Miniconda installer script
    wget -O installation.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    # Step 4: Make the installer script executable
    chmod +x installation.sh

    echo "Miniconda installer script has been downloaded and made executable as 'installation.sh'."

    # Execute the installer script
    ./installation.sh -b -p $HOME/miniconda

    # Initialize Conda for the current shell session
    initialize_conda

    echo "Miniconda installed."
else
    echo "Conda is already installed."
    # Initialize Conda for the current shell session
    initialize_conda
fi

# Step 5: Create a new Conda environment named 'scinel'
if conda env list | grep -q 'scinel'; then
    echo "Conda environment 'scinel' already exists."
else
    conda create -y -n scinel "python>=3.10"
    echo "Conda environment 'scinel' has been created with Python version >= 3.10."
fi

# Activate the 'scinel' environment
source activate scinel

# Step 7: Upgrade pip
python -m pip install --upgrade pip

# Step 8: Install general python dependencies
pip install sphinx sphinx_rtd_theme numpydoc coveralls pytest-cov nbval

# Step 9: Install solid_cinel dependencies
python -m pip install . --user

echo "To use the program, you need to activate the Conda environment 'scinel' by running 'conda activate scinel'."