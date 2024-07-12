# Upgrade pip
python -m pip install --upgrade pip
# install general python dependencies
pip install sphinx sphinx_rtd_theme numpydoc coveralls pytest-cov nbval
# install solid cinel requirements
pip install -r requirements.txt
# install solid_cinel
python -m pip install . --user