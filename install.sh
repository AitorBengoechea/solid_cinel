# Upgrade pip
python -m pip install --upgrade pip
# install general python dependencies
pip install sphinx sphinx_rtd_theme numpydoc coveralls pytest-cov nbval
# install solid_cinel
python -m pip install . --user