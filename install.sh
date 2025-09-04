# Step 7: Upgrade pip
python -m pip install --upgrade pip

# Step 8: Install general python dependencies
pip install sphinx sphinx_rtd_theme numpydoc coveralls pytest-cov nbval

# Step 9: Install solid_cinel dependencies
python -m pip install . --user