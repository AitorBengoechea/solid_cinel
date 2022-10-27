setup(
      name='solid_cinel',
      version='0.1',
      description='solid_cinel',
      url='https://github.com/AitorBengoechea/solid_cinel',
      author='Aitor Bengoechea',
      author_email='aitorabf@gmail.com',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3',
          ],
      #keywords=", ".join(keywords),
      # packages = find_packages(),
      data_files=[(x[0], list(map(lambda y: x[0]+'/'+y, x[2]))) for x in os.walk('solid_cinel')],
      #install_requires=open(requirements).read().splitlines(),
      zip_safe=False,
      # setup_requires=["pytest-runner",],
      tests_require=[
          "pytest",
          ],
      include_package_data=True,
      #ext_modules=extensions,
      #entry_points={
      #    'console_scripts': [
      #        'sandy=sandy.sampling:run',
      #        ],
      #    },
      )
