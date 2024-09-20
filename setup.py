import multiprocessing
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from sphinx.setup_command import BuildDoc


requirements = "requirements.txt"

long_description = """
Accurate representation of thermal neutron scattering in Monte Carlo transport 
simulations requires that the molecular vibrations of the target material be 
accounted for. Historically, this has been achieved by precomputing large 
multidimensional tables that are a function of temperature and the cosine of the 
scattering angle, as well as incoming and outgoing neutron energy. 

Solid cinel is a Python package for solid state physics and materials science. 
It is a collection of tools for the analysis of  thermal neutron scattering by 
the creation of in fly multidimensional tables. It is designed to be easy to use 
and to provide a high-level interface to the most common tasks in solid state 
physics. It is built on top of the popular Python packages  pandas, numpy and 
numba.
"""
class build_py(_build_py):
    def run(self):
        self.run_command('build_sphinx')
        _build_py.run(self)

cmdclass = {
    'build_sphinx': BuildDoc,
    'build_py': build_py,
}

if __name__ == "__main__":
    # Freeze to support parallel compilation when using spawn instead of fork
    multiprocessing.freeze_support()
    setup(
        name='solid_cinel',
        use_scm_version=True,
        setup_requires=['setuptools_scm'],
        description='solid_cinel',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/AitorBengoechea/solid_cinel/',
        author='Aitor Bengoechea',
        author_email='aitorabf@gmail.com',
        classifiers=[
            'Development Status :: Beta',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords='solid_cinel, physics, materials, science',
        packages=find_packages(exclude=["tests"]),
        install_requires=open('requirements.txt').read().splitlines(),
        extras_require={
            'gpu':  ['cupy'],  # general cupy package
            'docs': ['sphinx', 'sphinx-rtd-theme'],
        },
        entry_points={
            'console_scripts': [
                'scinel = solid_cinel.application.scinel:main',
            ],
        },
        zip_safe=False,
        tests_require=[
            "pytest",
        ],
        python_requires='>=3.10',
        include_package_data=True,
        cmdclass=cmdclass,
        command_options={
            'build_sphinx': {
                'project': ('setup.py', 'solid_cinel'),
                'version': ('setup.py', '0.1.0'),
                'release': ('setup.py', '0.1.0'),
                'source_dir': ('setup.py', 'docs/source'),
                'build_dir': ('setup.py', 'docs/build'),
            }
        },
    )