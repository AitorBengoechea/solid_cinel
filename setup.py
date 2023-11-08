from setuptools import setup, find_packages
import multiprocessing
from setuptools.extension import Extension
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
requirements = "requirements.txt"

# Check if C compiler is available
compiler = new_compiler()
customize_compiler(compiler)
if compiler.has_function('printf'):  # Check for a basic function
    # If compiler is available, compile Cython extensions
    from Cython.Build import cythonize
    extensions = [
        Extension(
            "solid_cinel.CLM",
            ["solid_cinel/cython_modules/clm.pyx"],
            extra_compile_args=['-fopenmp', '-O3'],
            extra_link_args=['-fopenmp'],
            include_dirs=[np.get_include()],
        )
    ]
    extension = cythonize(extensions)
else:
    extension = []

if __name__ == "__main__":
    # Freeze to support parallel compilation when using spawn instead of fork
    multiprocessing.freeze_support()
    setup(
        name='solid_cinel',
        use_scm_version=True,
        setup_requires=['setuptools_scm'],
        description='solid_cinel',
        url='https://github.com/AitorBengoechea/solid_cinel/',
        author='Aitor Bengoechea',
        author_email='aitorabf@gmail.com',
        classifiers=[
            'Development Status :: Beta',
            'Programming Language :: Python :: 3',
        ],
        packages=find_packages(exclude=["tests"]),
        install_requires=open(requirements).read().splitlines(),
        zip_safe=False,
        # setup_requires=["pytest-runner",],
        tests_require=[
            "pytest",
        ],
        python_requires='>=3.9',
        include_package_data=True,
        ext_modules=extension,
    )