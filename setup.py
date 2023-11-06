import os
from setuptools import setup
import multiprocessing
from setuptools.extension import Extension
from Cython.Build import cythonize
requirements = "requirements.txt"

extensions = [
    Extension(
        "clm",
        ["solid_cinel/cython_modules/clm.pyx"],
        extra_compile_args=['-O3'],
        extra_link_args=[],
    )
]

if __name__ == "__main__":
    # Freeze to support parallel compilation when using spawn instead of fork
    multiprocessing.freeze_support()
    setup(
        name='solid_cinel',
        version='0.1',
        description='solid_cinel',
        url='https://github.com/AitorBengoechea/solid_cinel/',
        author='Aitor Bengoechea',
        author_email='aitorabf@gmail.com',
        classifiers=[
            'Development Status :: Beta',
            'Programming Language :: Python :: 3',
        ],
        data_files=[(x[0], list(map(lambda y: x[0] + '/' + y, x[2]))) for x in
                    os.walk('solid_cinel')],
        install_requires=open(requirements).read().splitlines(),
        zip_safe=False,
        # setup_requires=["pytest-runner",],
        tests_require=[
            "pytest",
        ],
        include_package_data=True,
        ext_modules=cythonize(extensions),
        # entry_points={
        #    'console_scripts': [
        #        'sandy=sandy.sampling:run',
        #       ],
        #    },
    )