from setuptools import setup, find_packages
import multiprocessing
import subprocess

requirements_file = "requirements.txt"

# Check for CUDA JIT
try:
    result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if 'release' in result.stdout.decode('utf-8'):
        with open(requirements_file, 'a') as f:
            f.write('\ncupy\n')
except FileNotFoundError:
    pass

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
        install_requires=open(requirements_file).read().splitlines(),
        zip_safe=False,
        tests_require=[
            "pytest",
        ],
        python_requires='>=3.9',
        include_package_data=True,
    )