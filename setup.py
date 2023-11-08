from setuptools import setup, find_packages
import multiprocessing
requirements = "requirements.txt"

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
    )