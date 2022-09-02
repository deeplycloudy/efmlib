from setuptools import setup, find_packages

setup(name='efmlib',
    version=0.1,
    description='Balloon-borne electric field meter signal processing',
    packages=find_packages(),
    url='https://github.com/deeplycloudy/efmlib/',
    long_description=open('README.md').read(),
    include_package_data=True,
    )