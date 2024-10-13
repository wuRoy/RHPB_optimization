from setuptools import setup, find_packages

setup(
    name='RHP_blend_optimization',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True
)
