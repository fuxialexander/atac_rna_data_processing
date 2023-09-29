from setuptools import setup, find_packages

setup(
    name='atac_rna_data_processing',
    version='0.1',
    packages=find_packages(),
    package_data={
        'atac_rna_data_processing': ['data/*.pkl', 'data/gencode.v40.hg38.feather'],
    },
    install_requires=[
    ],
)