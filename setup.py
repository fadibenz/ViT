from setuptools import setup, find_packages

setup(
    name='ViT',
    version='0.1.0',
    description='',
    author='Fadi Benzaima',
    packages=find_packages(where='ViT'),
    package_dir={'': 'ViT'},
    install_requires=[
        'torch',
        'torchvision',
        'einops'
    ],
)