from setuptools import setup, find_packages

setup(
    name='GenSeg',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'mmcv-full>=1.3.0',
        'mmsegmentation>=0.18.0',
        'numpy',
        'opencv-python',
        'pillow',
        'scipy',
        'tqdm'
    ],
)