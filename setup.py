"""
End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution
"""

from setuptools import setup

setup(
    name='easyocr',
    packages=['easyocr'],
    include_package_data=True,
    version='1.1.4',
    install_requires=['torch', 'torchvision>=0.5', 'opencv-python', 'scipy', 'numpy', 'Pillow', 'scikit-image'],
    entry_points={"console_scripts": ["easyocr= easyocr.cli:main"]},
    license='Apache License 2.0',
    description='End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution',
    author='Rakpong Kittinaradorn',
    author_email='r.kittinaradorn@gmail.com',
    url='https://github.com/jaidedai/easyocr',
    download_url='https://github.com/jaidedai/easyocr.git',
    keywords=['ocr optical character recognition deep learning neural network'],
    classifiers=[
        'Development Status :: 5 - Production/Stable'
    ],
)
