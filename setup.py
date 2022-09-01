"""
End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution
"""
import os
import subprocess
from io import open
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

def compile_dbnet_dcn(script_dir):
    script_path = os.path.join(script_dir, 'easyocr', 'scripts', 'compile_dbnet_dcn.py')
    subprocess.run(
        "python {}".format(script_path), shell=True
    )

class CustomCommand_install(install):
    def run(self):
        install.run(self)
        compile_dbnet_dcn(self.install_lib)

class CustomCommand_develop(develop):
    def run(self):
        develop.run(self)
        compile_dbnet_dcn(self.install_dir)

setup(
    name='easyocr',
    packages=['easyocr'],
    include_package_data=True,
    version='1.6.1',
    install_requires=requirements,
    entry_points={"console_scripts": ["easyocr= easyocr.cli:main"]},
    license='Apache License 2.0',
    description='End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Rakpong Kittinaradorn',
    author_email='r.kittinaradorn@gmail.com',
    url='https://github.com/jaidedai/easyocr',
    download_url='https://github.com/jaidedai/easyocr.git',
    keywords=['ocr optical character recognition deep learning neural network'],
    classifiers=[
        'Development Status :: 5 - Production/Stable'
    ],
    cmdclass={
        'install': CustomCommand_install,
        'develop': CustomCommand_develop,
    }
)
