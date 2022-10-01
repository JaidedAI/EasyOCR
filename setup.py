<<<<<<< HEAD
from setuptools import setup

if __name__ == "__main__":
    try:
        setup(use_scm_version={"version_scheme": "no-guess-dev"})
    except Exception:
        print(
            "\n\nAn error occurred while building the project, "
            + "please ensure you have the most updated version of setuptools, "
            + "setuptools_scm and wheel with:\n"
            + "   pip install -U setuptools setuptools_scm wheel\n\n",
        )
        raise
=======
"""
End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution
"""
from io import open
from setuptools import setup

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

setup(
    name='easyocr',
    packages=['easyocr'],
    include_package_data=True,
    version='1.6.2',
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

)
>>>>>>> 06753992c0aa7b9c74f46ce558bba4ba5a28493b
