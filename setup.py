"""
End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution
"""
import os
import subprocess

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def compile_dbnet_dcn(script_dir):
    script_path = os.path.join(script_dir, "easyocr", "scripts", "compile_dbnet_dcn.py")
    subprocess.run("python {}".format(script_path), shell=True)


class CustomCommand_install(install):
    def run(self):
        install.run(self)
        compile_dbnet_dcn(self.install_lib)


class CustomCommand_develop(develop):
    def run(self):
        develop.run(self)
        compile_dbnet_dcn(self.install_dir)


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
