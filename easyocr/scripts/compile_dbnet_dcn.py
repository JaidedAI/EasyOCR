'''
Created by Jaided AI
Released Date: 19/08/2022
Description:
A wrapper for DCN operator for DBNet. This script is called inside the setup.py
of EasyOCR. It can also be called as a standalone script to compile the operator
manually.
'''
import os
import subprocess

def main():
    url = "https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/DBNet"
    cwd = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir,'DBNet', 'log.txt'), "w") as fid:
        try:
            print("Compiling DCN operator...")
            os.chdir(os.path.join(parent_dir,'DBNet','assets','ops','dcn'))
            subprocess.run(
                "python setup.py build_ext --inplace", shell=True, stdout = fid
            )
            os.chdir(os.path.join(parent_dir,'DBNet'))
            subprocess.run(
                "touch dcn_compiling_success", shell=True, stdout = fid
            )
            os.chdir(cwd)
            print("DCN operator is compiled successfully.")
        except Exception as error:
            print("Failed to compile dcn operator for DBNet with the following error.", file = fid)
            print("{}".format(error), file = fid)
            print("Failed to compile dcn operator for DBNet.")
            print("EasyOCR can still be used with CRAFT text detector (default).")
            print("To use DBNet text detector, please check {} for troubleshoot and compile dcn operator manually.".format(url))

if __name__ == '__main__':
    main()
