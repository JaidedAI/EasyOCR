"""
Created by Jaided AI
Released Date: 19/08/2022
Description:
A wrapper for DCN operator for DBNet. This script is called inside the setup.py
of EasyOCR. It can also be called as a standalone script to compile the operator
manually.
"""
import glob
import os
import subprocess
import sys
from datetime import datetime


def print_error(errors, log_path):
    if not isinstance(errors, list):
        errors = [errors]
    errors = [
        error if isinstance(error, bytes) else error.encode("utf-8") for error in errors
    ]
    url = "https://github.com/JaidedAI/EasyOCR/tree/master/easyocr/DBNet"
    print("Failed to compile dcn operator for DBNet.")
    with open(log_path, "wb") as fid:
        fid.write(
            (datetime.now().strftime("%H:%M:%S - %d %b %Y") + "\n").encode("utf-8")
        )
        fid.write(
            "Failed to compile dcn operator for DBNet with the following error.\n".encode(
                "utf-8"
            )
        )
        fid.write(("#" * 42 + "\n").encode("utf-8"))
        [fid.write(error) for error in errors]
    print("Error message can be found in {}.".format(os.path.abspath(log_path)))
    print("#" * 42)
    print("EasyOCR can still be used with CRAFT text detector (default).")
    print(
        "To use DBNet text detector, please check {} for troubleshoot and compile dcn operator manually.".format(
            url
        )
    )


def print_success(text, log_path):
    with open(log_path, "wb") as fid:
        fid.write(
            (datetime.now().strftime("%H:%M:%S - %d %b %Y") + "\n").encode("utf-8")
        )
        fid.write((text + "\n").encode("utf-8"))
    print(text)


def validate_compilation(parent_dir, log_path, cpu_or_cuda):
    dcn_dir = os.path.join(parent_dir, "DBNet", "assets", "ops", "dcn")
    # Just to be safe, check explicitly.
    if cpu_or_cuda == "cpu":
        conv_cpu_exist = glob.glob(
            os.path.join(dcn_dir, "deform_conv_cpu.*.so")
        ) or glob.glob(os.path.join(dcn_dir, "deform_conv_cpu.*.pyd"))
        pool_cpu_exist = glob.glob(
            os.path.join(dcn_dir, "deform_pool_cpu.*.so")
        ) or glob.glob(os.path.join(dcn_dir, "deform_pool_cpu.*.pyd"))
        return conv_cpu_exist and pool_cpu_exist
    elif cpu_or_cuda == "cuda":
        conv_cuda_exist = glob.glob(
            os.path.join(dcn_dir, "deform_conv_cuda.*.so")
        ) or glob.glob(os.path.join(dcn_dir, "deform_conv_cuda.*.pyd"))
        pool_cuda_exist = glob.glob(
            os.path.join(dcn_dir, "deform_pool_cuda.*.so")
        ) or glob.glob(os.path.join(dcn_dir, "deform_pool_cuda.*.pyd"))
        return conv_cuda_exist and pool_cuda_exist
    else:
        raise ValueError("'cpu_or_cuda' must be either 'cpu' or 'cuda'")


def main():
    cwd = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(parent_dir, "DBNet", "log.txt")
    try:
        if sys.platform == "win32":
            os.environ["CXX"] = os.environ["CXX_"]
        print("Compiling DCN operators...")
        os.chdir(os.path.join(parent_dir, "DBNet", "assets", "ops", "dcn"))
        result = subprocess.run(
            ["python", "setup.py", "build_ext", "--inplace"], capture_output=True
        )
        if result.returncode == 0:
            os.chdir(os.path.join(parent_dir, "DBNet"))
            if validate_compilation(parent_dir, log_path, "cpu"):
                success_message = (
                    "DCN CPU operator is compiled successfully at {}.".format(
                        os.path.abspath(os.path.join(parent_dir, "DBNet"))
                    )
                )
                print_success(success_message, log_path)
                with open("dcn_cpu_compiling_success", "w+") as fp:
                    fp.write("")
            if validate_compilation(parent_dir, log_path, "cuda"):
                success_message = (
                    "DCN CUDA operator is compiled successfully at {}.".format(
                        os.path.abspath(os.path.join(parent_dir, "DBNet"))
                    )
                )
                print_success(success_message, log_path)
                with open("dcn_cuda_compiling_success", "w+") as fp:
                    fp.write("")
            success_message = "DCN operator is compiled successfully at {}.".format(
                os.path.abspath(os.path.join(parent_dir, "DBNet"))
            )
            print_success(success_message, log_path)
        else:
            print(result.__dict__)
            print_error([result.stdout, result.stderr], log_path)
    except Exception as error:
        print_error("{}".format(error), log_path)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
