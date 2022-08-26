import os
from unit_test import UnitTest

# %% Set up paths 
easyocr_module = "../easyocr"
verbose = 2
test_data = "./data/EasyOcrUnitTestPackage.pickle"
image_data_dir = "../examples"

# %% Initialize UnitTest
unit_test = UnitTest(easyocr_module, 
                     test_data,
                     image_data_dir
                     )
# %% Run UnitTest at verbosity level 2
unit_test.do_test(verbose = 2)