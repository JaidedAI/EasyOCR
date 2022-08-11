# Unit Test

##Description
This module contains unit test for EasyOCR.

## Usage
This module can be used as a typical python module. One python wrapper script and on ipython notebook are provided.

### Python script (*recommneded*)
The script can be called with (assuming calling from `EasyOCR/`);
```
python ./unit_test/run_unit_test.py --easyocr ./easyocr --verbose 2 --test ./unit_test/EasyOcrUnitTestPackage.pickle --data_dir ./examples 
```

#### Script arguments
 * easyocr: [Required] EasyOCR package to test. This should point to a directory where `__init__.py` of EasyOCR is located.
 * verbose (-v): [Optional] Verbosity level to report test results (The default is 0)
    * 0: Report only the final result
    * 1: Same as 0 and also results of each tested module.
    * 2: Same as 1 and also results of each test of each module.
    * 3: Same as 2 and also the calculated and the expected outputs of each test.
    * 4 or higher: Same as 3 and also the inputs of each test. (This will produce a lot of text on console).
 * test_data (-t): [Optional] Path to test package to use (The default is `./unit_test/data/EasyOcrUnitTestPackage.pickle`).
 * data_dir (-d): [Optional] Path to EasyOCR example images directory. (The default is `./examples/`
 
### Ipython notebook
Please see `demo.ipynb` for documentation.