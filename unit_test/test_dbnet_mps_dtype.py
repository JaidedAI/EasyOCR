import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array)
        self.dtype = self._array.dtype

    def float(self):
        return _FakeTensor(self._array.astype(np.float32))

    def to(self, device):
        if device == "mps" and self.dtype == np.float64:
            raise TypeError(
                "Cannot convert a MPS Tensor to float64 dtype as the MPS "
                "framework doesn't support float64. Please use float32 instead."
            )
        return self


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_fake_torch():
    torch_mod = types.ModuleType("torch")
    torch_mod.from_numpy = lambda arr: _FakeTensor(arr)
    torch_mod.no_grad = _FakeNoGrad
    torch_mod.backends = types.SimpleNamespace(cudnn=types.ModuleType("cudnn"))
    return torch_mod


@contextmanager
def _load_dbnet_modules():
    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "easyocr",
            "easyocr.DBNet",
            "easyocr.DBNet.model",
            "easyocr.DBNet.model.constructor",
            "easyocr.DBNet.DBNet",
            "easyocr.detection_db",
            "torch",
            "torch.backends",
            "torch.backends.cudnn",
            "cv2",
            "pyclipper",
            "shapely",
            "shapely.geometry",
        )
    }
    try:
        fake_torch = _make_fake_torch()
        sys.modules["torch"] = fake_torch
        sys.modules["torch.backends"] = fake_torch.backends
        sys.modules["torch.backends.cudnn"] = fake_torch.backends.cudnn
        sys.modules["cv2"] = types.ModuleType("cv2")
        sys.modules["pyclipper"] = types.ModuleType("pyclipper")
        shapely_mod = types.ModuleType("shapely")
        shapely_geom = types.ModuleType("shapely.geometry")
        shapely_geom.Polygon = object
        shapely_mod.geometry = shapely_geom
        sys.modules["shapely"] = shapely_mod
        sys.modules["shapely.geometry"] = shapely_geom

        for pkg_name, pkg_path in (
            ("easyocr", REPO_ROOT / "easyocr"),
            ("easyocr.DBNet", REPO_ROOT / "easyocr" / "DBNet"),
            ("easyocr.DBNet.model", REPO_ROOT / "easyocr" / "DBNet" / "model"),
        ):
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_path)]
            sys.modules[pkg_name] = pkg

        def load_submod(mod_name, rel_path):
            spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            return mod

        load_submod("easyocr.DBNet.model.constructor", "easyocr/DBNet/model/constructor.py")
        dbnet_mod = load_submod("easyocr.DBNet.DBNet", "easyocr/DBNet/DBNet.py")
        detection_db_mod = load_submod("easyocr.detection_db", "easyocr/detection_db.py")
        yield dbnet_mod, detection_db_mod
    finally:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_dbnet_normalize_image_returns_float32():
    with _load_dbnet_modules() as (dbnet_mod, _):
        detector = dbnet_mod.DBNet(initialize_model=False, device="mps")
        assert detector.BGR_MEAN.dtype == np.float32

        sample_uint8 = np.zeros((32, 32, 3), dtype=np.uint8)
        sample_float32 = np.zeros((32, 32, 3), dtype=np.float32)
        sample_float64 = np.zeros((32, 32, 3), dtype=np.float64)
        assert detector.normalize_image(sample_uint8).dtype == np.float32
        assert detector.normalize_image(sample_float32).dtype == np.float32
        assert detector.normalize_image(sample_float64).dtype == np.float32


def test_detection_db_test_net_converts_to_float32_before_mps_transfer():
    with _load_dbnet_modules() as (dbnet_mod, detection_db_mod):
        dbnet = dbnet_mod.DBNet(initialize_model=False, device="mps")
        captured_dtypes = []

        class _StubDetector:
            def resize_image(self, img, canvas_size=None):
                return img, img.shape[:2]

            def normalize_image(self, img):
                return dbnet.normalize_image(img)

            def image2hmap(self, image_tensor):
                captured_dtypes.append(image_tensor.dtype)
                return image_tensor

            def hmap2bbox(self, *args, **kwargs):
                return [[]], [[]]

        sample = np.zeros((32, 32, 3), dtype=np.float32)
        bboxes, polys = detection_db_mod.test_net(sample, _StubDetector(), device="mps")
        assert captured_dtypes == [np.float32]
        assert bboxes == [[]]
        assert polys == [[]]


def test_detection_db_test_net_casts_float64_normalized_images_before_mps_transfer():
    with _load_dbnet_modules() as (_, detection_db_mod):
        captured_dtypes = []

        class _Float64Detector:
            def resize_image(self, img, canvas_size=None):
                return img, img.shape[:2]

            def normalize_image(self, img):
                return img.astype(np.float64)

            def image2hmap(self, image_tensor):
                captured_dtypes.append(image_tensor.dtype)
                return image_tensor

            def hmap2bbox(self, *args, **kwargs):
                return [[]], [[]]

        sample = np.zeros((32, 32, 3), dtype=np.uint8)
        bboxes, polys = detection_db_mod.test_net(sample, _Float64Detector(), device="mps")
        assert captured_dtypes == [np.float32]
        assert bboxes == [[]]
        assert polys == [[]]
