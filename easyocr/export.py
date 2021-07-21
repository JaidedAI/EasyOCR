import argparse

import onnx
import torch
import easyocr
import numpy as np


def export_detector(detector_onnx_save_path,
                    in_shape=[1, 3, 608, 800],
                    lang_list=["en"],
                    model_storage_directory=None,
                    user_network_directory=None,
                    download_enabled=True,
                    dynamic=True,
                    device="cpu",
                    quantize=True,
                    detector=True,
                    recognizer=True):
    if dynamic is False:
        print('WARNING: it is recommended to use -d dynamic flag when exporting onnx')
    ocr_reader = easyocr.Reader(lang_list,
                                gpu=False if device == "cpu" else True,
                                detector=detector,
                                recognizer=detector,
                                quantize=quantize,
                                model_storage_directory=model_storage_directory,
                                user_network_directory=user_network_directory,
                                download_enabled=download_enabled)

    # exporting detector if selected
    if detector:
        dummy_input = torch.rand(in_shape)
        dummy_input = dummy_input.to(device)

        # forward pass
        with torch.no_grad():
            y_torch_out, feature_torch_out = ocr_reader.detector(dummy_input)
            torch.onnx.export(ocr_reader.detector,
                              dummy_input,
                              detector_onnx_save_path,
                              export_params=True,
                              do_constant_folding=True,
                              opset_version=12,
                              # model's input names
                              input_names=['input'],
                              # model's output names, ignore the 2nd output
                              output_names=['output'],
                              # variable length axes
                              dynamic_axes={'input': {0: 'batch_size', 2: "height", 3: "width"},
                                            'output': {0: 'batch_size', 1: "dim1", 2: "dim2"}
                                            } if dynamic else None,
                              verbose=False)

        # verify exported onnx model
        detector_onnx = onnx.load(detector_onnx_save_path)
        onnx.checker.check_model(detector_onnx)
        print(f"Model Inputs:\n {detector_onnx.graph.input}\n{'*'*80}")
        print(f"Model Outputs:\n {detector_onnx.graph.output}\n{'*'*80}")

        # onnx inference validation
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(detector_onnx_save_path)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            return tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        y_onnx_out, feature_onnx_out = ort_session.run(None, ort_inputs)

        print(f"torch outputs: y_torch_out.shape={y_torch_out.shape} feature_torch_out.shape={feature_torch_out.shape}")
        print(f"onnx outputs: y_onnx_out.shape={y_onnx_out.shape} feature_onnx_out.shape={feature_onnx_out.shape}")
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(
            to_numpy(y_torch_out), y_onnx_out, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(
            to_numpy(feature_torch_out), feature_onnx_out, rtol=1e-03, atol=1e-05)

        print(f"Model exported to {detector_onnx_save_path} and tested with ONNXRuntime, and the result looks good!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang_list',
                        nargs='+', type=str,
                        default=["en"],
                        help='-l en ch_sim ... (language lists for easyocr)')
    parser.add_argument('-s', '--detector_onnx_save_path', type=str,
                        default="detector_craft.onnx",
                        help="export detector onnx file path ending in .onnx" +
                        "Do not pass in this flag to avoid exporting detector")
    parser.add_argument('-d', '--dynamic',
                        action='store_true',
                        help="Dynamic  input output shapes for detector")
    parser.add_argument('-is', '--in_shape',
                        nargs='+', type=int,
                        default=[1, 3, 608, 800],
                        help='-is 1 3 608 800 (bsize, channel, height, width)')
    parser.add_argument('-m', '--model_storage_directory', type=str,
                        help="model storage directory for craft model")
    parser.add_argument('-u', '--user_network_directory', type=str,
                        help="user model storage directory")
    args = parser.parse_args()
    dpath = args.detector_onnx_save_path
    args.detector_onnx_save_path = None if dpath == "None" else dpath
    if len(args.in_shape) != 4:
        raise ValueError(
            f"Input shape must have four values (bsize, channel, height, width) eg. 1 3 608 800")
    return args


def main():
    args = parse_args()
    export_detector(detector_onnx_save_path=args.detector_onnx_save_path,
                    in_shape=args.in_shape,
                    lang_list=args.lang_list,
                    model_storage_directory=args.model_storage_directory,
                    user_network_directory=args.user_network_directory,
                    dynamic=args.dynamic)


if __name__ == "__main__":
    main()
