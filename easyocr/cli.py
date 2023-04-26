import argparse
import easyocr


def parse_args():
    parser = argparse.ArgumentParser(description="Process EasyOCR.")
    parser.add_argument(
        "-l",
        "--lang",
        nargs='+',
        required=True,
        type=str,
        help="for languages",
    )
    parser.add_argument(
        "--gpu",
        type=bool,
        choices=[True, False],
        default=True,
        help="Using GPU (default: True)",
    )
    parser.add_argument(
        "--model_storage_directory",
        type=str,
        default=None,
        help="Directory for model (.pth) file",
    )
    parser.add_argument(
        "--user_network_directory",
        type=str,
        default=None,
        help="Directory for custom network files",
    )
    parser.add_argument(
        "--recog_network",
        type=str,
        default='standard',
        help="Recognition networks",
    )
    parser.add_argument(
        "--download_enabled",
        type=bool,
        choices=[True, False],
        default=True,
        help="Enable Download",
    )
    parser.add_argument(
        "--detector",
        type=bool,
        choices=[True, False],
        default=True,
        help="Initialize text detector module",
    )
    parser.add_argument(
        "--recognizer",
        type=bool,
        choices=[True, False],
        default=True,
        help="Initialize text recognizer module",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        choices=[True, False],
        default=True,
        help="Print detail/warning",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        choices=[True, False],
        default=True,
        help="Use dynamic quantization",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        type=str,
        help="input file",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["greedy", 'beamsearch', 'wordbeamsearch'],
        default='greedy',
        help="decoder algorithm",
    )
    parser.add_argument(
        "--beamWidth",
        type=int,
        default=5,
        help="size of beam search",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch_size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of processing cpu cores",
    )
    parser.add_argument(
        "--allowlist",
        type=str,
        default=None,
        help="Force EasyOCR to recognize only subset of characters",
    )
    parser.add_argument(
        "--blocklist",
        type=str,
        default=None,
        help="Block subset of character. This argument will be ignored if allowlist is given.",
    )
    parser.add_argument(
        "--detail",
        type=int,
        choices=[0, 1],
        default=1,
        help="simple output (default: 1)",
    )
    parser.add_argument(
        "--rotation_info",
        type=list,
        default=None,
        help="Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations.",
    )
    parser.add_argument(
        "--paragraph",
        type=bool,
        choices=[True, False],
        default=False,
        help="Combine result into paragraph",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=20,
        help="Filter text box smaller than minimum value in pixel",
    )
    parser.add_argument(
        "--contrast_ths",
        type=float,
        default=0.1,
        help="Text box with contrast lower than this value will be passed into model 2 times. First is with original image and second with contrast adjusted to 'adjust_contrast' value. The one with more confident level will be returned as a result.",
    )
    parser.add_argument(
        "--adjust_contrast",
        type=float,
        default=0.5,
        help="target contrast level for low contrast text box",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.7,
        help="Text confidence threshold",
    )
    parser.add_argument(
        "--low_text",
        type=float,
        default=0.4,
        help="Text low-bound score",
    )
    parser.add_argument(
        "--link_threshold",
        type=float,
        default=0.4,
        help="Link confidence threshold",
    )
    parser.add_argument(
        "--canvas_size",
        type=int,
        default=2560,
        help="Maximum image size. Image bigger than this value will be resized down.",
    )
    parser.add_argument(
        "--mag_ratio",
        type=float,
        default=1.,
        help="Image magnification ratio",
    )
    parser.add_argument(
        "--slope_ths",
        type=float,
        default=0.1,
        help="Maximum slope (delta y/delta x) to considered merging. Low value means tiled boxes will not be merged.",
    )
    parser.add_argument(
        "--ycenter_ths",
        type=float,
        default=0.5,
        help="Maximum shift in y direction. Boxes with different level should not be merged.",
    )
    parser.add_argument(
        "--height_ths",
        type=float,
        default=0.5,
        help="Maximum different in box height. Boxes with very different text size should not be merged. ",
    )
    parser.add_argument(
        "--width_ths",
        type=float,
        default=0.5,
        help="Maximum horizontal distance to merge boxes.",
    )
    parser.add_argument(
        "--y_ths",
        type=float,
        default=0.5,
        help="Maximum Vertical distance to merge boxes (when paragraph = True).",
    )
    parser.add_argument(
        "--x_ths",
        type=float,
        default=1.0,
        help="Maximum horizontal distance to merge boxes (when paragraph = True).",
    )
    parser.add_argument(
        "--add_margin",
        type=float,
        default=0.1,
        help="Extend bounding boxes in all direction by certain value. This is important for language with complex script (E.g. Thai).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["standard", 'dict', 'json'],
        default='standard',
        help="output format.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reader = easyocr.Reader(lang_list=args.lang,\
                            gpu=args.gpu,\
                            model_storage_directory=args.model_storage_directory,\
                            user_network_directory=args.user_network_directory,\
                            recog_network=args.recog_network,\
                            download_enabled=args.download_enabled,\
                            detector=args.detector,\
                            recognizer=args.recognizer,\
                            verbose=args.verbose,\
                            quantize=args.quantize)
    for line in reader.readtext(args.file,\
                                decoder=args.decoder,\
                                beamWidth=args.beamWidth,\
                                batch_size=args.batch_size,\
                                workers=args.workers,\
                                allowlist=args.allowlist,\
                                blocklist=args.blocklist,\
                                detail=args.detail,\
                                rotation_info=args.rotation_info,\
                                paragraph=args.paragraph,\
                                min_size=args.min_size,\
                                contrast_ths=args.contrast_ths,\
                                adjust_contrast=args.adjust_contrast,\
                                text_threshold=args.text_threshold,\
                                low_text=args.low_text,\
                                link_threshold=args.link_threshold,\
                                canvas_size=args.canvas_size,\
                                mag_ratio=args.mag_ratio,\
                                slope_ths=args.slope_ths,\
                                ycenter_ths=args.ycenter_ths,\
                                height_ths=args.height_ths,\
                                width_ths=args.width_ths,\
                                y_ths=args.y_ths,\
                                x_ths=args.x_ths,\
                                add_margin=args.add_margin,\
                                output_format=args.output_format):
        print(line)


if __name__ == "__main__":
    main()
