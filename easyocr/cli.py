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
        "-f",
        "--file",
        required=True,
        type=str,
        help="input file",
    )
    parser.add_argument(
        "--detail",
        type=int,
        choices=[0, 1],
        default=1,
        help="simple output (default: 1)",
    )
    parser.add_argument(
        "--gpu",
        type=bool,
        choices=[True, False],
        default=True,
        help="Using GPU (default: True)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reader = easyocr.Reader(lang_list=args.lang, gpu=args.gpu)
    for line in reader.readtext(args.file, detail=args.detail):
        print(line)


if __name__ == "__main__":
    main()
