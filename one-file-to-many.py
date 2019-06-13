from pathlib import Path
import argparse
import re


def one_file_to_many(pfin, pdout, extension, relu):
    regex = r"\[?[ \t]*(-?\d+\.?\d*)\]?"
    with open(str(pfin), encoding='utf-8') as fhin:
        for idx, line in enumerate(fhin, 1):
            result = re.sub(regex, "\\1", line)
            if result:
                result = float(result)
                result = result if (result > 0 or not relu) else 0
                with open(str(pdout.joinpath(f"{idx}{extension}")), 'w', encoding='utf-8') as fhout:
                    fhout.write(str(result))
            else:
                print(f"WARNING: no regex match for line {idx}: {line}")

    return idx


def many_files_to_many(pdin, pdout, extension):
    regex = r"\[?[ \t]*(-?\d+\.?\d*)\]?"

    for idx, pfin in enumerate(pdin.glob('*'), 1):
        pfout = pdout.joinpath(pfin.with_suffix(extension).name)
        with open(str(pfin), encoding='utf-8') as fhin:
            line = ''.join(fhin.readlines()).strip()
            result = re.sub(regex, "\\1", line)
            if result:
                result = float(result)
                result = result if result > 0 else 0
                with open(str(pfout), 'w', encoding='utf-8') as fhout:
                    fhout.write(str(result))
            else:
                print(f"WARNING: no regex match for line {idx}: {line}")

    return idx


def main(pfin, pdin, pdout, extension, relu):
    if not pdin and not pdout:
        raise ValueError("-f or '-d' is required.")

    pdout.mkdir(parents=True)

    if pfin:
        nro_writes = one_file_to_many(pfin, pdout, extension, relu)
    elif pdin:
        nro_writes = many_files_to_many(pdin, pdout, extension)

    print(f"Finished. Wrote {str(nro_writes)} files to {str(pdout)}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Separate all lines a file into separate documents.')
    parser.add_argument('-f', default=None, help="Path to input file. Input file will be separate across files.")
    parser.add_argument('-d', default=None, help="Path to input dir. All files in input dir will be converted to"
                                                 " non-negative. (ReLU)")
    parser.add_argument('-o', '--output_dir', required=True, help="Path to output dir.")

    parser.add_argument('-x', '--extension', default='.cross', help="Extension for the generated files.")
    parser.add_argument('-r', '--relu', default=False, action='store_true', help="Convert negative values to zero.")

    args = parser.parse_args()

    input_path = Path(args.f).resolve() if args.f else None
    input_dir = Path(args.d).resolve() if args.d else None
    output_d_path = Path(args.output_dir).resolve()
    main(input_path, input_dir, output_d_path, args.extension, args.relu)
