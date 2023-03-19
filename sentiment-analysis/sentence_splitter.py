import csv
import re

def sentence_splitter(input_file, output_file):
    with open(input_file, 'r') as f:
        with open(output_file, 'w', newline='') as out:
            writer = csv.writer(out)
            in_section = False
            for line in f:
                # Remove any base 10 numbers (decimals)
                line = re.sub(r'\d+\.\d+|\d+', '', line)
                # Remove leading and trailing whitespace and periods
                line = line.strip().strip('.')
                if '"' in line:
                    # Start of a new section
                    if not in_section:
                        out.write('""\n')
                        in_section = True
                    # Add quotes around the line
                    line = f'"{line}"'
                else:
                    # End of a section
                    if in_section:
                        out.write('""\n')
                        in_section = False
                out.write(line + '\n')

if __name__ == "__sentence_splitter__":
    sentence_splitter("iliad.txt", "iliad_serpated.csv")