import csv
import re

def remove_numbers(input_file):
    with open(input_file, 'r') as f:
        # Initialize section and preceding lines
        section = ""
        preceding_lines = ""

        for line in f:
            # Check if line only has 1 space and no other characters - as modified to iliad_separated.csv by the sentence_splitter.py program
            if re.match(r'^\s*$', line):
                # Makes a new file for the section
                with open(preceding_lines.strip() + '.csv', 'w', newline='') as out:
                    writer = csv.writer(out)
                    # write the preceding parts of the section into the CSV file
                    writer.writerow([preceding_lines.strip()])
                    # Write the current section into the CSV file
                    for row in section.strip().split('\n'):
                        writer.writerow(row.strip().split(','))

                # Reset section and preceding lines for the next line in our input file
                section = ""
                preceding_lines = ""
            else:
                # Add current line to section
                section += line

                # Add current line to preceding lines
                preceding_lines += line

if __name__ == "__remove_numbers__":
    remove_numbers("iliad.txt")