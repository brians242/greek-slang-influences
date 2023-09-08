import csv
import re

def main(input_file):
    with open(input_file, 'r') as f:
        count = 0
        sentence = ""
        for line in f:
            # Adds the line to curr sentence
            sentence += line.strip() + " "
            # check line for a period
            if re.search(r'(?<=\w)\.(?=\s|$)', sentence):
                # no clue
                sentence1 = re.findall(r'(.*?\.)(?=\s|$)', sentence)
                for i in sentence1:
                    # remove base 10
                    i = re.sub(r'\d+\.\d+|\d+', '', i)
                    # remove "" and periods
                    i = i.strip().strip('.')
                    if not i:
                        continue
                    # add to count for file names
                    count += 1
                    filename = f"{input_file.split('.')[0]}_{count}.csv"
                    with open(filename, 'w', newline='') as out:
                        writer = csv.writer(out)
                        writer.writerow([i])
                sentence = ""
            else:
                continue
        # remaining text is reconsidered
        if sentence:
            # same as above
            sentence = re.sub(r'\d+\.\d+|\d+', '', sentence)
            # same as above
            sentence = sentence.strip().strip('.')
            if sentence:
                # add to count for file names
                count += 1
                filename = f"{input_file.split('.')[0]}_{count}.csv"
                with open(filename, 'w', newline='') as out:
                    writer = csv.writer(out)
                    writer.writerow([sentence])

# make sure the main function only runs in this program
if __name__ == "__main__":
    main("book2.txt")