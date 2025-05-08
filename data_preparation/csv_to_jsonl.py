import csv
import json
import sys

def csv_to_jsonl(csv_path, jsonl_path):
    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        with open(jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in reader:
                json.dump(row, jsonl_file)
                jsonl_file.write('\n')

if __name__ == "__main__":
    # Example usage: python csv_to_jsonl.py input.csv output.jsonl
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_file> <output_jsonl_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_jsonl = sys.argv[2]
    csv_to_jsonl(input_csv, output_jsonl)
