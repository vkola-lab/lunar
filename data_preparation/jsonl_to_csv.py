import json
import csv
import sys
import random

def jsonl_to_csv(jsonl_path, csv_path):
    with open(jsonl_path, mode='r', encoding='utf-8') as jsonl_file:
        lines = [json.loads(line) for line in jsonl_file if line.strip()]
    
    if not lines:
        print("The JSONL file is empty or invalid.")
        return

    # Get all unique keys across all JSON objects
    fieldnames = set()
    for line in lines:
        fieldnames.update(line.keys())
    fieldnames = sorted(fieldnames)  # Consistent column order

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for line in lines:
            writer.writerow(line)

if __name__ == "__main__":
    # Example usage: python script.py input.jsonl output.csv
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_jsonl = sys.argv[1]
    output_csv = sys.argv[2]
    jsonl_to_csv(input_jsonl, output_csv)
