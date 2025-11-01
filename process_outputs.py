import json
import argparse


def compile_results(json_file_name):
    count = 0.0
    score = 0.0
    with open(json_file_name, 'r') as f:
        for line in f:
            if not line.strip():  # skip empty lines
                continue
            record = json.loads(line)
            count = count + 1
            score = score + float(record['eval'])
    return score/count





if __name__=="__main__":
    parser = argparse.ArgumentParser('Parse json output files to provide score\n')
    parser.add_argument("filename", type=str, help='path to input json file')
    args = parser.parse_args()

    result = compile_results(args.filename)
    print('Result:', result)


