import json 
from datasets import load_dataset

N = 10000
N_start = 5000
N_end = N

filename_b = 
filename_output = 

def format_chat(prompt, response):


    return example


def main():
    dataset_a = load_dataset("wiki_bio", split="train")
    subset_a = dataset_a.select(range(N))['target_text']

    dataset_b = load_dataset("json", data_files=filename_b,split="train")
    subset_b = dataset_b['response']

    with open(filename_output, "w") as f:
        for n in range(N_start, N_end):
            if dataset_b['eval'] == True:
                example = format_chat(subset_a[n], subset_b[n])
                f.write(example + '\n')
    return


if __name__ = "__main__":
    main()



