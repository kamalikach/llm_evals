import json 
from datasets import load_dataset

N = 10000
N_start = 5000
N_end = N

filename_b = "./wiki_bio.txt" 
filename_output = "wikibio_messages.jsonl" 

system_prompt_file = "/home/kamalika/llm_evals/system_prompts/convert_json.txt"

def format_chat(prompt, response):
    example = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }
    return json.dumps(example, ensure_ascii=False, indent=2)

def main():
    dataset_a = load_dataset("wiki_bio", split="train")
    subset_a = dataset_a.select(range(N))['target_text']

    dataset_b = load_dataset("json", data_files=filename_b,split="train")
    subset_b = dataset_b['response']

    with open(system_prompt_file, "r") as f:
        system_prompt = f.read()

    with open(filename_output, "w") as f:
        for n in range(N_start, N_end):
            print(subset_a[n])
            print(dataset_b[n]['response'])
            print(dataset_b[n]['eval'])
            if dataset_b[n]['eval']:
                example = format_chat(system_prompt + ":" + subset_a[n], subset_b[n])
                f.write(example + '\n')
    return


if __name__ == "__main__":
    main()



