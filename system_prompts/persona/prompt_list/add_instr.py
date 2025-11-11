from pathlib import Path

path = Path('.')

to_add = "\n VERY IMPORTANT: Answer ONLY with Yes or No, DO NOT SAY ANYTHING ELSE \n"

for filename in path.glob('*.txt'):
    print(filename)

    with open(filename, "a") as f:
        f.write(to_add)


