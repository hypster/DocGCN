import os
# "ohsumed","R8","R52","20ng",
dataset = ["mr"]
for file in dataset:
    meta_file = f"https://github.com/yao8839836/text_gcn/raw/master/data/{file}.txt"
    content_file = f"https://github.com/yao8839836/text_gcn/raw/master/data/corpus/{file}.clean.txt"
    print(meta_file)
    print(content_file)

    # command = subprocess.run(["pwd"], stdout=PIPE, stdin=PIPE)
    # sys.stdout.buffer.write(command.stdout)
    os.system(f'curl {meta_file} -L --output {file}.txt')
    os.system(f'curl {content_file} -L --output {file}.clean.txt')

