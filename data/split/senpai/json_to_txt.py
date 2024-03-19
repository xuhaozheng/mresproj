import json


json_file_path = '/Users/liyangke/Documents/Programs/mresproj/split/senpai/test.json'
output_file_path = '/Users/liyangke/Documents/Programs/mresproj/split/senpai/test.txt'


with open(json_file_path, 'r') as file:
    data = json.load(file)


with open(output_file_path, 'w') as file:
    for item in data['image']:
        file_path = item['file_name']
        # no zero
        if '/0.png' in file_path:
            continue
        second_slash_index = file_path.find('/', file_path.find('/') + 1)
        modified_file_path = file_path[second_slash_index + 1:]
        cls = item['cls']
        file.write(f"{modified_file_path} {cls}\n")


