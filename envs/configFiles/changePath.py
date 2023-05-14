import os 
import yaml
data = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}
# Get the list of files in the folder
files = os.listdir(os.getcwd())

# Filter out only the text files
text_files = [file for file in files if file.endswith('.yaml')]

# Print the names of the text files
for file_name in text_files:
    print(file_name)
    with open(file_name, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.SafeLoader)
        #yaml.dump(yaml_data, file)
        for line in yaml_data:
            #print(line)
            pass
    break
