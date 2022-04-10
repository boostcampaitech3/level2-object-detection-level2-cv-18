
path = '/opt/ml/detection/dataset/test.txt'

new_text_content = ''

with open(path,'r') as f:
    lines = f.readlines()
    for i,l in enumerate(lines):
        new_string = lines[i].replace('./images','/opt/ml/detection/dataset')

        if new_string:
            new_text_content += new_string
    
with open(path,'w') as f:
    f.write(new_text_content)
        

