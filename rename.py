import os
folder = "data/train/patient204/image"
# Iterate through the folder
for file in os.listdir(folder):
        # construct current name using file name and path
        old_name = os.path.join(folder, file)
        only_name = os.path.splitext(file)[0]

        # Adding the new name with extension
        new_base =  'img' + str(int(only_name)) + '.png'
        # construct full file path
        new_name = os.path.join(folder, new_base)
        # Renaming the file
        os.rename(old_name, new_name)
# verify the result
res = os.listdir(folder)
print(res)