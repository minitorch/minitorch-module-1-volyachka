import os
import shutil
import sys

if len(sys.argv) != 3:
    print(
        "Invalid argument count! Please pass source directory and destination directory after the file name"
    )
    sys.exit()

# Get the users path to evaluate the username and root directory
current_path = r"C:\Users\user\Documents\workspace\minitorch-module-1-volyachka"
grandparent_path = r"C:\Users\user\Documents\workspace\minitorch-module-0-volyachka"

print(current_path)
print(grandparent_path)

print("Looking for modules in : ", grandparent_path)

# List of files which we want to move
f = open(r"C:\Users\user\Documents\workspace\minitorch-module-1-volyachka\files_to_sync.txt", "r+")
files_to_move = f.read().splitlines()
f.close()

# get the source and destination from arguments
source = sys.argv[1]
dest = sys.argv[2]

# copy the files from source to destination
try:
    for file in files_to_move:
        print(f"Moving file : ", file)

        # print(os.path.join(source, file))
        # print(os.path.join(dest, file))
        # shutil.copy(
        #     os.path.join(source, file),
        #     os.path.join(dest, file),
        # )
    print(f"Finished moving {len(files_to_move)} files")
except Exception as e:
    print(
        "Something went wrong! please check if the source and destination folders are present in same folder"
    )
