import sys, os
import random, shutil

#flatten directory structure inside the given folder, giving files random names (to prevent multiple files with the same name)

directory = sys.argv[1]

print(f"Unfucking directory structure for folder: {sys.argv[1]}")

for (dirpath, dirnames, filenames) in os.walk(directory):
    for filename in filenames:
        path = dirpath + "/" + filename
        newpath = directory + "/" + str(random.randint(1, 99999)) + ".jpg"
        print(f"Moving {path} to {newpath}")
        shutil.move(path, newpath)
