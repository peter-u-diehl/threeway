import os
import glob
import fnmatch

matches = []
for root, dirnames, filenames in os.walk('.'):
  for filename in fnmatch.filter(filenames, '*.npy'):
    matches.append(os.path.join(root, filename))

for filename in matches:
    if filename[-5] != 'e' and filename[-5] != 'i':
        print filename
        os.remove(filename)
