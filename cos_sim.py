import numpy as np
import sys



a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
nbex = int(sys.argv[3])
with open(sys.argv[4]) as f:
    names_a = f.readlines()
with open(sys.argv[5]) as f:
    names_b = f.readlines()

dim = a.shape[0] // nbex 
a.resize(nbex, dim)
b.resize(nbex, dim)
#print(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))

for i in range(nbex):
    for j in range(nbex):
        cossim = np.sum(a[i] * b[j]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[j]))
        print('%s %s: %f' % (names_a[i].strip(), names_b[j].strip(), cossim))
