from scipy.ndimage import imread
import extraction
import glob
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

features_std = np.zeros((3000, 12))
features_prop = np.zeros((3000, 12))
labels = np.zeros((3000, 10))
index = 0
print('começando')
for digit in range(10):
    lbl = np.zeros(10)
    lbl[digit] = 1
    labels[digit * 300: (digit * 300 + 300)] = lbl
    for file in glob.iglob('imagens/%s/*.bmp' % digit):
        print('index = %d' % index, end='\r', flush=True)
        image = imread(file, flatten=True)
        features_std[index] = np.reshape(extraction.stdev(image), -1)
        features_prop[index] = np.reshape(extraction.proportion(image), -1)
        index += 1

np.save('label.npy', labels)
np.save('feature_std.npy', features_std)
np.save('feature_prop.npy', features_prop)

indices = np.array([0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700])

print(labels[indices])
print(features_std[indices])
print(features_prop[indices])
