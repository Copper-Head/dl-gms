import glob
import json
import os
import configparser
from functools import partial
from keras.preprocessing.image import load_img, img_to_array
import h5py

config = configparser.ConfigParser()
config.read('config.ini')
data_path = partial(os.path.join, config['paths']['data_dir'])

all_paths = glob.glob(data_path("*.png"))
with open(data_path('noisy_real.json'), 'w') as f:
    json.dump(all_paths, f)

with h5py.File(data_path('noisy_real.h5'), 'w') as f:

    images = f.create_dataset('images', (len(all_paths), 64, 64, 3))

    for index, path in enumerate(all_paths):
        images[index] = img_to_array(load_img(path))
