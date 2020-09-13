#!/usr/bin/env python3
import json
import os
from glob import glob


class ClassCorrector:

    def __init__(self):
        self.dataset_base_dir = '/media/benjamin/DOPE/dataset_backup/randomized'
        self.filename_list = self.create_filename_list()
        self.dataset_size = len(self.filename_list)

    def run(self):

        for i, filename in enumerate(self.filename_list):
            print('{}/{}: Updating {}'.format(i, self.dataset_size,  filename))
            data = None
            with open(filename, 'r') as current_file:
                data = json.load(current_file)
                for item in data['objects']:
                    item.update({"class": "car"})
            with open(filename, 'w') as current_file:
                json.dump(data, current_file, indent=4)

    def create_filename_list(self):
        dir_list = [root for root,dirs,_ in os.walk(self.dataset_base_dir)]
        filename_list = []
        for image_dir in dir_list:
            filename_list.extend(glob(image_dir + "/*.json"))
        return filename_list

if __name__ == '__main__':
    ClassCorrector().run()
    print('\ndone.')

