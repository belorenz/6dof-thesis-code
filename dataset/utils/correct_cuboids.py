#!/usr/bin/env python3
import json
import os
from PIL import Image, ImageDraw
from glob import glob


class CuboidCorrector:

    def __init__(self):
        self.dataset_base_dir = '/media/benjamin/Liesbeth/MultiCarPose_v0.9'
        self.filename_list = self.create_filename_list()
        self.dataset_size = len(self.filename_list)

    def run(self):

        for i, filename in enumerate(self.filename_list):
            if filename.endswith('settings.json'):
                continue
            print('{}/{}: Updating {}'.format(i, self.dataset_size,  filename))
            data = None
            with open(filename, 'r') as current_file:
                data = json.load(current_file)

            first_object = data['objects'][0]
            if 'pose_transform' in first_object.keys():
                continue

            for item in data['objects']:
                item.update({"projected_cuboid": self.rearrange_points(item["projected_cuboid"])})
                #item.update({"cuboid": self.rearrange_points(item["cuboid"])})
            #filename_image = filename[:-4] + "png"
            #self.draw_cuboids(filename_image, data['objects'])
            with open(filename, 'w') as current_file:
                json.dump(data, current_file, indent=4)

    def rearrange_points(self, points):
        return [
            points[4],
            points[7],
            points[3],
            points[0],
            points[5],
            points[6],
            points[2],
            points[1]
        ]

    def draw_cuboids(self, filename, objects):

        im = Image.open(filename)
        d = ImageDraw.Draw(im)
        for detection in objects:
            projected_cuboid = detection.get('projected_cuboid')
            points = [(int(point[0]), int(point[1])) for point in projected_cuboid]

            # geometric shapes to draw
            front = [points[0], points[1], points[2], points[3], points[0] ]
            back = [points[4], points[5], points[6], points[7], points[4] ]
            vert1 = [points[0], points[4]]
            vert2 = [points[1], points[5]]
            vert3 = [points[2], points[6]]
            vert4 = [points[4], points[7]]
            centeroid = detection.get('projected_cuboid_centroid')

            # draw_shapes
            d.line(front, fill=(255, 0, 0), width=1)
            d.line(back, fill=(0, 0, 255), width=1)
            d.line(vert1, fill=(0, 0, 255), width=1)
            d.line(vert2, fill=(0, 0, 255), width=1)
            d.line(vert3, fill=(0, 0, 255), width=1)
            d.line(vert4, fill=(0, 0, 255), width=1)
            d.point(centeroid, fill=(0, 0, 255))

            d.text(xy=points[0], text="0", fill=(0, 0, 255))
            d.text(xy=points[1], text="1", fill=(0, 0, 255))
            d.text(xy=points[2], text="2", fill=(0, 0, 255))
            d.text(xy=points[3], text="3", fill=(0, 0, 255))
            d.text(xy=points[4], text="4", fill=(0, 0, 255))
            d.text(xy=points[5], text="5", fill=(0, 0, 255))
            d.text(xy=points[6], text="6", fill=(0, 0, 255))
            d.text(xy=points[7], text="7", fill=(0, 0, 255))

        im.save(filename)


    def create_filename_list(self):
        dir_list = [root for root,dirs,_ in os.walk(self.dataset_base_dir)]
        filename_list = []
        for image_dir in dir_list:
            filename_list.extend(glob(image_dir + "/*.json"))
        return filename_list

if __name__ == '__main__':
    CuboidCorrector().run()
    print('\ndone.')

