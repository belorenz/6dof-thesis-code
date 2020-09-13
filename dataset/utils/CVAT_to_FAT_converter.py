#!/usr/bin/env python3
import json
import os
import time
import shutil
from PIL import Image, ImageDraw, ImageFont
from glob import glob
import random
from xml.dom import minidom
import pygame as pygame


class CVATConverter:

    def __init__(self):
        self.xml_dir = '/home/benjamin/Documents/Studium/Informatik/abschlussarbeit/dataset/sample/cvat/shanghai_2'
        self.image_dir = '/home/benjamin/Downloads/datensatz/MultiCarPose_v.0.1/test/final/shanghai_+ann'
        self.xml_filename_list = self.create_filename_list()
        assert len(self.xml_filename_list) > 0

    def run(self):
        for xml_filename in self.xml_filename_list:
            xml = self.read_xml(xml_filename)
            self.convert(xml)

    def read_xml(self,filename):
        with open(filename, 'r') as xml_file:
            xml = minidom.parse(filename)
        return xml

    def convert(self,xml):
        images = xml.getElementsByTagName('image')
        for image_tag in images:
            image_name = image_tag.attributes['name'].value
            image_path = self.image_dir + '/' + image_name
            assert os.path.isfile(image_path)

            objects = []
            for cuboid_tag in image_tag.getElementsByTagName('cuboid'):
                cuboid_points = [
                    [float(cuboid_tag.attributes['xtl1'].value),
                     float(cuboid_tag.attributes['ytl1'].value)],
                    [float(cuboid_tag.attributes['xbl1'].value),
                     float(cuboid_tag.attributes['ybl1'].value)],
                    [float(cuboid_tag.attributes['xtr1'].value),
                     float(cuboid_tag.attributes['ytr1'].value)],
                    [float(cuboid_tag.attributes['xbr1'].value),
                     float(cuboid_tag.attributes['ybr1'].value)],
                    [float(cuboid_tag.attributes['xtl2'].value),
                     float(cuboid_tag.attributes['ytl2'].value)],
                    [float(cuboid_tag.attributes['xbl2'].value),
                     float(cuboid_tag.attributes['ybl2'].value)],
                    [float(cuboid_tag.attributes['xtr2'].value),
                     float(cuboid_tag.attributes['ytr2'].value)],
                    [float(cuboid_tag.attributes['xbr2'].value),
                     float(cuboid_tag.attributes['ybr2'].value)],
                ]
                frontface = cuboid_tag.getElementsByTagName(
                    'attribute')[0].lastChild.data

                ordered_points = self.reorder_cuboid_points(points=
                                                            cuboid_points,
                                                            frontface=frontface)

                object = {'projected_cuboid': ordered_points, 'class': 'car'}
                objects.append(object)

            #self.draw_cuboid_points(objects, image_path)
            self.create_annotation_file(objects, image_path)

    def create_annotation_file(self, objects_dict, image_path):
        output = { 'camera_data' : {'location_worldframe' : [0, 0, 0 ],
                                    'quaternion_xyzw_worldframe' : [0, 0, 0, 1] },
                   'objects' : objects_dict}

        filename = image_path[:-3]+'json'

        with open(filename, "w") as write_file:
            json.dump(output, write_file, indent=4)


    def reorder_cuboid_points(self,points, frontface):
        ordered_points = []
        if frontface == 'right':
            ordered_points = [
                points[2],
                points[4],
                points[5],
                points[3],
                points[0],
                points[6],
                points[7],
                points[1],
            ]
        elif frontface == 'left':
            ordered_points = [
                points[6],
                points[0],
                points[1],
                points[7],
                points[4],
                points[2],
                points[3],
                points[5],
            ]
        elif frontface == 'front':
            ordered_points = [
                points[0],
                points[2],
                points[3],
                points[1],
                points[6],
                points[4],
                points[5],
                points[7],
            ]
        elif frontface == 'back':
            ordered_points = [
                points[4],
                points[6],
                points[7],
                points[5],
                points[2],
                points[0],
                points[1],
                points[3],
            ]
        else:
            raise RuntimeError, 'unknown frontface: {}'.format(
                frontface)
        return ordered_points

    def draw_cuboid_points(self, detections, filename):
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        with Image.open(filename) as im:
            d = ImageDraw.Draw(im)
            for detection in detections:
                projected_cuboid = detection.get('projected_cuboid')
                points = [(int(point[0]), int(point[1])) for point in projected_cuboid]

                # geometric shapes to draw

                #front = [points[0], points[3], points[7], points[4], points[0] ]
                #back = [points[1], points[2], points[6], points[5], points[1] ]
                #vert1 = [points[4], points[5]]
                #vert2 = [points[7], points[6]]
                #vert3 = [points[0], points[1]]
                #vert4 = [points[3], points[2]]

                # draw_shapes
                #d.line(front, fill=(255, 0, 0), width=1)
                #d.line(back, fill=(0, 0, 255), width=1)
                #d.line(vert1, fill=(0, 0, 255), width=1)
                #d.line(vert2, fill=(0, 0, 255), width=1)
                #d.line(vert3, fill=(0, 0, 255), width=1)
                #d.line(vert4, fill=(0, 0, 255), width=1)


                d.text(xy=points[0], text="0", font=fnt, fill=(0, 0, 255))
                d.point(points[0], fill=(0, 0, 255))
                d.text(xy=points[1], text="1", font=fnt, fill=(0, 0, 255))
                d.point(points[1], fill=(0, 0, 255))
                d.text(xy=points[2], text="2", font=fnt, fill=(0, 0, 255))
                d.point(points[2], fill=(0, 0, 255))
                d.text(xy=points[3], text="3", font=fnt, fill=(0, 0, 255))
                d.point(points[3], fill=(0, 0, 255))
                d.text(xy=points[4], text="4", font=fnt, fill=(0, 0, 255))
                d.point(points[4], fill=(0, 0, 255))
                d.text(xy=points[5], text="5", font=fnt, fill=(0, 0, 255))
                d.point(points[5], fill=(0, 0, 255))
                d.text(xy=points[6], text="6", font=fnt, fill=(0, 0, 255))
                d.point(points[6], fill=(0, 0, 255))
                d.text(xy=points[7], text="7", font=fnt, fill=(0, 0, 255))
                d.point(points[7], fill=(0, 0, 255))

            im.save(filename)

    def create_filename_list(self):
        dir_list = [root for root,dirs,_ in os.walk(self.xml_dir)]
        filename_list = []
        for annotation in dir_list:
            filename_list.extend(glob(annotation + "/*.xml"))
        return filename_list




if __name__ == '__main__':
    CVATConverter().run()
    print('\ndone.')

