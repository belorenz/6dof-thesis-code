#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import glob
import json
import os
import sys
import time
import argparse
import logging
import random
import yaml

from PIL import Image, ImageDraw


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from cuboids import CuboidsCreator
from sync import CarlaSyncMode

VIEW_WIDTH = 640
VIEW_HEIGHT = 480
VIEW_FOV = 90


class DatasetCreator(object):

    def __init__(self):
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '-d', '--dataset_dir',
            metavar='d',
            default='/home/benjamin/test',
            type=str,
            help='directory to store data in')
        argparser.add_argument(
            '-n', '--number-of-vehicles',
            metavar='N',
            default=10,
            type=int,
            help='number of vehicles (default: 10)')
        argparser.add_argument(
            '-w', '--number-of-walkers',
            metavar='W',
            default=80,
            type=int,
            help='number of walkers (default: 50)')
        argparser.add_argument(
            '--safe',
            action='store_true',
            help='avoid spawning vehicles prone to accidents')
        argparser.add_argument(
            '--filterv',
            metavar='PATTERN',
            default='vehicle.*',
            help='vehicles filter (default: "vehicle.*")')
        argparser.add_argument(
            '--filterw',
            metavar='PATTERN',
            default='walker.pedestrian.*',
            help='pedestrians filter (default: "walker.pedestrian.*")')
        argparser.add_argument(
            '--tm-port',
            metavar='P',
            default=8000,
            type=int,
            help='port to communicate with TM (default: 8000)')
        argparser.add_argument(
            '--sync',
            action='store_true',
            help='Synchronous mode execution')
        argparser.add_argument(
            '--hybrid',
            action='store_true',
            help='Enanble')
        argparser.add_argument(
            '--yaml_path',
            default='/home/benjamin/test/places',
            help='Path to yaml file containing camera positions')
        argparser.add_argument(
            '-i', '--images_to_be_taken',
            default=60000,
            type=int,
            help='Number of all photorealistic images needed.'
        )
        argparser.add_argument(
            '--map',
            help='Current Map'
        )

        self.args = argparser.parse_args()
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        self.synchronous_master = False
        self.work_done = False
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.all_cams = []

        self.image_dir = self.args.dataset_dir
        self.current_pitch = None
        self.current_height = None
        self.current_yaw = None

        self.transform_to_cam_dict = {}

        self.image_id = 0

        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(5)
        self.world = self.client.get_world()
        assert self.args.map == self.world.get_map().name, 'map is {}, but should be {}'.format(self.world.get_map().name, self.args.map)
        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)

        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        if self.args.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)

        if self.args.sync:
            self.traffic_manager.set_synchronous_mode(True)
            self.settings = self.world.get_settings()
            if not self.settings.synchronous_mode:
                self.synchronous_master = True
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(self.settings)
            else:
                self.synchronous_master = False

    def spawn_actors(self):
        print('Spawning Cars and Pedestrians.')
        sys.stdout.flush()

        #################################################################
        # more or less untouched template code from  carla examples
        #################################################################
        self.blueprints = self.world.get_blueprint_library().filter(self.args.filterv)
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('cybertruck')]
        #self.blueprints = [x for x in self.blueprints if not x.id.endswith('carlacola')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('isetta')]
        self.blueprints = [x for x in self.blueprints if not x.id.endswith('police')]        
        assert len(self.blueprints) > 0
        self.blueprintsWalkers = self.world.get_blueprint_library().filter(self.args.filterw)

        if self.args.safe:
            self.blueprints = [x for x in self.blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('isetta')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('carlacola')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('cybertruck')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('t2')]


        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if self.args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.args.number_of_vehicles, number_of_spawn_points)
            self.args.number_of_vehicles = number_of_spawn_points

        #tudu cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        number_of_vehicles = int(len(spawn_points)/2)
        #number_of_vehicles = 50
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle:
                vehicle.set_autopilot()
                self.vehicles_list.append(vehicle.id)
                #print('Spawned car no: {} with id: {}'.format(n, vehicle.id))
                #sys.stdout.flush()
            else:
                number_of_vehicles += 1

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 2.0      # how many pedestrians will run
        percentagePedestriansCrossing = 2.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(self.blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure self.client receives the last transform of the walkers we have just created
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        self.number_of_vehicles = len(self.vehicles_list)
        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))
        sys.stdout.flush()
        # example of how to use parameters
        self.traffic_manager.global_percentage_speed_difference(30.0)

    def run(self):


        try:
            #################################################################
            # own code and configuration
            #################################################################

            self.setup_cam()
            self.place_cams()
            sys.stdout.flush()
            self.spawn_actors()
            self.create_transform_to_cam_dict()


            weathers = [ 'ClearNoon', 'CloudyNoon',
                         'SoftRainNoon',  'WetCloudyNoon',
                         'WetNoon', 'MidRainyNoon' ]

            # todo add to text why I exluced those
            excluded = ['HardRainNoon', 'HardRainSunset','MidRainSunset', 'WetSunset', 'ClearSunset',
                        'CloudySunset', 'SoftRainSunset', 'WetCloudySunset', 'Default']

            number_of_maps = 8
            images_per_map = self.args.images_to_be_taken / number_of_maps


            # Create a synchronous mode context
            with CarlaSyncMode(self.world, *self.all_cams) as sync_mode:

                _ = sync_mode.tick(timeout=2)

                while self.image_id <= images_per_map:
                    for weather in weathers:
                        if self.image_id > images_per_map:
                            break

                        self.world.set_weather(getattr(carla.WeatherParameters, weather))

                        print("Images saved: {}\nLet some time pass!\n".format(self.image_id))
                        sys.stdout.flush()
                        for i in range(0,10):
                            self.world.tick()

                        images = sync_mode.tick(timeout=2)
                        all_vehicles = self.world.get_actors().filter('vehicle.*')
                        assert self.number_of_vehicles == len(all_vehicles), \
                            'stuck actor: {} cars to beginning, currently active cars {}'.format(
                                self.number_of_vehicles, len(all_vehicles))
                        vehicles_to_capture = []
                        for vehicle in all_vehicles:
                            if not vehicle.type_id.endswith('omafiets') and not vehicle.type_id.endswith('crossbike') and not \
                                    vehicle.type_id.endswith('low_rider') and not vehicle.type_id.endswith('ninja') and not \
                                    vehicle.type_id.endswith('yzf') and not vehicle.type_id.endswith('century') and not \
                                    vehicle.type_id.endswith('carlacola'):
                                vehicles_to_capture.append(vehicle)

                        for cam in self.all_cams:
                            # get bounding boxes of cars
                            cam.objects, cam.bounding_boxes = CuboidsCreator.get_objects(vehicles_to_capture, cam, cam.max_distance)

                            #cam.bounding_boxes = CuboidsCreator.cuboids_in_image(all_cuboids)
                        self.save_data(images)
            self.work_done = True


        finally:
            if self.args.sync and self.synchronous_master:
                self.settings = self.world.get_settings()
                self.settings.synchronous_mode = False
                self.settings.fixed_delta_seconds = None
                self.traffic_manager.set_synchronous_mode(False)
                self.world.apply_settings(self.settings)

            for cam in self.all_cams:
                cam.stop()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_cams])
            self.all_cams = []

            print('\ndestroying %d vehicles' % len(self.vehicles_list))
            sys.stdout.flush()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

            for actor in self.all_actors:
                if actor.type_id == 'controller.ai.walker':
                    actor.stop()


            print('\ndestroying %d walkers' % len(self.walkers_list))
            sys.stdout.flush()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

            # segmentation fault workaround
            if self.work_done:
                exit_code = '0'
            else:
                exit_code = '1'

            with open(r'{}/exit_code'.format(self.args.yaml_path), "w") as text_file:
                print(exit_code, file=text_file)

            if not self.work_done:
                raise RuntimeError('Job is not finished. Something went wrong!')


    def place_cams(self):

        # load cam positions from yaml file
        filename = r'{}/{}.yaml'.format(self.args.yaml_path, self.world.get_map().name)
        assert os.path.exists(filename), 'No yaml file for current map.'
        with open(filename) as file:
            self.cam_configs = yaml.load(file, Loader=yaml.FullLoader)

        assert len(self.cam_configs) > 0

        # place all cams
        last_transform = None
        self.client.set_timeout(60)
        for id, data in self.cam_configs.items():
            transform = carla.Transform()
            transform.location.x = data.get('x')
            transform.location.y = data.get('y')
            transform.location.z = data.get('z')
            transform.rotation.yaw = data.get('yaw')
            transform.rotation.pitch = data.get('pitch')
            transform.rotation.roll = data.get('roll')

            if last_transform is not None:
                if self.is_equal_transform(transform,last_transform, 0.0001):
                    print('ID:{} is a duplicate of ID {}. Skipping this one!'.format(id, id-1))
                    sys.stdout.flush()
                    continue

            #spawn camera
            cam = self.world.spawn_actor(self.cam_blueprint, transform,)
            cam.max_distance = data.get('max_distance')
            cam.transform_check = transform

            # set calibration
            calibration = np.identity(3)
            calibration[0, 2] = VIEW_WIDTH / 2.0
            calibration[1, 2] = VIEW_HEIGHT / 2.0
            calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
            cam.calibration = calibration

            self.all_cams.append(cam)
            print('created cam n°:'+ str(id) + ' of type ' + str(cam.type_id) + ' with id ' + str(cam.id) + ' in ' + str(transform))
            sys.stdout.flush()
            last_transform = transform
        assert len(self.all_cams) > 0

    def create_transform_to_cam_dict(self):
        """
        I need to associate each image with a cam  to calculate the cuboids but
        there happens some kind of rounding error
        to make sure we can find the correct cam to the image later
        i make sure that the deviation of the cam position is not too big
        """

        for cam in self.all_cams:

            assert self.is_equal_transform(cam.transform_check, cam.get_transform(), 0.0001), \
                'Transform is not equal:\n{}\nvs.\n{}'.format(cam.transform_check, cam.get_transform())

            # add cam to cam dict (referenced with a string of the transform)
            self.transform_to_cam_dict.update({str(cam.get_transform()): cam})

        assert len(self.transform_to_cam_dict) > 0

    def setup_cam(self):
        self.cam_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.cam_blueprint.set_attribute('image_size_x', '640')
        self.cam_blueprint.set_attribute('image_size_y', '480')
        self.cam_blueprint.set_attribute('sensor_tick', '0')
        self.cam_blueprint.set_attribute('enable_postprocess_effects', 'True')

        # photo settings
        #self.cam_blueprint.set_attribute('fstop', '22')
        #self.cam_blueprint.set_attribute('iso', '256000')
        #self.cam_blueprint.set_attribute('shutter_speed', '256000')
        #self.cam_blueprint.set_attribute('focal_distance', '1000') # default value

        # intrinsics
        #self.cam_blueprint.set_attribute('lens_circle_falloff', '5.0') # vignettierung
        #self.cam_blueprint.set_attribute('lens_circle_multiplier', '0') # crop factor
        #self.cam_blueprint.set_attribute('lens_kcube', '0')

        # postprocessing settings
        #self.cam_blueprint.set_attribute('blur_amount', '0')
        #self.cam_blueprint.set_attribute('motion_blur_intensity', '0')
        #self.cam_blueprint.set_attribute('motion_blur_max_distortion', '0')


    def save_data(self,data):
        if data:
            images_to_save = {}
            cams_with_no_bounding_box = []
            annotations_to_save = {}
            draw_dict = {}

            for image in data:
                if type(image) is carla.Image:

                    cam = self.transform_to_cam_dict.get(str(image.transform))

                    if not cam:
                        print('Cam not found, deleting image..')
                        sys.stdout.flush()
                        os.remove(filename)
                        continue

                    # save image only if cars are on it
                    if cam.bounding_boxes:
                        # save image
                        filename = os.path.join(self.image_dir,'{:06d}.png'.format(
                            self.image_id,))
                        images_to_save.update({filename: image})
                        annotations_to_save.update(self.create_annotation_file(cam))
                        draw_dict.update({filename: cam})
                        self.image_id += 1
                    else:
                        cams_with_no_bounding_box.append(cam.id)
                        continue

            # save image and annotation files to disk
            saved = []
            for filename, image in images_to_save.items():
                image.save_to_disk(path=filename)
                #self.draw_cuboids(cam=draw_dict.get(filename), filename=filename)
                saved.append(filename)

            for filename, output in annotations_to_save.items():
                with open(filename, "w") as write_file:
                    json.dump(output, write_file, indent=4)

            # Terminal output!
            print('saved: ' + ''.join(' {}\n'.format(filename) for filename in saved))
            print('skipped: ' + ''.join(' {}'.format(cam_id) for cam_id in cams_with_no_bounding_box))
            sys.stdout.flush()

    def create_annotation_file(self, cam):

        transform = cam.get_transform()
        x, y, z, w = CuboidsCreator.euler_to_quaternion(transform.rotation)

        # check if the calculation is correct
        yaw, pitch, roll = CuboidsCreator.quaternion_to_euler(x, y, z, w)
        assert abs(yaw - transform.rotation.yaw) < 0.0001, 'calucalted yaw:{} != initially:{}'.format(yaw, transform.rotation.yaw)
        assert abs(pitch - transform.rotation.pitch) < 0.0001, 'calucalted yaw:{} != initially:{}'.format(pitch, transform.rotation.pitch)
        assert abs(roll - transform.rotation.roll) < 0.0001,'calucalted yaw:{} != initially:{}'.format(roll, transform.rotation.roll)

        output = { 'camera_data' : {'location_worldframe' :
                                        [transform.location.x,transform.location.y,transform.location.z ],
                                    'quaternion_xyzw_worldframe' : [x, y, z, w] },
                   'objects' : cam.objects}

        filename=os.path.join(self.image_dir,'{:06d}.json'.format(
            self.image_id,))

        return {filename: output}


    def draw_cuboids(self,cam, filename):
        # get the cam which took the image

        im = Image.open(filename)
        d = ImageDraw.Draw(im)
        for detection in cam.objects:
            projected_cuboid = detection.get('projected_cuboid')
            points = [(int(point[0]), int(point[1])) for point in projected_cuboid]

            # geometric shapes to draw
            front = [points[0], points[3], points[7], points[4], points[0] ]
            back = [points[1], points[2], points[6], points[5], points[1] ]
            vert1 = [points[4], points[5]]
            vert2 = [points[7], points[6]]
            vert3 = [points[0], points[1]]
            vert4 = [points[3], points[2]]
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

    def is_equal_transform(self, t1, t2, deviation):
        return abs(t1.location.x - t2.location.x) <= deviation and \
            abs(t1.location.y - t2.location.y) <= deviation and \
            abs(t1.location.z - t2.location.z) <= deviation and \
            abs(t1.rotation.yaw - t2.rotation.yaw) <= deviation and \
            abs(t1.rotation.roll - t2.rotation.roll) <= deviation and \
            abs(t1.rotation.pitch - t2.rotation.pitch) <= deviation

if __name__ == '__main__':
    DatasetCreator().run()
    print('\ndone.')
    time.sleep(2)
