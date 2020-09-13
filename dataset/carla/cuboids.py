#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 640
VIEW_HEIGHT = 480
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

# ==============================================================================
# -- CuboidsCreator ---------------------------------------------------
# ==============================================================================


class CuboidsCreator(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """


    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            color = BB_COLOR
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, color, points[0], points[1])
            pygame.draw.line(bb_surface, color, points[0], points[1])
            pygame.draw.line(bb_surface, color, points[1], points[2])
            pygame.draw.line(bb_surface, color, points[2], points[3])
            pygame.draw.line(bb_surface, color, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, color, points[4], points[5])
            pygame.draw.line(bb_surface, color, points[5], points[6])
            pygame.draw.line(bb_surface, color, points[6], points[7])
            pygame.draw.line(bb_surface, color, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, color, points[0], points[4])
            pygame.draw.line(bb_surface, color, points[1], points[5])
            pygame.draw.line(bb_surface, color, points[2], points[6])
            pygame.draw.line(bb_surface, color, points[3], points[7])
        display.blit(bb_surface, (0, 0))


    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = CuboidsCreator._create_bb_points(vehicle)
        cords_x_y_z, world_cords = CuboidsCreator._vehicle_to_sensor(bb_cords, vehicle, camera)
        cords_x_y_z = cords_x_y_z[:3, :]
        world_cords = np.transpose(world_cords[:3, :])


        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox, world_cords

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """
        world_cord = CuboidsCreator._vehicle_to_world(cords, vehicle)
        sensor_cord = CuboidsCreator._world_to_sensor(world_cord, sensor)
        return sensor_cord, world_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = CuboidsCreator.get_matrix(bb_transform)
        vehicle_world_matrix = CuboidsCreator.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = CuboidsCreator.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    ##########################################################################################
    ##########################              my code             ##############################
    ##########################################################################################

    @staticmethod
    def get_objects(vehicles, camera, max_distance):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        # cuboids = [CuboidsCreator.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        objects = []
        cuboids = []
        for vehicle in vehicles:
            cuboid_as_matrix, cuboid_in_world = CuboidsCreator.get_bounding_box(vehicle, camera)


            # must be in front of camera and not too far away
            if all(cuboid_as_matrix[:, 2] > 0) and all(cuboid_as_matrix[:, 2] < max_distance):

                # add vehicle if there is a point of the cuboid in the image
                for point in [(int(cuboid_as_matrix[i, 0]), int(cuboid_as_matrix[i, 1])) for i in range(8)]:
                    if CuboidsCreator.point_is_in_image(point):
                        vehicle_transform = vehicle.get_transform()
                        pose_transform_permuted = [] # todo add pose_transform_permuted


                        # carla default distance is m! m --> cm = m * 100
                        cuboid_in_world_in_cm = (cuboid_in_world * 100).tolist()

                        # order of the points as defined in FAT (see FAT dataset overview)
                        cuboid_in_FAT_definition = \
                            CuboidsCreator.rearrange_points(cuboid_in_world_in_cm)

                        cuboid_centroid = [vehicle_transform.location.x * 100,
                                           vehicle_transform.location.y * 100,
                                           vehicle_transform.location.z * 100]

                        projected_cuboid_in_wrong_order = [[cuboid_as_matrix[i,
                                                                           0], cuboid_as_matrix[i, 1]] for i in range(8)]



                        # add all vectors in np_array and divide it by the number of points (8) to get the centeroid
                        # calculate center of cuboid in image space and keep first two elements (x,y) in list, leave out z
                        cuboid_as_np_array = [ np.array(cuboid_as_matrix)[i] for i in range(8)]
                        projected_cuboid_centroid = (sum(cuboid_as_np_array)/len(cuboid_as_np_array)).tolist()[:2]
                        bounding_box = {} # todo

                        detection = {'class': 'car',
                                      'visibility' : 1 ,
                                      'location': [vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.location.z],
                                      'quaternion_xyzw': CuboidsCreator.euler_to_quaternion(vehicle_transform.rotation),
                                      'pose_transform_permuted': [],
                                      'cuboid_centroid': cuboid_centroid,
                                      'projected_cuboid_centroid': projected_cuboid_centroid ,
                                      'bounding_box': {'top_left': [], 'bottom_right': []} , # i leave that out for now
                                      'cuboid': cuboid_in_FAT_definition,
                                      'projected_cuboid': CuboidsCreator.rearrange_points(projected_cuboid_in_wrong_order),
                                      }

                        cuboids.append(detection.get('projected_cuboid'))
                        objects.append(detection)
                        break

        return objects, cuboids

    @staticmethod
    def rearrange_points(points):
        return [points[4],
                points[7],
                points[3],
                points[0],
                points[5],
                points[6],
                points[2],
                points[1]]

    @staticmethod
    def point_is_in_image(point):
        return point[0] >= 0 and point[0] <= VIEW_WIDTH and point[1] >= 0 and point[1] <= VIEW_HEIGHT

    @staticmethod
    def cuboids_in_image(cuboids):
        """
        Reduces set of bounding boxes to the ones that are visible in the image.
        """
        cuboids_in_image = []
        for cuboid in cuboids:
            for point in [(int(cuboid[i, 0]), int(cuboid[i, 1])) for i in range(8)]:
                if CuboidsCreator.point_is_in_image(point):
                    cuboids_in_image.append(cuboid)
                    break
        return cuboids_in_image

    @staticmethod
    def euler_to_quaternion(rotation):
        '''
        :param rotation: transform rotation object [yaw, pitch, roll] in degrees
        :return: quaternion representation of the input angle
        '''
        # rotation is given in degrees --> calculate radians
        roll = float(rotation.roll) * (np.pi/180.0)
        yaw = float(rotation.yaw) * (np.pi/180.0)
        pitch = float(rotation.pitch) * (np.pi/180.0)

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    @staticmethod
    def quaternion_to_euler(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.math.atan2(t3, t4)

        # from rad to degree
        yaw = (yaw * 180.0) / np.pi
        roll = (roll * 180.0) / np.pi
        pitch = (pitch * 180.0) / np.pi
        return [yaw, pitch, roll]

