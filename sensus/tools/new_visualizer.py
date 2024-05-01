from .data_processor import SensusInstance

import torch
import numpy as np
import cv2
import open3d as o3d
from open3d.web_visualizer import draw

from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
from mmdet3d.structures import points_cam2img

class LidarVisualizer:
    def __init__(self, pcd, calib, bboxes):
        self.pcd = pcd
        self.calib = calib
        self.bboxes = bboxes

    def create_box(self, dimensions):
        box = o3d.geometry.TriangleMesh.create_box(width=dimensions[0], height=dimensions[1], depth=dimensions[2])
        box.paint_uniform_color([1.0, 0.0, 0.0])
        return box

    def translate_box(self, box, position, dimensions):
        box.translate([position[0], position[1], position[2]])
        # box.translate([-dimensions[2]/2, -dimensions[1]/2, 0])
        box.translate([-dimensions[0]/2, -dimensions[1]/2, 0])
        return box

    def rotate_box(self, box, rotation):
        center = box.get_center()
        rotation = box.get_rotation_matrix_from_xyz((0, 0, rotation[0]))
        box.rotate(rotation, center=center)
        return box

    def generate_bbox(self, bbox, bbox_color = [1, 0, 0]):
        dim = bbox.dimensions
        loc = bbox.location
        rot = bbox.rotation

        box = self.create_box(dim)
        box = self.translate_box(box, loc, dim)
        box = self.rotate_box(box, rot)

        lines = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 0], [2, 3], [3, 1], [4, 5], [4, 6], [6, 7], [7, 5], [0, 4], [1, 5], [2, 6], [3, 7]]))
        lines.paint_uniform_color(bbox_color)
        return lines
    
    def draw_bboxes_from_labels(self, num_cars):
        lines = []
        cars = 0
        for bbox in self.bboxes:
            if bbox.type == 'Car':
                cars += 1
                lines.append(self.generate_bbox(bbox))
            if cars >= num_cars:
                break
        draw([self.pcd, *lines], width=900, height=600, point_size=2)

    def draw_bboxes_from_result(self, score):
        lines = []
        for bbox in self.bboxes:
            if bbox.score >= score:
                lines.append(self.generate_bbox(bbox))
        draw([self.pcd, *lines], width=900, height=600, point_size=2)


class ImageVisualizer:
    def __init__(self, img, calib, bboxes):
        self.image = img
        self.calib = calib
        self.bboxes = bboxes

    def draw_3d_box(self, bbox, pitch, thickness=2):
        # Get dimensions, location and rotation_y from label
        l, h, w = bbox.dimensions
        x, y, z = bbox.location
        ry = bbox.rotation_y

        # Define 3D bounding box vertices in object's local coordinate system
        vertices = np.array([
            [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],
            [0, 0, 0, 0, -h, -h, -h, -h],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        ])

        # Rotation matrix around Y-axis in camera coordinates
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Rotation matrix around X-axis for pitch
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        # Rz = np.array([
        # [np.cos(roll), -np.sin(roll), 0],
        # [np.sin(roll), np.cos(roll), 0],
        # [0, 0, 1]
        # ])

        # Apply rotations    
        vertices = np.dot(Ry, vertices)
        vertices = np.dot(Rx, vertices)
        # vertices = np.dot(Rz, vertices)


        # Translate vertices to world coordinate
        vertices[0, :] = vertices[0, :] + x
        vertices[1, :] = vertices[1, :] + y
        vertices[2, :] = vertices[2, :] + z

        # Project to image plane
        P = np.array(self.calib['P2']).reshape(3, 4)
        vertices_2d = self.project_3d_to_2d(vertices.T, P)

        # Draw lines connecting the vertices
        for i in range(4):
            cv2.line(self.image, tuple(np.int32(vertices_2d[i])), tuple(np.int32(vertices_2d[(i+1)%4])), (255, 0, 0), thickness)
            cv2.line(self.image, tuple(np.int32(vertices_2d[i+4])), tuple(np.int32(vertices_2d[(i+1)%4+4])), (255, 0, 0), thickness)
            cv2.line(self.image, tuple(np.int32(vertices_2d[i])), tuple(np.int32(vertices_2d[i+4])), (255, 0, 0), thickness)

        # Draw 'X' on the front face of the car
        cv2.line(self.image, tuple(np.int32(vertices_2d[0])), tuple(np.int32(vertices_2d[5])), (0, 255, 0), thickness)
        cv2.line(self.image, tuple(np.int32(vertices_2d[1])), tuple(np.int32(vertices_2d[4])), (0, 255, 0), thickness)

    def get_pitch(self, transformation_matrix):
        """
        Extracts the Euler angles from a 4x4 transformation matrix.
        
        Args:
        - transformation_matrix (numpy.array): 4x4 transformation matrix

        Returns:
        - roll, pitch, yaw (floats): Euler angles in radians
        """
        # Converting the list to a 3x4 numpy array first
        matrix_3x4 = np.array(transformation_matrix).reshape(3, 4)

        # Appending the row [0, 0, 0, 1] at the bottom to make it a 4x4 matrix
        transformation_matrix = np.vstack([matrix_3x4, [0, 0, 0, 1]])
        assert transformation_matrix.shape == (4, 4), "Matrix must be 4x4"
        
        # Extract 3x3 rotation matrix from the 4x4 transformation matrix
        R = transformation_matrix[:3, :3]
        
        # Pitch
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2)) + np.pi/2
        return pitch

    def project_3d_to_2d(self, points_3d, P):
        points_3d_ext = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        points_2d = np.dot(P, points_3d_ext.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
        return points_2d
    
    def draw_monodetection_labels(self, num_cars, thickness, out_dir):
        cars = 1
        pitch = self.get_pitch(self.calib['Tr_velo_to_cam'])
        for bbox in self.bboxes:
            if bbox.type == 'Car':
                cars += 1
                self.draw_3d_box(bbox, pitch, thickness)

            if cars >= num_cars:
                break

        cv2.imwrite(out_dir + '.png', self.image)

    def draw_monodetection_results(self, score, thickness, out_dir, index_car, pitch=0.0, roll = 0.0):
        pitch = self.get_pitch(self.calib['Tr_velo_to_cam'])
        for bbox in self.bboxes:
            if bbox.score >= score and bbox.type == index_car:
                self.draw_3d_box(bbox, pitch, thickness)

        cv2.imwrite(out_dir + '.png', self.image)
        return self.image


def draw_monodetection_labels(img_file, calib_path, labels_path, num_cars, thickness=2, out_path = 'viz_img'):
    data = SensusInstance()
    data.load_image_from_path(img_file)
    data.load_calib(calib_path)
    data.load_bboxes_from_labels(labels_path, cam_bbox=True, lidar_bbox=False)    
    viz = ImageVisualizer(data.image, data.calib, data.camera_bboxes)
    viz.draw_monodetection_labels(num_cars = num_cars, thickness = thickness, out_dir=out_path)


def draw_monodetection_results(img_file, calib_path, results, score=0.25, thickness=2, out_path = 'viz_img', index_car = 0):
    data = SensusInstance()
    data.load_image_from_path(img_file)
    data.load_calib(calib_path)
    data.load_bboxes_from_result(results, cam_bbox=True, lidar_bbox=False)    
    viz = ImageVisualizer(data.image, data.calib, data.camera_bboxes)
    viz.draw_monodetection_results(score = score, thickness = thickness, out_dir=out_path, index_car = index_car)

def draw_lidar_labels(lidar_path, calib_path, labels_path, num_cars=10):
    data = SensusInstance()
    data.load_lidar_data(lidar_path)
    data.load_calib(calib_path)
    data.load_bboxes_from_labels(labels_path, cam_bbox=False, lidar_bbox=True)

    viz = LidarVisualizer(data.lidar_pcd, data.calib, data.lidar_bboxes)
    viz.draw_bboxes_from_labels(num_cars = num_cars)

    return data


def draw_lidar_results(lidar_path, calib_path, results, score=0.25):
    data = SensusInstance()
    data.load_lidar_data(lidar_path)
    data.load_calib(calib_path)
    data.load_bboxes_from_result(results, cam_bbox=False, lidar_bbox=True)

    viz = LidarVisualizer(data.lidar_pcd, data.calib, data.lidar_bboxes)
    viz.draw_bboxes_from_result(score=score)

    return data
    
