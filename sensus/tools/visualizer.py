import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
from open3d.web_visualizer import draw
import cv2
import numpy as np
from mmengine.config import Config
import os.path as osp
from mmdet3d.utils import register_all_modules
from mmdet3d.datasets import *


from data_processor import DataProcessor

class LidarVisualizer:
    def __init__(self, bin_path, img_path):
        """
        Initialize the LidarVisualizer class

        Parameters
        ----------
        bin_path : str
            Path to the .bin file.
        img_path : str
            Path to the image file.
        """

        self.bin_path = bin_path
        self.img_path = img_path
        self.load_point_cloud()

    def load_point_cloud(self):
        with open(self.bin_path, 'rb') as f:
            points = np.fromfile(f, dtype=np.float32, count=-1).reshape([-1, 4])

        self.pcd_bin = o3d.geometry.PointCloud()
        self.pcd_bin.points = o3d.utility.Vector3dVector(points[:, :3])

    def load_labels(self, labels_path):
        self.labels = DataProcessor(labels_path, None).process_label_file()

    def load_calib(self, calib_path):
        self.calib = DataProcessor(None, calib_path).process_calib_file()

    def load_results(self, results):
        self.results = results

    def get_lidar_in_cam_T(self):
        Tr_velo_to_cam = np.array(self.calib['Tr_velo_to_cam'])
        Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
        R = Tr_velo_to_cam[0:3, 0:3]
        T = Tr_velo_to_cam[0:3, 3]
        lidar_in_cam_T = np.eye(4)
        lidar_in_cam_T[0:3, 0:3] = R.T
        lidar_in_cam_T[0:3, 3] = -R.T @ T
        return lidar_in_cam_T
    
    def create_box(self, dimensions):
        box = o3d.geometry.TriangleMesh.create_box(width=dimensions[0], height=dimensions[1], depth=dimensions[2])
        box.paint_uniform_color([1.0, 0.0, 0.0])
        return box
    
    def convert_position_to_lidar(self, position):
        lidar_in_cam_T = self.get_lidar_in_cam_T()
        pos_in_cam = np.array(position + [1])
        pos_in_lidar = lidar_in_cam_T @ pos_in_cam
        return pos_in_lidar
    
    def translate_box(self, box, position, dimensions):
        box.translate([position[0], position[1], position[2]])
        # box.translate([-dimensions[2]/2, -dimensions[1]/2, 0])
        box.translate([-dimensions[0]/2, -dimensions[1]/2, 0])
        return box
    
    def rotate_box(self, box, rotation_y):
        center = box.get_center()
        # rotation = box.get_rotation_matrix_from_xyz((0, 0, -rotation_y - np.pi/2))
        rotation = box.get_rotation_matrix_from_xyz((0, 0, +rotation_y))
        box.rotate(rotation, center=center)
        return box
    
    def generate_bbox_from_label(self, label, color=[1, 0, 0]):
        """
        Generate bbox from label dict.

        Parameters
        ----------
        label : dict
            label dictionary.
        """
        dimensions = label['dimensions']
        position = label['location']
        rotation_y = label['rotation_y']
        
        dimensions[2], dimensions[0] = dimensions[0], dimensions[2]
        box = self.create_box(dimensions)
        position = self.convert_position_to_lidar(position)

        rotation_y = -rotation_y + np.pi/2

        box = self.translate_box(box, position, dimensions)
        box = self.rotate_box(box, rotation_y)
        
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 0], [2, 3], [3, 1], [4, 5], [4, 6], [6, 7], [7, 5], [0, 4], [1, 5], [2, 6], [3, 7]]))
        lines.paint_uniform_color(color)
        
        return lines
    
    def generate_bbox_from_result(self, result):
        """
        Generate bbox from results array.

        Parameters
        ----------
        result : array
            results array [position, dimensions, rotation_y]
        """
        dimensions = result[3:6]
        # dimensions = [dimensions[2], dimensions[1], dimensions[0]]
        position = result[0:3]
        rotation_y = result[6]

        box = self.create_box(dimensions)
        box = self.translate_box(box, position, dimensions)
        box = self.rotate_box(box, rotation_y)
        
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 0], [2, 3], [3, 1], [4, 5], [4, 6], [6, 7], [7, 5], [0, 4], [1, 5], [2, 6], [3, 7]]))
        lines.paint_uniform_color([1, 0, 0])
        
        return lines
    
    def generate_img(self, label):
        """
        Load and show img with label rectangles.

        Parameters
        ----------
        label: dict
            label dictionary
        """
        img = mpimg.imread(self.img_path)
        rect = plt.Rectangle((label['bbox'][0], label['bbox'][1]), label['bbox'][2] - label['bbox'][0], label['bbox'][3] - label['bbox'][1], linewidth=1, edgecolor='r', facecolor='none')
        fig, ax = plt.subplots(1)
        ax.add_patch(rect)
        plt.imshow(img)

    def draw_cars_from_labels(self, num_cars):
        """
        Draws num_cars from label file.

        Parameters
        ----------
        labels: str
            path to labels file
        calib: str
            path to calibration file
        num_cars: int
            number of cars to draw
        """

        cars = []
        for label in self.labels:
            if label['type'] == 'Car':
                cars.append(label)
            if len(cars) == num_cars:
                break

        lines = []
        for car in cars:
            lines.append(self.generate_bbox_from_label(car))
            self.generate_img(car)

        # Draw bbox into point cloud
        draw([self.pcd_bin, *lines], width=900, height=600, point_size=2)

    def draw_cars_from_monodet_result(self, results):
        """
        Draws num_cars from label file.

        Parameters
        ----------
        labels: str
            path to labels file
        calib: str
            path to calibration file
        num_cars: int
            number of cars to draw
        """

        cars_result = []
        for result in results:
            cars_result.append(result)

        lines = []
        for car in cars_result:
            lines.append(self.generate_bbox_from_label(car, color = [0, 1, 0]))

        cars_labels = []
        for label in self.labels:
            if label['type'] == 'Car':
                cars_labels.append(label)

        for car in cars_labels:
            lines.append(self.generate_bbox_from_label(car, color = [1, 0, 0]))

        # Draw bbox into point cloud
        draw([self.pcd_bin, *lines], width=900, height=600, point_size=2)

    def draw_one_car_from_label(self, label, calib):
        lines = []
        lines.append(self.generate_bbox(label, calib))
        self.generate_img(label)

        # Draw bbox into point cloud
        draw([self.pcd_bin, *lines], width=900, height=600, point_size=2)

    def draw_cars_from_results(self, results, num_cars):
        """
        Draws num_cars from inference results.

        Parameters
        ----------
        results: array
            mmdet inference results
        calib: str
            path to calibration file
        num_cars: int
            number of cars to draw
        """
        results = results.pred_instances_3d.bboxes_3d.tensor.to('cpu').detach().numpy()
        self.load_results(results)
        cars = []
        for result in self.results:
            cars.append(result)
            if len(cars) == num_cars:
                break
        
        lines = []
        for car in cars:
            lines.append(self.generate_bbox_from_result(car))
        draw([self.pcd_bin, *lines], width=900, height=600, point_size=2)




class ImageVisualizer:
    def __init__(self, img_path):
        self.image = cv2.imread(img_path)

    def load_labels(self, labels_path):
        self.labels = DataProcessor(labels_path, None).process_label_file()

    def load_calib(self, calib_path):
        self.calib = DataProcessor(None, calib_path).process_calib_file()
        print(self.calib['P2'])

    def load_results(self, results):
        self.results = results.pred_instances_3d

    def load_calib_from_cam_to_img(self, cam_to_img):
        # Flatten the tensor and remove the last four zeros
        flattened = [item for sublist in cam_to_img for item in sublist][:-4]

        # Convert to desired dictionary format
        self.calib = {'P2': list(flattened)}
        print(self.calib['P2'])

    def project_3d_to_2d(self, points_3d, P):
        points_3d_ext = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        points_2d = np.dot(P, points_3d_ext.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
        return points_2d

    def draw_3d_box(self, dimensions, location, rotation, pitch, thickness=2):
        # Get dimensions, location and rotation_y from label
        h, w, l = dimensions
        x, y, z = location
        ry = rotation

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

        # Apply rotations    
        vertices = np.dot(Ry, vertices)
        vertices = np.dot(Rx, vertices)


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

    def draw_monodetection_labels(self, num_cars, pitch, thickness=2):
        cars = 0
        for label in self.labels:
            if label['type'] == 'Car':
                dimensions = label['dimensions']
                location = label['location']
                rotation = label['rotation_y']

                print(dimensions)
                print(location)
                print(rotation)

                # intrinsic_matrix = np.array(self.calib['P2']).reshape(3, 4)
                # intrinsic_matrix = intrinsic_matrix[:3, :3]

                # Draw the 3D bounding box
                self.draw_3d_box(dimensions, location, rotation, pitch, thickness)
                cars += 1
            if cars >=num_cars:
                break
        
        cv2.imwrite('viz_img.png', self.image)

    def draw_monodetection_results(self, score, pitch, thickness=2):
        for bbox in self.results:
            if bbox.scores_3d[0] > score:
                bbox = bbox.bboxes_3d.tensor[0].cpu().tolist()

                bbox_location = bbox[0:3]
                bbox_dim = bbox[3:6]
                bbox_rotation = bbox[6]

                # intrinsic_matrix = np.array(self.calib['P2']).reshape(3, 4)
                # intrinsic_matrix = intrinsic_matrix[:3, :3]

                bbox_dim = [bbox_dim[1], bbox_dim[2], bbox_dim[0]]
                image = self.draw_3d_box(bbox_dim, bbox_location, bbox_rotation, pitch=pitch, thickness=thickness)
            else:
                break
        
        cv2.imwrite('result_sensus.png', self.image)

    def draw_monodetectionlabels_from_config(self, pitch, gt_boxes_3d, thickness=1):
        cars = 0
        for bbox in gt_boxes_3d:
            dimensions = bbox[4], bbox[5], bbox[3]
            location = bbox[0:3]
            rotation = bbox[6]

            dimensions = list(dimensions)
            location = list(location)
            dimensions = [t.item() for t in dimensions]
            location = [t.item() for t in location]
            rotation = rotation.item()

            print(dimensions)
            print(location)
            print(rotation)

            # Draw the 3D bounding box
            self.draw_3d_box(dimensions, location, rotation, pitch, thickness)
            cars += 1
        
        cv2.imwrite('viz_img_cfg.png', self.image)

def draw_lidar_labels(pcd_file, calib, labels, img_path, num_cars):
    viz = LidarVisualizer(pcd_file, img_path)
    viz.load_calib(calib)
    viz.load_labels(labels)
    viz.draw_cars_from_labels(num_cars)

def draw_lidar_results(pcd_file, calib, results, num_cars):
    viz = LidarVisualizer(pcd_file, None)
    viz.load_calib(calib)
    viz.draw_cars_from_results(results, num_cars)

def draw_monodetection_labels(img_file, calib, labels, num_cars, pitch, thickness=2):
    viz = ImageVisualizer(img_file)
    viz.load_calib(calib)
    viz.load_labels(labels)
    # print(viz.labels)
    viz.draw_monodetection_labels(num_cars, pitch, thickness=thickness)

def draw_monodetection_results(img_file, calib, results, score, pitch, thickness=2):
    viz = ImageVisualizer(img_file)
    viz.load_calib(calib)
    viz.load_results(results)
    viz.draw_monodetection_results(score, pitch, thickness=thickness)

def adapt_monodetresult_to_label(result):
    result_adapted = {
        'type': 'Car',
        'location': result[:3].tolist(),
        'dimensions': result[3:6].tolist(),
        'rotation_y': result[6].item()
    }
    result_adapted['dimensions'] = list([result_adapted['dimensions'][1], result_adapted['dimensions'][2], result_adapted['dimensions'][0]])
    return result_adapted

def adapt_monodetresults_to_labels(results):
    results_adapted = []
    for result in results:
        results_adapted.append(adapt_monodetresult_to_label(result))
    return results_adapted

def draw_monorestults_in_lidar(bin_path, calib, results, img_path, labels):
    results_adapted = adapt_monodetresults_to_labels(results.pred_instances_3d.bboxes_3d.tensor.to('cpu').detach().numpy())
    viz = LidarVisualizer(bin_path, img_path)
    viz.load_calib(calib)
    viz.load_labels(labels)
    viz.draw_cars_from_monodet_result(results_adapted)

def proccess_config(config_file, mono=True, Lidar=True):
    cfg = Config.fromfile(config_file)
    if mono==True and cfg['input_modality']['use_camera']==False:
        raise ValueError("Error: config file is not set to use camera images")
    if Lidar==True and cfg['input_modality']['use_lidar']==False:
        raise ValueError("Error: config file is not set to use lidar data")
    
    register_all_modules()

    train_data_cfg = cfg['train_dataloader']['dataset']

    # Obtain dataset type and then delete the "type" field from the config
    dataset_type = train_data_cfg.pop("type")

    DatasetClass = globals()[dataset_type]
    train_dataset = DatasetClass(**train_data_cfg)

    return train_dataset 

def draw_monolabels_from_config(config_path, sample_id):
    train_dataset = proccess_config(config_path, mono=True, Lidar=False)
    
    sample = train_dataset[sample_id]
    img_path = sample['data_samples'].img_path
    cam_to_img = sample['data_samples'].cam2img
    gt_boxes_3d = sample['data_samples'].gt_instances_3d.bboxes_3d.tensor

    print(img_path)

    viz = ImageVisualizer(img_path)
    viz.load_calib_from_cam_to_img(cam_to_img)

    viz.draw_monodetectionlabels_from_config(pitch=0.2031, gt_boxes_3d=gt_boxes_3d)

    print(viz.calib)

