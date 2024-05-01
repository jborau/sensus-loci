import pickle
import cv2
import numpy as np
import open3d as o3d

class LidarBbox:
    def __init__(self, location, dimensions, rotation, type = None, score=None):
        self.location = location # x, y, z
        self.dimensions = dimensions # l, w, h
        self.rotation = rotation # yaw, pitch, roll
        if type is not None:
            self.type = type
        if score is not None:
            self.score = score

    def __str__(self):
        return f"LidarBbox Type: {self.type}, Location: {self.location}, \n Dimensions: {self.dimensions}, Rotation Y: {self.rotation}, Score: {getattr(self, 'score', 'Not available')})"

    @classmethod
    def from_label(cls, loc, dim, rot, type, Tr_velo_to_cam):
        """
        Create a LidarBbox instance from a label.
        
        This assumes 'label' is a format like: [type, x, y, z, l, w, h, yaw, pitch, roll, score]
        Adjust the indices based on your label format.
        """

        # Convert to lidar coordinates
        location = loc[0], loc[1], loc[2]
        dimensions = dim[0], dim[2], dim[1]
        rotation = [-rot + np.pi/2]
        type = type

        pos_lidar = cls.transform_position_to_lidar_coords(location, Tr_velo_to_cam)
        
        return cls(pos_lidar, dimensions, rotation, type)
    
    @classmethod
    def from_result(cls, bbox_result, score, label):

        location = bbox_result[0:3]
        dimensions = bbox_result[3:6]
        rotation = bbox_result[6:9]
        type = label
        score = score
        return cls(location, dimensions, rotation, score=score, type = type)

        # Function to transform a position from camera to LiDAR coordinates
    def transform_position_to_lidar_coords(position, tr_velo_to_cam):
        # Reshape to 3x4 matrix
        tr_velo_to_cam_matrix = tr_velo_to_cam.reshape((3, 4))

        # Extend to 4x4 matrix for homogeneous coordinates
        tr_velo_to_cam_matrix_4x4 = np.vstack([tr_velo_to_cam_matrix, [0, 0, 0, 1]])

        # Invert the matrix
        tr_cam_to_velo_matrix_4x4 = np.linalg.inv(tr_velo_to_cam_matrix_4x4)

        # Example of using this matrix (transforming a point from camera to LiDAR coordinates)
        # Assuming you have a point in camera coordinates
        point_in_camera = np.array([position[0], position[1], position[2], 1]) # replace x, y, z with actual values

        # Transforming to LiDAR coordinates
        point_in_lidar = np.dot(tr_cam_to_velo_matrix_4x4, point_in_camera)
        
        return point_in_lidar[0:3]

class CameraBbox:
    def __init__(self, location, dimensions, rotation_y, type=None, score=None):
        self.location = location
        self.dimensions = dimensions
        self.rotation_y = rotation_y
        if type is not None:
            self.type = type
        if score is not None:
            self.score = score

    def __str__(self):
        return f"CameraBbox Type: {getattr(self, 'type', 'Not available')}, Location: {self.location}, \n Dimensions: {self.dimensions}, Rotation Y: {self.rotation_y}, Score: {getattr(self, 'score', 'Not available')})"



class SensusInstance:
    def __init__(self):
        self.images = None  # or an appropriate default value
        self.lidar_pcd = None  # or an appropriate default value
        self.camera_bboxes = []  # List of CameraBBox objects
        self.lidar_bboxes = []  # List of LidarBBox objects
        self.calib = dict() # Dictionary containing the calibration data

    def load_image_from_path(self, img_path):
        self.image = cv2.imread(img_path)

    def load_lidar_data(self, lidar_path):
        with open(lidar_path, 'rb') as f:
            points = np.fromfile(f, dtype=np.float32, count=-1).reshape([-1, 4])
        self.lidar_pcd = o3d.geometry.PointCloud()
        self.lidar_pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    def load_calib(self, calib_path):
        with open(calib_path, 'r') as file:
            for line in file:
                line = line.split(' ')
                if line[0] == 'P2:':
                    self.calib['P2'] = [float(line[i]) for i in range(1, len(line))]
                elif line[0] == 'Tr_velo_to_cam:':
                    self.calib['Tr_velo_to_cam'] = [float(line[i]) for i in range(1, len(line))]
                elif line[0] == 'R0_rect:':
                    self.calib['R0_rect'] = [float(line[i]) for i in range(1, len(line))]
                elif line[0] == 'Tr_imu_to_velo:':
                    self.calib['Tr_imu_to_velo'] = [float(line[i]) for i in range(1, len(line))]

    def load_camera_bboxes_from_labels(self, labels):
        for label in labels:
            loc = label['location']
            dim = label['dimensions']
            dim = [dim[2], dim[0], dim[1]]
            rot = label['rotation_y']
            type = label['type']
            self.camera_bboxes.append(CameraBbox(loc, dim, rot, type))

    def load_lidar_bboxes_from_labels(self, labels):
        for label in labels:
            loc = label['location']
            dim = label['dimensions']
            dim = [dim[2], dim[0], dim[1]]
            rot = label['rotation_y']
            type = label['type']
            Tr_velo_to_cam = np.array(self.calib['Tr_velo_to_cam'])
            bbox = LidarBbox.from_label(loc, dim, rot, type, Tr_velo_to_cam)
            self.lidar_bboxes.append(bbox)

    def load_bboxes_from_labels(self, labels_path, cam_bbox=True, lidar_bbox=True):
        labels = process_label_file(labels_path)
        if cam_bbox:
            self.load_camera_bboxes_from_labels(labels)
        if lidar_bbox:
            self.load_lidar_bboxes_from_labels(labels)

    def load_camera_bboxes_from_result(self, result):
        result3d = result.pred_instances_3d
        for i in range (len(result3d.scores_3d)):
            score = result3d.scores_3d[i].cpu().tolist()
            bbox = result3d.bboxes_3d.tensor[i].cpu().tolist()
            loc = bbox[0:3]
            dim = bbox[3:6]
            rot = bbox[6]
            type = result3d.labels_3d[i].cpu().tolist()
            self.camera_bboxes.append(CameraBbox(loc, dim, rot, score=score, type=type))

    def load_lidar_bboxes_from_result(self, result):
        result3d = result.pred_instances_3d
        for i in range (len(result3d.scores_3d)):
            score = result3d.scores_3d[i].cpu().tolist()
            bbox = result3d.bboxes_3d.tensor[i].cpu().tolist()
            label = result3d.labels_3d[i].cpu().tolist()
            bbox3d = LidarBbox.from_result(bbox, score, label)
            self.lidar_bboxes.append(bbox3d)

    def load_bboxes_from_result(self, result, cam_bbox=True, lidar_bbox=True):
        if cam_bbox:
            self.load_camera_bboxes_from_result(result)
        if lidar_bbox:
            self.load_lidar_bboxes_from_result(result)

def process_label_line(label_line):
    label_line = label_line.split(' ')
    label = dict()
    label['type'] = label_line[0]
    label['truncated'] = float(label_line[1])
    label['occluded'] = int(label_line[2])
    label['alpha'] = float(label_line[3])
    label['bbox'] = [float(label_line[i]) for i in range(4, 8)]
    label['dimensions'] = [float(label_line[i]) for i in range(8, 11)]
    label['location'] = [float(label_line[i]) for i in range(11, 14)]
    label['rotation_y'] = float(label_line[14])
    label['score'] = float(label_line[15]) if len(label_line) == 16 else None
    return label

def process_label_file(labels_path):
        labels = []
        with open(labels_path, 'r') as file:
            for line in file:
                label = process_label_line(line)
                labels.append(label)
        return labels
