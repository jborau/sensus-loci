import pickle

class DataProcessor:
    def __init__(self, labels_path, calib_path):
        self.labels_path = labels_path
        self.calib_path = calib_path

    def process_label_line(self, label_line):
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

    def process_label_file(self):
            labels = []
            with open(self.labels_path, 'r') as file:
                for line in file:
                    label = self.process_label_line(line)
                    labels.append(label)
            return labels

    def process_calib_file(self):
        calib = dict()
        with open(self.calib_path, 'r') as file:
            for line in file:
                line = line.split(' ')
                if line[0] == 'P2:':
                    calib['P2'] = [float(line[i]) for i in range(1, len(line))]
                elif line[0] == 'Tr_velo_to_cam:':
                    calib['Tr_velo_to_cam'] = [float(line[i]) for i in range(1, len(line))]
                elif line[0] == 'R0_rect:':
                    calib['R0_rect'] = [float(line[i]) for i in range(1, len(line))]
                elif line[0] == 'Tr_imu_to_velo:':
                    calib['Tr_imu_to_velo'] = [float(line[i]) for i in range(1, len(line))]
        return calib

def create_pkl(file_name, calib_file, cam_type = 'CAM_BACK', pkl_path = 'data.pkl'):
    # Get the camera to image matrix
    array = DataProcessor(None, calib_file).process_calib_file()['P2']

    # Remove the last 3 elements of the array
    indices_to_remove = [3, 7, 11]
    array = [array[i] for i in range(len(array)) if i not in indices_to_remove]

    # Reshape the array into a 3x3 matrix
    cam2img = [array[i:i+3] for i in range(0, len(array), 3)]

    # Create the data dictionary
    data = [{'images': {cam_type: {'img_path': f'{file_name}.png', 'cam2img': cam2img}}}]

    # Save data to the pkl file
    with open(pkl_path, "wb") as file:
        pickle.dump(data, file)

    print(f"Data saved to {pkl_path}")