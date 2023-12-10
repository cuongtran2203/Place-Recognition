import os
from torch.utils.data import Dataset
import numpy as np
import scipy.io
from sklearn.neighbors import NearestNeighbors
import cv2
import random
import math


class GSVDataset(Dataset):
    def __init__(self, data_dir, distance_inner_threshold=80, distance_outer_threshold=10000, limit_angle=10, batch_size=16, transform=None):
        """
        data_dir: path to the data folder which contains images, Cartesian_Location_Coordinates.mat, GPS_Long_Lat_Compass.mat
            example:
                GSV_data/
                    000001_0.jpg
                    000001_1.jpg
                    ...
                    010343_5.jpg
                    Cartesian_Location_Coordinates.mat
                    GPS_Long_Lat_Compass.mat
        distance_inner_threshold: threshold for samples that are considered to have same labels (meters)
        distance_outer_threshold: threshold for samples that are considered to have different labels (meters)
        limit_angle: threshold for the difference in viewing angle and connection between two samples that are considered to have same labels
        """
        self.filenames = sorted([os.path.join(data_dir, i) for i in os.listdir(data_dir) if i.endswith('.jpg') and not i.endswith('_0.jpg') and not i.endswith('_5.jpg')])
        self.coordinates = scipy.io.loadmat(os.path.join(data_dir, 'Cartesian_Location_Coordinates.mat'))['XYZ_Cartesian']
        self.gps_compass = scipy.io.loadmat(os.path.join(data_dir, 'GPS_Long_Lat_Compass.mat'))['GPS_Compass']
        self.gps = self.gps_compass[:, :2]
        self.compass = self.gps_compass[:, -1]
        self.distance_inner_threshold = distance_inner_threshold
        self.distance_outer_threshold = distance_outer_threshold
        self.limit_angle = limit_angle
        self.batch_size = batch_size
        self.nn = NearestNeighbors(n_jobs=-1)
        self.nn.fit(self.coordinates)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def validate_angle(self, path1, path2):
        basename1 = os.path.basename(path1)
        basename2 = os.path.basename(path2)
        position_id1 = int(basename1.split('_')[0])
        position_id2 = int(basename2.split('_')[0])
        side_id1 = int(basename1.split('_')[-1].split('.')[0])
        side_id2 = int(basename2.split('_')[-1].split('.')[0])
        angle1 = self.compass[position_id1-1]
        angle1 = (angle1 - 90*side_id1 + 360) % 360
        angle2 = self.compass[position_id2-1]
        angle2 = (angle2 - 90*side_id2 + 366) % 360
        x1, y1 = self.gps[position_id1-1] - self.gps[position_id2-1]
        x2, y2 = self.gps[position_id2-1] - self.gps[position_id1-1]
        connection_angle1 = (math.atan2(y1,x1)/math.pi*180 + 360) % 360
        connection_angle2 = (math.atan2(y2,x2)/math.pi*180 + 360) % 360
        
        if (abs(connection_angle1-angle1) < self.limit_angle or abs(connection_angle1-angle1) > 360-self.limit_angle) and (abs(connection_angle1-angle2) < self.limit_angle or abs(connection_angle1-angle2) > 360-self.limit_angle):
            return True
        if (abs(connection_angle2-angle1) < self.limit_angle or abs(connection_angle2-angle1) > 360-self.limit_angle) and (abs(connection_angle2-angle2) < self.limit_angle or abs(connection_angle2-angle2) > 360-self.limit_angle):
            return True
        return False

    def __getitem__(self, i):
        final_paths = []
        final_labels = np.array([0]*self.batch_size)
        path = self.filenames[i]
        # Read index image
        basename = os.path.basename(path)
        # Get the id of the image
        position_id = int(basename.split('_')[0])
        query_position = self.coordinates[position_id-1]
        # Get positions that are in the specified radius
        neigh_dist, neigh_ind = self.nn.radius_neighbors(np.array([query_position]), radius=self.distance_inner_threshold)
        neigh_dist = neigh_dist[0]
        neigh_ind = neigh_ind[0]
        inner_paths = []
        for j in range(len(self.filenames)//4):
            if j in neigh_ind and j != position_id-1:
                inner_paths.extend(self.filenames[j*4:(j+1)*4])
        positive_paths = [j for j in inner_paths if self.validate_angle(j, path)]
        random.shuffle(positive_paths)
        final_paths.append(path)
        final_paths.extend(positive_paths[:self.batch_size//2-1])
        # Get positions that are outside the specified radius
        neigh_dist, neigh_ind = self.nn.radius_neighbors(np.array([query_position]), radius=self.distance_outer_threshold)
        neigh_ind = neigh_ind[0]
        outer_paths = []
        for j in range(len(self.filenames)//4):
            if j not in neigh_ind:
                outer_paths.extend(self.filenames[j*4:(j+1)*4])
        final_labels[len(final_paths):] = np.arange(1, self.batch_size-len(final_paths)+1)
        random.shuffle(outer_paths)
        final_paths.extend(outer_paths[:self.batch_size-len(final_paths)])
        images = [cv2.imread(j) for j in final_paths]
        if self.transform:
            images = [self.transform(image=j)['image'] for j in images]
        images = np.array(images)
        return images, final_labels