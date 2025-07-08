import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pickle as pkl
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class MyPM25Dataset(Dataset):

    def __init__(self, root_dir, holdout, crop_dim=0, img_transform=None, mode='train', train_stations=-1, 
                 requires_meteo=False, meteo_model=None, rf_train=None, rf_test=None, normalized=False):
    
        if mode not in ['train', 'test']:
            raise Exception('Mode must be either \'train\' or \'test\'.')
        if requires_meteo and not meteo_model:
            raise Exception('If meteo features are required, you must pass in a model to transform the meteo features.')
        if requires_meteo:
            if mode == 'train' and rf_train is None:
                raise Exception('Please pass in training predictions from Random Forest model')
            elif mode == 'test' and rf_test is None:
                raise Exception('Please pass in test predictions from Random Forest model')
        
        # Pass in parameters
        self.crop_dim = crop_dim
        self.img_transform = img_transform
        self.mode = mode
        self.holdout = holdout
        self.train_stations = train_stations
        self.requires_meteo = requires_meteo
        self.y_train_pred_rf = rf_train
        self.y_test_pred_rf = rf_test
        self.normalized = normalized
        
        # Private variables
        self.img_train_PM25, self.img_test_PM25 = [], []
        self.PM25, self.PM25_train, self.PM25_test = [], [], []
        self.train_set = set()
        self.scaler = None
        
        self.meteo_raw = []
        self.meteo_raw_train, self.meteo_raw_test = [], []
        self.meteo_transformed_train, self.meteo_transformed_test = [], []      
        
        # Load images, meteo features and targets for PM2.5 data
        with open(root_dir, "rb") as fp:
            images = pkl.load(fp)
            for data_point in images:
                self.PM25.append(data_point['PM25'])
                if data_point['Station_index'] not in self.holdout:
                    self.train_set.add(data_point['Station_index'])
            self.train_set = sorted(list(self.train_set))
            
            if self.normalized:
                self.scaler = StandardScaler()
                self.PM25 = np.squeeze(self.scaler.fit_transform(np.array(self.PM25).reshape(-1, 1)))
                for i in range(len(images)):
                    images[i]['PM25'] = self.PM25[i]
            
            if self.train_stations != -1:
                self.train_set = self.train_set[:train_stations]

            for data_point in tqdm(images, position=0, leave=True):
                if data_point['Station_index'] in self.train_set:
                    self.img_train_PM25.append(data_point['Image'])
                    self.PM25_train.append(data_point['PM25'])
                    if self.requires_meteo:
                        self.meteo_raw_train.append(data_point['Meteo'].values)
                elif data_point['Station_index'] in self.holdout:
                    self.img_test_PM25.append(data_point['Image'])
                    self.PM25_test.append(data_point['PM25'])
                    if self.requires_meteo:
                        self.meteo_raw_test.append(data_point['Meteo'].values)
        
        
            
        # Remove unnecessary data
        if self.mode == 'train':
            del self.img_test_PM25, self.PM25_test, self.meteo_transformed_test
        else:
            del self.img_train_PM25, self.PM25_train, self.meteo_transformed_train
        del self.meteo_raw, self.meteo_raw_train, self.meteo_raw_test
            
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.img_train_PM25)
        else:
            return len(self.img_test_PM25)
        
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get images, transformed meteo features and targets
        if self.mode == 'train':
            img = self.img_train_PM25[idx]
            target = self.PM25_train[idx]
            if self.requires_meteo:
                meteo = self.meteo_transformed_train[idx]
                target_pred = self.y_train_pred_rf[idx]
        else:
            img = self.img_test_PM25[idx]
            target = self.PM25_test[idx] 
            if self.requires_meteo:
                meteo = self.meteo_transformed_test[idx]
                target_pred = self.y_test_pred_rf[idx]
        # Crop the image if crop_dim is specified
        if self.crop_dim != 0:
            crop = transforms.Compose([transforms.ToPILImage(), 
                                       transforms.CenterCrop((self.crop_dim, self.crop_dim)),
                                       transforms.ToTensor()])
            img = crop(img)
        # Perform data augmentation if transform function is specified
        if self.img_transform:
            img = self.img_transform(img)
        
        if self.requires_meteo:
            return img, meteo, target, target_pred
        else:
            return img, target

# Initialize the data loader for CNN models with regular MSE loss
def initializeCNNdata(root_dir, img_transform, batch_size, crop_dim=0, holdout=None, train_stations=-1, requires_meteo=False, rt_model=None, rf_train=None, rf_test=None, normalized=False):
    if requires_meteo:
        if (rt_model is None) or (rf_train is None) or (rf_test is None):
            raise Exception("Must specify rt_model, rf_train and rf_test.")
        train_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, holdout=holdout, img_transform=img_transform, crop_dim=crop_dim,  mode='train', 
                                           train_stations=train_stations, requires_meteo=requires_meteo, meteo_model=rt_model, rf_train=rf_train, normalized=normalized)
        test_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, holdout=holdout, img_transform=img_transform, crop_dim=crop_dim, mode='test', 
                                          train_stations=train_stations, requires_meteo=requires_meteo, meteo_model=rt_model, rf_test=rf_test, normalized=normalized)
    else:
        train_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, holdout=holdout, img_transform=img_transform, crop_dim=crop_dim, mode='train', 
                                           train_stations=train_stations, requires_meteo=requires_meteo, normalized=normalized)
        test_dataset_PM25 = MyPM25Dataset(root_dir=root_dir, holdout=holdout, img_transform=img_transform, crop_dim=crop_dim, mode='test', 
                                          train_stations=train_stations, requires_meteo=requires_meteo, normalized=normalized)
    train_dataloader_PM25 = DataLoader(train_dataset_PM25, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=np.random.seed(2020))
    train_dataloader_PM25_for_test = DataLoader(train_dataset_PM25, batch_size=128, shuffle=False)
    test_dataloader_PM25 = DataLoader(test_dataset_PM25, batch_size=128, shuffle=False)
    print(len(train_dataset_PM25), len(test_dataset_PM25))
    if requires_meteo:
        return train_dataloader_PM25, train_dataloader_PM25_for_test, test_dataloader_PM25, train_dataset_PM25[0][1].shape[0]
    else:
        return train_dataloader_PM25, train_dataloader_PM25_for_test, test_dataloader_PM25, train_dataset_PM25.scaler

def getTestStations(root_dir, holdout, sort=False):
    with open(root_dir, "rb") as fp:
        images = pkl.load(fp)
        if sort:
            images.sort(key=lambda x: x['Meteo'].name)
        test_stations = []
        for data_point in images:
            if data_point['Station_index'] in holdout:
                test_stations.append(data_point['Station_index'])
    return test_stations

# Get all the station names
def getAllStations(root_dir):
    with open(root_dir, "rb") as fp:
        stations = []
        for data_point in pkl.load(fp):
            if data_point['Station_index'] not in stations:
                stations.append(data_point['Station_index'])
    return stations

# Calculate spatial Pearson R and RMSE of all stations for testing
def calculateSpatial(y_test_pred, y_test, test_stations):
    df = pd.DataFrame({'y_test_pred': y_test_pred, 'y_test': y_test, 'test_stations': test_stations}).groupby(['test_stations']).mean()
    test_station_avg_pred = np.array(df.y_test_pred)
    test_station_avg = np.array(df.y_test)
    _, _, Rsquared_pearson, _ = eval_stat(test_station_avg_pred, test_station_avg)
    rmse = np.sqrt(metrics.mean_squared_error(test_station_avg, test_station_avg_pred))
    return Rsquared_pearson, rmse, test_station_avg_pred, test_station_avg

     