import torch
import copy
import numpy as np
from torch import optim
from torchvision import transforms
from sklearn import metrics
import os
import pickle as pkl
from .Dataclass import initializeCNNdata,calculateSpatial,getTestStations
from .model import ResNet_SimCLR_SimSiam_no_meteo
from .train import run_with_regular_loss
from ..utils.test_image_loader import check_loader

np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True

def run_supervised_SimCLR(requires_meteo=False, train_stations=-1, lr=5e-7):
    root_dir = '/kaggle/input/ujikko/Delhi_labeled.pkl'
    img_transform = transforms.ToTensor()
    holdout = ['Shadipur', 'North_Campus', 'R_K_Puram', 'Sector116', 'Sirifort', 'Patparganj', 'CRRI_MTR_Rd', 'Sector125', 
               'Major_Dhyan_Chand_National_Stadium', 'Aya_Nagar', 'NSIT_Dwarka', 'Sri_Aurobindo_Marg', 'Bawana', 'Loni', 
               'Sector1', 'Narela', 'Dwarka_Sector_8', 'Mundka', 'Sanjay_Nagar', 'ITO', 'Jahangirpuri', 'Alipur', 'Ashok_Vihar', 
               'Sonia_Vihar', 'New_Collectorate', 'Okhla_Phase2', 'Pusa_IMD']
    test_stations = getTestStations(root_dir, holdout=holdout)
    batch_size = 8
    fig_size = 1000
    scale_factor = 0.95
    scaler = None
    
    # Build Random Trees Embedding and Random Forest Model
    if requires_meteo:
        rt_dir = '../../rt_rf_checkpoint/rt_model_Delhi.pkl'
        rf_dir = '../../rt_rf_checkpoint/ML_RF_singlemet_Delhi.pkl'
    
    # Initialize the data for CNN
    if requires_meteo:
        print("no")
    else:
        train_loader, train_loader_for_test, test_loader, scaler = initializeCNNdata(root_dir, img_transform, batch_size, 
                                                                     holdout=holdout, train_stations=train_stations, requires_meteo=False, normalized=False)
    # Visualize the Random Forest predictions
    if requires_meteo:
        print('no')
        
        
    # Run supervised learning
    max_epochs = 200
    early_stopping_threshold = 20
    early_stopping_metric = 'spatial_rmse'
    encoder_name = 'resnet50_SimCLR'
    ssl_path = '/kaggle/input/dataset/encoder_params_resnet50_spatiotemporal_Delhi_SimCLR.pkl'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checky_loader = check_loader()
    if requires_meteo:
        print('no')
    else:
        model = ResNet_SimCLR_SimSiam_no_meteo(ssl_path, backbone='resnet50').to(device)
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.75, weight_decay=0.1)
    #optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.01, 0.01), weight_decay=0.1)
    gamma = 0.005
    exp_func = lambda epoch: np.exp(-gamma*epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_func)
    if requires_meteo:
        y_train_pred, y_train, y_test_pred, y_test, loss_train, loss_test, spatial_R_test, spatial_rmse_test, current_epochs = run_with_regular_loss(
            model, optimizer, device, train_loader, train_loader_for_test, test_loader, 
            encoder_name=encoder_name, 
            max_epochs=max_epochs, 
            save_model=False, 
            lr_scheduler=scheduler, 
            early_stopping_threshold=early_stopping_threshold, 
            early_stopping_metric=early_stopping_metric, 
            requires_meteo=True, 
            scale_factor=scale_factor, 
            test_stations=test_stations
        )
    else:
        y_train_pred, y_train, y_test_pred, y_test, loss_train, loss_test, spatial_R_test, spatial_rmse_test, current_epochs = run_with_regular_loss(
            model, optimizer, device,checky_loader,train_loader, train_loader_for_test, test_loader, 
            encoder_name=encoder_name, 
            max_epochs=max_epochs, 
            save_model=True, 
            lr_scheduler=scheduler, 
            early_stopping_threshold=early_stopping_threshold, 
            early_stopping_metric=early_stopping_metric, 
            requires_meteo=False, 
            test_stations=test_stations, 
            scaler=scaler
        )
    
    if scaler is not None:
        y_train_pred, y_train = scaler.inverse_transform(y_train_pred), scaler.inverse_transform(y_train)
        y_test_pred, y_test = scaler.inverse_transform(y_test_pred), scaler.inverse_transform(y_test)
    
    # Calculate spatial Pearson R
    spatial_R, spatial_rmse, station_avg_pred, station_avg = calculateSpatial(y_test_pred, y_test, test_stations)
    
    # Save spatial statistics
    result_stats = {'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)), 'Spatial_R': spatial_R, 'Spatial_RMSE': spatial_rmse}
    result_path = './model_results/results_SimCLR_spatiotemporal.pkl'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'ab') as fp:
        pkl.dump(result_stats, fp)

def cli_main():
    stations_num = [15]
    lrs_no_meteo = [1e-7]   # ResNet50
    lrs_meteo = []   # ResNet50

    for i in range(len(stations_num)):
        # run_supervised_SimCLR(requires_meteo=True, train_stations=stations_num[i], lr=lrs_meteo[i])
        run_supervised_SimCLR(requires_meteo=False, train_stations=stations_num[i], lr=lrs_no_meteo[i])
if __name__ == '__main__':
    cli_main()
