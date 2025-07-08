import torch
import copy
from .Dataclass import calculateSpatial
import torch.nn as nn
def train_with_regular_loss(model, device, train_loader, criterion, optimizer, epoch, requires_meteo=False, scale_factor=-1, 
                            residual_factor=1):       
    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)
    if requires_meteo:
        None
    else:
        for batch_idx, (img, target) in enumerate(train_loader):
            img, target = img.to(device), torch.squeeze(target.to(device))
            optimizer.zero_grad()
            output = torch.squeeze(model(img,epoch))
            if len(output.shape) == 0:
                continue
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            y_pred = torch.cat((y_pred, output))
            y_true = torch.cat((y_true, target))
        
    train_loss = criterion(y_pred, y_true)
    print('Train Epoch: {} Loss: {:.6f}'.format(epoch, train_loss))

# Test function with regular MSE loss
def test_with_regular_loss(model, device,epoch, test_loader, criterion, use_train=False, requires_meteo=False, scale_factor=-1,
                           residual_factor=1, test_stations=None, scaler=None):  
    model.eval()
    y_pred = torch.empty(0).to(device)
    y_true = torch.empty(0).to(device)
    if requires_meteo:
        None
    else:
        with torch.no_grad():
            for img, target in test_loader:
                img, target = img.to(device), torch.squeeze(target.to(device))
                output = torch.squeeze(model(img,epoch))
                if len(output.shape) == 0:
                    continue
                y_pred = torch.cat((y_pred, output))
                y_true = torch.cat((y_true, target))
    
    test_loss = criterion(y_pred, y_true)
    if test_stations:
        if scaler is not None:
            spatial_R, spatial_rmse, _, _ = calculateSpatial(scaler.inverse_transform(y_pred.cpu().numpy()), scaler.inverse_transform(y_true.cpu().numpy()), test_stations)
        else:
            spatial_R, spatial_rmse, _, _ = calculateSpatial(y_pred.cpu().numpy(), y_true.cpu().numpy(), test_stations)
    
    if use_train:
        print('Train set Loss: {:.4f}'.format(test_loss))
    else:
        print('Test set Loss: {:.4f}'.format(test_loss))
        if test_stations:
            print('Test spatial RMSE: {:.4f}'.format(spatial_rmse))
    
    if test_stations:
        return y_pred, y_true, test_loss, spatial_R, spatial_rmse
    else:
        return y_pred, y_true, test_loss


# Run training and testing with regular MSE loss
def run_with_regular_loss(model, optimizer,device,check_loader,train_loader, train_loader_for_test, test_loader, encoder_name, max_epochs=500, save_model=False, lr_scheduler=None, 
                          early_stopping_threshold=-1, early_stopping_metric='test_loss', requires_meteo=False, scale_factor=-1, residual_factor=1, test_stations=None, scaler=None):
    assert early_stopping_metric in [None, 'test_loss', 'spatial_r', 'spatial_rmse'], "Early stopping metric should be one of the [test_loss, spatial_r, spatial_rmse]."
    
    criterion_train = nn.MSELoss(reduction='mean')
    criterion_test = nn.MSELoss(reduction='mean')
    
    y_train_pred_final, y_test_pred_final = torch.empty(0), torch.empty(0)
    y_train_final, y_test_final = torch.empty(0), torch.empty(0)
    loss_train_arr, loss_test_arr = [], []
    spatial_R_test_arr, spatial_rmse_test_arr = [], []
    loss_test_smallest, spatial_rmse_test_smallest, spatial_R_test_largest, early_stopping_count = 1e9, 1e9, 0, 0
    current_epochs = max_epochs + 0
    for epoch in range(1, max_epochs + 1):
        train_with_regular_loss(model, device, train_loader, criterion_train, optimizer, epoch, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor)
        if lr_scheduler is not None:
            lr_scheduler.step()
        y_train_pred, y_train, loss_train = test_with_regular_loss(model, device,epoch,train_loader_for_test, criterion_test, use_train=True, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor)
        if test_stations:
            y_test_pred, y_test, loss_test, spatial_R_test, spatial_rmse_test = test_with_regular_loss(model, device,epoch, test_loader, criterion_test, use_train=False, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor, test_stations=test_stations, scaler=scaler)
        else:
            y_test_pred, y_test, loss_test = test_with_regular_loss(model, device,epoch, test_loader, criterion_test, use_train=False, requires_meteo=requires_meteo, scale_factor=scale_factor, residual_factor=residual_factor, test_stations=test_stations, scaler=scaler)
        loss_train_arr.append(loss_train)
        loss_test_arr.append(loss_test)
        if test_stations is not None:
            spatial_rmse_test_arr.append(spatial_rmse_test)
            spatial_R_test_arr.append(spatial_R_test)
        if early_stopping_metric == 'test_loss':
            if loss_test < loss_test_smallest:
                early_stopping_count = 0
                loss_test_smallest = loss_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        elif early_stopping_metric == 'spatial_rmse' and test_stations is not None:
            if spatial_rmse_test < spatial_rmse_test_smallest:
                early_stopping_count = 0
                spatial_rmse_test_smallest = spatial_rmse_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        elif early_stopping_metric == 'spatial_r' and test_stations is not None:
            if spatial_R_test > spatial_R_test_largest:
                early_stopping_count = 0
                spatial_R_test_largest = spatial_R_test + 0
                y_train_pred_final = copy.copy(y_train_pred)
                y_test_pred_final = copy.copy(y_test_pred)
                y_train_final = copy.copy(y_train)
                y_test_final = copy.copy(y_test)
            else:
                early_stopping_count += 1
                if early_stopping_threshold > 0 and early_stopping_count > early_stopping_threshold:
                    current_epochs = epoch + 0
                    print('Early stopping criterion reached. Exiting...')
                    break
        else:
            y_train_pred_final = copy.copy(y_train_pred)
            y_test_pred_final = copy.copy(y_test_pred)
            y_train_final = copy.copy(y_train)
            y_test_final = copy.copy(y_test)
    
    if (save_model):
        if requires_meteo:
            torch.save(model.state_dict(), '/work/zj63/Contrastive_learning_for_PM25_prediction/model_checkpoint/pipeline_params_' + encoder_name + '_meteo.pkl')
        else:
            torch.save(model.state_dict(), '/kaggle/working/pipeline_params_' + encoder_name + '_no_meteo.pkl')

    