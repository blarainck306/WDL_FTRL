from matplotlib import pyplot as plt
def plot_training_stats(train_loss_history,test_loss_history):
  plt.figure(figsize=(16,6))
  plt.title('loss vs epochs')
  plt.subplot(221)
  plt.plot(train_loss_history,label  = 'train epoch loss')
  plt.grid()
  plt.ylabel('training loss')
  plt.legend()

  plt.subplot(222)
  plt.plot(train_loss_history,label = 'train epoch loss (log scale)')
  plt.yscale("log")
  plt.grid()
  plt.ylabel('training loss')
  plt.legend()

  plt.subplot(223)
  plt.plot(test_loss_history,label  = 'test epoch loss')
  plt.grid()
  plt.ylabel('test loss')
  plt.legend()

  plt.subplot(224)
  plt.plot(test_loss_history,label = 'test epoch loss (log scale)')
  plt.yscale("log")
  plt.ylabel('test loss')
  plt.grid()
  plt.legend()
  plt.show()
  
  
#------use 'Per-parameter options' of torch.optim
def customize_para_group_1(model, selected_weight_decay = 1e-4, skip_list=()):
    decay = []
    decay_name = []
    
    no_decay = []
    no_decay_name = []
    for name, param in model.named_parameters():
        # freezed prams
        if not param.requires_grad:
            continue
        # no decay: embedding layer, batch norm,
        if ('emb' in name) or ('bn' in name) or ('bias' in name) or (name in skip_list):
            no_decay.append(param)
            no_decay_name.append(name)
        # decay: 
        else:
            decay.append(param)
            decay_name.append(name)
    return [
          {'params': no_decay},
          {'params': decay, 'weight_decay': selected_weight_decay}],decay_name,no_decay_name



def print_lr(optimizer):
    '''
    given an optimizer, print out its learning rate
    '''
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        
def calc_batch_interval(num_print_per_epoch,full_data,batch_size):
    if full_data:
      total_data = 36210029
    else:
      total_data = 362088
    batch_interval = total_data/batch_size/num_print_per_epoch
    batch_interval = int(batch_interval)
    return batch_interval
           

import numpy as np
import torch

def get_transform_spec(loader_cols):
    return TransformSpec(func = None, selected_fields = loader_cols)

def prepare_ROC_AUC(model,converter_test,loader_cols,optimizer_mode):
    num_samples = len(converter_test)
    y_test = np.empty(num_samples,dtype = np.float32)
    probas = np.empty(num_samples,dtype = np.float32)
    batch_size = 64 # batch sise here applied to predictino below, it does not matter what the batch size we used in training
    model.w_values_array = np.empty((batch_size,model.num_total_wide_features),dtype = np.float32) # np.array
    with torch.no_grad():
        with converter_test.make_torch_dataloader(batch_size = batch_size, transform_spec = model.get_transform_spec(loader_cols),num_epochs = 1,shuffle_row_groups = False, workers_count = 2) as test_loader:
            for i, row_batch in enumerate(test_loader):
                X_w_indices = row_batch['hashed_wide'].int()
                X_d = row_batch['embedding_indexed'].long()
                y = row_batch['label']
                if torch.cuda.is_available():
                    X_d, y = X_d.cuda(), y.cuda()
                y_test[i*batch_size:i*batch_size+y.size(0)] = y.cpu()
                if optimizer_mode== 'OGD_WIDE' or optimizer_mode== 'OGD':
                  y_pred = torch.empty(y.shape,dtype = y.dtype,requires_grad = False,device = y.device)
                  model(X_w_indices, X_d,y_pred,y, training = False)
                else: # 'FTRL' and 'OGD_DEEP'
                  y_pred = model(X_w_indices, X_d,y, training = False)
                if len(y_pred.shape) == 2:
                  probas[i*batch_size:i*batch_size+y.size(0)] = y_pred[:,0].cpu() # use slicing in case of y_pred is [64,1], could not assign to probas
                else: #len(y_pred.shape) == 1
                  probas[i*batch_size:i*batch_size+y.size(0)] = y_pred.cpu()
    return y_test, probas

  
  

          
          
