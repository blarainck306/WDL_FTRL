# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
from math import exp,sqrt
# import line_profiler

from petastorm import TransformSpec

use_cuda = torch.cuda.is_available()

debug_mode = False
def print_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    
class WideDeep(nn.Module):
    """ Wide and Deep model. As explained in Heng-Tze Cheng et al., 2016, the
    model taked the wide features and the deep features after being passed through
    the hidden layers and connects them to an output neuron. For details, please
    refer to the paper and the corresponding tutorial in the tensorflow site:
    https://www.tensorflow.org/tutorials/wide_and_deep

    Parameters:
    --------
    DEEP
    - embeddings_input (tuple): 3-elements tuple with the embeddings "set-up" -
    (col_name, unique_values, embeddings dim)
    - continuous_cols (list) : list with the name of the continuum columns
    - deep_column_idx (dict) : dictionary where the keys are column names and the values
    their corresponding index in the deep-side input tensor
    - hidden_layers (list) : list with the number of units per hidden layer
    - encoding_dict (dict) : dictionary with the label-encode mapping
    
    - dropout (float)
    
    WIDE:
    - num_total_wide_features: number of wide features, is used to tell the size of w_array one the 2nd dimension for a single data observation, including interactive features....
    - num_deep_features: not usefule in this verison
    - D : similar to wide_dim, number of weights (including bias) to use for the "wide" part

    OTHERS:
    - n_class (int) : number of classes. Defaults to 1 if logistic or regression
    ====WIDE

    """
    def __init__(self,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 batch_norm,
                 dropout,
                 n_class,
                 num_total_wide_features,
                 D):

        super(WideDeep, self).__init__()
        #------deep part
        # self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        # self.encoding_dict = encoding_dict
        self.n_class = n_class
        self.num_total_wide_features = num_total_wide_features
        # self.num_deep_features = num_deep_features

        #-------wide part:
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = np.zeros(D, dtype = np.float32)
        self.z = np.zeros(D, dtype = np.float32)
        

        # ---feature related parameters
        self.D = D


        # only one layer for deep side, for learning bias term, input to this layer is going to be zero vector
        self.final_partial_fc = nn.Linear(1, self.n_class, bias = True)


    def compile(self, optimizer, alpha, beta, L1, L2,method="logistic"):
        """Wrapper to set the activation, loss and the optimizer.

        Parameters:
        ----------
        method (str) : regression, logistic or multiclass
        optimizer (str): SGD, Adam, or RMSprop
        """
        # ---hyper parameters
        self.alpha = alpha # learning rate
        self.beta = beta   # smoothing parameter for adaptive learning rate#
        self.L1 = L1       # L1 regularization, larger value means more regularized
        self.L2 = L2       # L2 regularization, larger value means more regularized
        
        if method == 'regression':
            self.activation, self.criterion = None, F.mse_loss
        if method == 'logistic': # Avazu
            self.activation, self.criterion = torch.sigmoid, F.binary_cross_entropy # used to use F.sigmoid
        if method == 'multiclass':
            self.activation, self.criterion = F.softmax, F.cross_entropy

        self.optimizer = optimizer

        # if optimizer == "Adagrad":
        #     self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        # if optimizer == "Adam":
        #     self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # if optimizer == "RMSprop":
        #     self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        # if optimizer == "SGD":
        #     self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.method = method

    def get_transform_spec(self,loader_cols):
        return TransformSpec(func = None, selected_fields = loader_cols)

    def indices_inter(self, x):
        ''' 
        # first yield index of the bias term,note hash(0)=0 in python, liu: but assume we don't have  zero value in the the feature values for wide features?
        # - yield 0 # bias introduced in deep side
        # - then yield the normal indices
        '''
        pass


    def forward_wide(self,x_w_indices,j):
        '''
        doc: this funciton calculate wide_z for a SINGLE data observation
        x_wide: a list of index of hashed features
        z_wide (scaler): z from wide side 
        '''
        # hyper parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z

        # ----vectorized implementation for calculating wTx 
        z_temp = z[x_w_indices]
        n_temp = n[x_w_indices]
        sign_temp =np.where(z_temp<0,-1.,1.)
        w_temp = np.where(sign_temp*z_temp<= L1,0.,(sign_temp * L1 - z_temp) / ((beta + np.sqrt(n_temp)) / alpha + L2))
        wTx = np.sum(w_temp)

        # cache the current w for update stage
        self.w_values_array[j,:] = w_temp

        z_wide = max(min(wTx, 35.), -35.)
        return z_wide


    def forward(self, X_w_indices, X_d,y,training = True):
        """Implementation of the forward pass.

        Parameters:
        ----------
        x_wide (list):  (batch_size,), wide-side input
        X_d (2-d tensor) : (batch_size, num_features),deep-side input tensor; 

        Returns:
        --------
        out (torch.tensor) : result of the output neuron(s)
        """
        # =========Deep Side
        # ---- get embedding concatenated with continuous features: 'deep_inp'
        # -- get embeddings for embedding features: emb (a list of embeddings, one embedding for each feature respectively)
        fake_input = torch.zeros((X_w_indices.shape[0],1),requires_grad = False, dtype = torch.float32)
        deep_z = self.final_partial_fc(fake_input)

        # ========= wide side 
        wide_z = torch.empty(deep_z.shape, requires_grad = False, dtype = torch.float32, device = X_d.device)
        #iterate over training samples within a batch
        for j in range(X_w_indices.shape[0]):
            wide_z[j] = self.forward_wide(X_w_indices[j,:],j)

        y_pred = self.activation(wide_z + deep_z)

        # iterate within a batch
        if training:
            for j in range(X_w_indices.shape[0]):
                self.update(X_w_indices[j,:],y_pred[j],y[j],j) # update parameters for wide side
        return y_pred

    def update(self,x_wide,y_pred,y,j):
        '''
        update necessary states for calculating the gradients of w;
        MODIFIES:
                self.n: increase by squared gradient
                self.z: weights

        parameters
        ---------
        x_wide (iterable):  (num_total_wide_features,), feature indices on wide side
        y_pred (float) : prediction (probablity) from model
        y (float): gound truth

        return
        None
        ----------
        '''
        # convert y_pred,y  to scaler
        p = y_pred.item()
        y = y.item()

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        # w = self.w_dic_array[j]
        w = self.w_values_array[j,:]

        # gradient under logloss, this is because: if x_i != 0, g = (p-y)x_i= p-y
        g = p - y

        # ---new vectorized implementation: update z and n, for x_i = 0, gradient is zero, so no update for them
        sigma_temp = (np.sqrt(n[x_wide] + g * g) - np.sqrt(n[x_wide]))/alpha
        z[x_wide] += (g - sigma_temp * self.w_values_array[j,:])
        n[x_wide] += g * g


    def eval_model(self, converter_test, best_loss,best_model_wts,loader_cols, batch_size=32):
        '''
        This function is called  after each epoch of training on the training data. 
        This function measure the performance of the model using the dev dataset.

        inputs: 
        - test_loader
        outputs:
        - test_loss: test loss for current epoch
        - best_loss: best loss so far
        - best_model_wts: best model weights so far
        '''
        running_loss = 0.0
        running_num_samples = 0.0

        with torch.no_grad():
            with converter_test.make_torch_dataloader(batch_size = batch_size, transform_spec = self.get_transform_spec(loader_cols),num_epochs = 1,shuffle_row_groups = False, workers_count = 2) as test_loader:
                for i, row_batch in enumerate(test_loader):
                    X_w_indices = row_batch['hashed_wide'].int()
                    X_d = row_batch['embedding_indexed'].long()
                    y = row_batch['label']

                    if use_cuda:
                        X_d, y = X_d.cuda(), y.cuda()
                    # forward:
                    y_pred = self(X_w_indices, X_d,y, training = False)# y_pred got updated, passed y_pred as arguments
                    loss = self.criterion(y_pred, y.view(-1,1))

                    running_loss += loss.item() * y.size(0)
                    running_num_samples += y.size(0)

        # calculating test loss on all dev dataset
        test_loss = running_loss / running_num_samples # avg loss/sample
        # update best test loss so far if necessary
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(self.state_dict())
        print('Current test loss: %.4f. So far best test loss: %.4f' % (test_loss,best_loss))
        return test_loss, best_loss, best_model_wts

    def fit(self, converter_train,best_model_wts, best_loss, converter_test,batch_interval,loader_cols, n_epochs, batch_size,shuffle_row_groups):
        """Run the model for the training set at dataset.

        Parameters:
        ----------
        - dataset (dict): dictionary with the training sets: X_wide_train, X_deep_train, target
        - n_epochs (int)
        - batch_size (int)
        """
        self.w_values_array = np.empty((batch_size,self.num_total_wide_features),dtype = np.float32) # np.array
        train_loss_history = []
        test_loss_history = []

        # evalate the model at the very beginning
        # self.eval()
        # test_loss, best_loss, best_model_wts = self.eval_model(converter_test, best_loss, best_model_wts,loader_cols,batch_size)
        # test_loss_history.append(test_loss)
        
        for epoch in range(n_epochs):
            print('======')
            print_time()
            self.train()
            # for epoch training performance tracking
            # running_loss = 0
            # running_total=0
            
            # for batch performance tracking
            running_loss_batch = 0
            running_total_batch=0

            # petastorm loader: note rows of data will be expressed as a dictionary, with column names as the keys
            with converter_train.make_torch_dataloader(batch_size = batch_size, transform_spec = self.get_transform_spec(loader_cols),num_epochs = 1,shuffle_row_groups = shuffle_row_groups, workers_count = 2) as train_loader:
                for i, row_batch in enumerate(train_loader):
                    X_w_indices = row_batch['hashed_wide'].int()
                    X_d = row_batch['embedding_indexed'].long()
                    y = row_batch['label']

                    if use_cuda:
                         X_d, y = X_d.cuda(), y.cuda()

                    self.optimizer.zero_grad()
                    # ----forward
                    y_pred = self(X_w_indices, X_d,y)# y_pred got updated, passed y_pred as arguments
                    loss = self.criterion(y_pred, y.view(-1,1))

                    # ----backward (calc gradients) for 'deep'
                    loss.backward()
                    
                    # ---- optimization:update gradient using gradient
                    #--deep:
                    self.optimizer.step()
                    #--wide: get gradient  and get ready for update for 'WIDE', the actual update for 'wide' is in function forward()

                    #TODO 
                    # ----record 'running_total','running_correct','running_loss'
                    # running_total+= y.size(0)
                    # running_loss += loss.item() *  y.size(0)

                    #----print out loss for current batch if it is multiple of batch_interval
                    running_loss_batch += loss.item() *  y.size(0)
                    running_total_batch += y.size(0)
                    if  i%batch_interval==0 and i!=0:
                        print('-----')
                        print_time()
                        batches_loss = running_loss_batch/running_total_batch
                        print("batch {}, avg training loss {} per sample within batches".format(i,round(batches_loss,3)) )
                        running_loss_batch, running_total_batch = 0,0
                        self.eval()
                        test_loss, best_loss, best_model_wts = self.eval_model(converter_test, best_loss,best_model_wts, loader_cols, batch_size)
                        self.train()
                    
            # # ------print out training loss, accuracy for each epoch
            # epoch_loss = running_loss / running_total
            # print ('Epoch {} of {}, Training Loss: {}'.format(epoch+1, n_epochs, round(epoch_loss,3)) )
            # train_loss_history.append(epoch_loss)

            # # ------at each epoch, evaluate the trained model using the test data
            # self.eval()
            # test_loss, best_loss, best_model_wts = self.eval_model(converter_test, best_loss,best_model_wts, loader_cols, batch_size)
            # test_loss_history.append(test_loss)
            # print_time()


        return train_loss_history, test_loss_history, best_loss, best_model_wts
