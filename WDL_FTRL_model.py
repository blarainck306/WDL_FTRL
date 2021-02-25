# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from datetime import datetime
import pytz
from math import exp,sqrt
# import line_profiler

from petastorm import TransformSpec

use_cuda = torch.cuda.is_available()

debug_mode = False
def print_time():
  tz_NY = pytz.timezone('America/New_York') 
  datetime_NY = datetime.now(tz_NY)
  print("time:", datetime_NY.strftime("%H:%M:%S"))
    
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
        self.w = [0.] * D
        self.b = 0.0
        self.decay_times = [0] * D # update decay_times already done for a coordinate when calculating activation
        self.q = 0 # recording the number of times each weights shold be decay, i.e., number of iterations all weights has been through, every iteration, each weight shold decay
        # ---feature related parameters
        self.D = D

        # only one layer for deep side, for learning bias term, input to this layer is going to be zero vector
        # self.final_partial_fc = nn.Linear(1, self.n_class, bias = True)

        self.activation, self.criterion = torch.sigmoid, F.binary_cross_entropy # used to use F.sigmoid


    def init_train_history(self):
        '''
        initialize object fileds before passing training history to current model object if necessary
        '''
        #----training history
        self.train_loss_history = [] # loss history since the very beginning

        self.test_loss_history = []
        self.best_test_loss_history = []

        self.best_test_loss = float('inf')
        self.best_model_wts = copy.deepcopy(self.state_dict()) # weights that got best log loss on dev set 
        self.best_n = None  
        self.best_z = None


    def compile(self, optimizer, lr_w,L2_decay):
        """
        the optimizer for wide and deep respectively

        Parameters:
        ----------
        method (str) : regression, logistic or multiclass
        optimizer (str): SGD, Adam, or RMSprop
        """
        # ---hyper parameters
        self.lr_w = lr_w
        self.L2_decay = L2_decay # L2_decay is proportional to lr according to the formula

        
        self.optimizer = optimizer

        # if optimizer == "Adagrad":
        #     self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        # if optimizer == "Adam":
        #     self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # if optimizer == "RMSprop":
        #     self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        # if optimizer == "SGD":
        #     self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)


    def get_transform_spec(self,loader_cols):
        return TransformSpec(func = None, selected_fields = loader_cols)

    def indices_inter(self, x):
        ''' 
        # first yield index of the bias term,note hash(0)=0 in python, liu: but assume we don't have  zero value in the the feature values for wide features?
        # - yield 0 # bias introduced in deep side
        # - then yield the normal indices
        '''
        pass


    def forward_wide(self,x_w_indices):
        '''
        doc: this funciton calculate wide_z for a SINGLE data observation
        x_w_indices: a list of index of hashed features
        z_wide (scaler): z from wide side for a single training sample
        '''

        decay_times = self.decay_times
        w = self.w
        q = self.q

        # wTx is the inner product of w and x
        wTx = 0.
        for i in x_w_indices:
            # weight decay if necessary:
            w[i]  = w[i] *((1-self.L2_decay)**(q-decay_times[i]))
            # update decay_times records
            decay_times[i] = q
            # sum up for z term of logistic function
            wTx += w[i]
        # bias from wide side
        wTx += self.b
        z_wide = max(min(wTx, 35.), -35.)
        return z_wide


    def forward(self, X_w_indices, X_d,y_pred,y,training = True):
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
        # fake_input = torch.zeros((X_w_indices.shape[0],1),requires_grad = False, dtype = torch.float32, device = X_d.device)
        # deep_z = self.final_partial_fc(fake_input)

        # ==========Deep + Wide sides
        wide_z = torch.empty((X_w_indices.shape[0],1), requires_grad = False, dtype = torch.float32, device = X_d.device)
        #iterate over training samples within a batch
        for j in range(X_w_indices.shape[0]):
            # forward z prediction:
            wide_z[j] = self.forward_wide(X_w_indices[j,:])
            # overall prediction:
            y_pred[j] = self.activation(wide_z[j])
            # back prop for wide:
            if training:
                self.update(X_w_indices[j,:],y_pred[j],y[j]) # update parameters for wide side


    def update(self,x_wide,y_pred,y):
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
        decay_times = self.decay_times
        w = self.w
        q = self.q
        lr_w = self.lr_w

        # convert y_pred,y  to scaler
        p = y_pred.item()
        y = y.item()

        # gradient under logloss, this is because: if x_i != 0, g = (p-y)x_i= p-y
        g = p - y

        #update w:
        self.q += 1 # everytime we backprop for a training sample, all wide weights should be decayed for one time
        for i in x_wide:
            # update w[i]
            w[i] = (1-self.L2_decay)**(q-decay_times[i]) * w[i] - lr_w*g
            # update decay time for the ith weight
            decay_times[i] = q
        #update bias:
        self.b = self.b - lr_w*g
        

    def decay_catch_up(self):
        '''
        --finish weights decay for all weights on the wide side, this function is called usually in the end of a fitting session to make sure the weights on self.w is up to date
        '''
        for i in range(len(self.w)):
            self.w[i] = (1-self.L2_decay)**(self.q-self.decay_times[i]) *self.w[i]
            self.decay_times[i] = self.q
        return

    def eval_model(self, converter_test,loader_cols, batch_size):
        '''
        This function is called  after each epoch of training on the training data. 
        This function measure the performance of the model using the dev dataset.

        inputs: 
        - test_loader
        outputs:
        - test_loss: test loss for current epoch
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
                    y_pred = torch.empty(y.shape,dtype = y.dtype,requires_grad = False,device = y.device)
                    self(X_w_indices, X_d,y_pred,y,training = False)# y_pred got updated, passed y_pred as arguments
                    loss = self.criterion(y_pred, y)#y.view(-1,1)

                    running_loss += loss.item() * y.size(0)
                    running_num_samples += y.size(0)
                
        # calculating test loss on all dev dataset
        test_loss = running_loss / running_num_samples # avg loss/sample
        # update best test loss so far if necessary
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_model_wts = copy.deepcopy(self.state_dict())
            self.best_n = self.n
            self.best_z = self.z
        print('Current test loss: %.4f. So far best test loss: %.4f' % (test_loss,self.best_test_loss))
        return test_loss

    def fit(self, converter_train, converter_test,batch_interval,loader_cols, n_epochs, batch_size,shuffle_row_groups):
        """Run the model for the training set at dataset.

        Parameters:
        ----------
        - dataset (dict): dictionary with the training sets: X_wide_train, X_deep_train, target
        - n_epochs (int)
        - batch_size (int)
        """

        # evalate the model at the very beginning
        self.eval()
        test_loss = self.eval_model(converter_test, loader_cols, batch_size)
        self.test_loss_history.append(test_loss)
        self.train()
        
        for epoch in range(n_epochs):
            print('======')
            print_time()
            self.train()

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
                    y_pred_leaf = torch.empty(y.shape,dtype = y.dtype,requires_grad = True,device = y.device); 
                    y_pred = y_pred_leaf.clone() # y_pred_leaf is to make y_pred not a leaf variable,so differentiable
                    self(X_w_indices, X_d,y_pred,y)# y_pred got updated, passed y_pred as arguments
                    loss = self.criterion(y_pred, y)#y.view(-1,1)

                    # ----backward (calc gradients) for 'deep'
                    loss.backward()
                    
                    # ---- optimization:update gradient using gradient
                    #--deep:
                    self.optimizer.step()
                    #--wide: get gradient  and get ready for update for 'WIDE', the actual update for 'wide' is in function forward()

                    #----print out loss for current batch if it is multiple of batch_interval
                    running_loss_batch += loss.item() *  y.size(0)
                    running_total_batch += y.size(0)

                    # training performance tracking based on batches...
                    if  i%batch_interval==0 and i!=0:
                        print('-----')
                        print_time()
                        batches_loss = running_loss_batch/running_total_batch
                        self.train_loss_history.append(batches_loss)
                        print("batch {}, avg training loss {} per sample within batches".format(i,round(batches_loss,3)) )
                        running_loss_batch, running_total_batch = 0,0
                        self.eval()
                        test_loss = self.eval_model(converter_test, loader_cols, batch_size)
                        self.test_loss_history.append(test_loss)
                        self.train()

            # training performance tracking based on epochs
            # print('-----')
            # print_time()
            # batches_loss = running_loss_batch/running_total_batch
            # self.train_loss_history.append(batches_loss)
            # print("batch {}, avg training loss {} per sample within batches".format(i,round(batches_loss,3)) )
            # running_loss_batch, running_total_batch = 0,0
            # self.eval()
            # test_loss = self.eval_model(converter_test, loader_cols, batch_size)
            # self.test_loss_history.append(test_loss)
            # self.train()
        return 