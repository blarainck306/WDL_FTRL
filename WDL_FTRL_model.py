# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
from math import exp,sqrt

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
    - embeddings_input (tuple): 3-elements tuple with the embeddings "set-up" -
    (col_name, unique_values, embeddings dim)
    - continuous_cols (list) : list with the name of the continuum columns
    - deep_column_idx (dict) : dictionary where the keys are column names and the values
    their corresponding index in the deep-side input tensor
    - hidden_layers (list) : list with the number of units per hidden layer
    - encoding_dict (dict) : dictionary with the label-encode mapping
    - n_class (int) : number of classes. Defaults to 1 if logistic or regression
    - dropout (float)

    ====WIDE
    - D : similar to wide_dim, number of weights (including bias) to use for the "wide" part
    - interaction: whether to use interaction in the "wide" part
    """

    def __init__(self,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 batch_norm,
                 dropout,
                 n_class,
                 num_wide_features,
                 num_deep_features,
                 ordered_wide_cols,
                 D,
                 interaction = False):

        super(WideDeep, self).__init__()
        # self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        # self.encoding_dict = encoding_dict
        self.n_class = n_class
        self.num_wide_features = num_wide_features
        self.num_deep_features = num_deep_features
        self.wide_cols = ordered_wide_cols


        # ----Build the embedding layers:  to be passed through the deep-side
        for col,val,dim in self.embeddings_input:
            setattr(self, 'emb_layer_'+col, nn.Embedding(val, dim))

        # ----Build the deep-side hidden layers (with dropout if specified)
        input_emb_dim = np.sum([emb[2] for emb in self.embeddings_input])
        # --1st hidden layer, 1st dropout
        self.linear_0 = nn.Linear(input_emb_dim+len(continuous_cols), self.hidden_layers[0], bias = not  self.batch_norm)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm1d(num_features = self.hidden_layers[0])
        if self.dropout:
            self.linear_0_drop = nn.Dropout(self.dropout[0])

        # --- following hidden layer, and dropout layers (if specified)
        for i,h in enumerate(self.hidden_layers[1:],1):
            setattr(self, 'linear_'+str(i), nn.Linear( self.hidden_layers[i-1], self.hidden_layers[i] , bias = not  self.batch_norm))
            if self.batch_norm:
                setattr(self, 'bn_'+str(i), nn.BatchNorm1d(self.hidden_layers[i]))
            if self.dropout:
                setattr(self, 'linear_'+str(i)+'_drop', nn.Dropout(self.dropout[i]))

        # PART of FC layer for deep side only, the othre half is with "WIDE", implemented outside Pytorch
        self.final_partial_fc = nn.Linear(self.hidden_layers[-1], self.n_class)



        #=========wide part:
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}
        self.w_array = []
        self.interaction = interaction

        # ---feature related parameters
        self.D = D

    def compile(self, optimizer, learning_rate, momentum, alpha, beta, L1, L2,method="logistic"):
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

        if optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.method = method

    def get_transform_spec(self,loader_cols):
        return TransformSpec(func = None, selected_fields = loader_cols)


    def indexing(self,X_w):
        X_w_indices = np.ones(X_w.shape,dtype = np.int32)
        for i in range(X_w.shape[0]):
            for j in range(X_w.shape[1]):
                X_w_indices[i,j] = abs(hash(self.wide_cols[j]+'_'+ str(X_w[i,j].item()))) % self.D
        return X_w_indices

    def indices_inter(self, x):
        ''' 
        - x: a list of index of hashed features
        A helper generator that yields the indices implied by x

        - The purpose of this generator is to make the following
        code a bit cleaner when doing feature interaction.
        - this function keeps value for x unchanged, mainly enerate hashing feature index of the interactive features
        '''

        # first yield index of the bias term,note hash(0)=0 in python, but assume we don't have  zero value in the the feature values for wide features.
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

    def forward_wide(self,x_w_indices):
        '''
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
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        # iterate through all non-zero feature values:
        for i in self.indices_inter(x_w_indices):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w
        self.w_array.append(w)
        z_wide = max(min(wTx, 35.), -35.)
        # bounded sigmoid function, this is the probability estimation
        # return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))
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
        emb = [getattr(self, 'emb_layer_'+col)(X_d[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
       
        # -- cont is for continuous features
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X_d[:, cont_idx].float()]
            deep_inp = torch.cat(emb+cont, 1)
        else:
            deep_inp = torch.cat(emb, 1)

        # ---- 1st hidden layer and dropout layers
        #linear+(batch norm)+ relu
        if not self.batch_norm:
            x_deep = F.relu(self.linear_0(deep_inp))
        else:
            x_deep = F.relu(self.bn0(self.linear_0(deep_inp)))
        # dropout
        if self.dropout:
            x_deep = self.linear_0_drop(x_deep)

        # ---- following hidden layers and dropout layers
        for i in range(1,len(self.hidden_layers)):
            #linear+(batch norm)+ relu
            if not self.batch_norm:
                x_deep = F.relu( getattr(self, 'linear_'+str(i))(x_deep) )
            else:
                x_deep = F.relu( getattr(self,'bn_'+str(i))(getattr(self, 'linear_'+str(i))(x_deep) ) )
            # dropout
            if self.dropout:
                x_deep = getattr(self, 'linear_'+str(i)+'_drop')(x_deep)

        # ==========Deep + Wide sides
        # deep
        deep_z = self.final_partial_fc(x_deep)
        # wide
        wide_z = torch.empty(deep_z.shape, requires_grad = False, dtype = deep_z.dtype, device = deep_z.device)
        self.w_array = []
        for j in range(X_w_indices.shape[0]):
            wide_z[j] = self.forward_wide(X_w_indices[j,:])

        y_pred = self.activation(wide_z + deep_z)
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
        x_wide (iterable):  (num_wide_features,), feature indices on wide side
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
        w = self.w_array[j]

        # gradient under logloss, this is because: if x_i != 0, g = (p-y)x_i= p-y
        g = p - y

        # update z and n, for x_i = 0, gradient is zero, so no update for them
        for i in self.indices_inter(x_wide):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

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
                for i, row_dic in enumerate(test_loader):
                    # decode features
                    X_batch = row_dic['all_features']
                    y = row_dic['label']
                    X_w = X_batch[:,:self.num_wide_features]
                    X_d = X_batch[:,-self.num_deep_features:]
                    X_w_indices = self.indexing(X_w) # X_w_indices: np.array

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


    def fit(self, converter_train,best_model_wts, best_loss, converter_test,batch_interval,loader_cols, n_epochs, batch_size):
        """Run the model for the training set at dataset.

        Parameters:
        ----------
        - dataset (dict): dictionary with the training sets: X_wide_train, X_deep_train, target
        - n_epochs (int)
        - batch_size (int)
        """
        train_loss_history = []
        test_loss_history = []

        # evalate the model at the very beginning
        self.eval()
        test_loss, best_loss, best_model_wts = self.eval_model(converter_test, best_loss, best_model_wts,loader_cols,batch_size)
        test_loss_history.append(test_loss)
        
        for epoch in range(n_epochs):
            print('-----')
            self.train()
            # for epoch training performance tracking
            running_loss = 0
            running_total=0
            # for batch performance tracking
            running_loss_batch = 0
            running_total_batch=0

            # petastorm loader: note rows of data will be expressed as a dictionary, with column names as the keys
            with converter_train.make_torch_dataloader(batch_size = batch_size, transform_spec = self.get_transform_spec(loader_cols),num_epochs = 1,shuffle_row_groups = False, workers_count = 2) as train_loader:
                for i, row_dic in enumerate(train_loader):
                    # decode features
                    X_batch = row_dic['all_features']
                    y = row_dic['label']
                    X_w = X_batch[:,:self.num_wide_features]
                    X_d = X_batch[:,-self.num_deep_features:]
                    X_w_indices = self.indexing(X_w) # X_w_indices: np.array

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
                    running_total+= y.size(0)
                    running_loss += loss.item() *  y.size(0)

                    #----print out loss for current batch if i is multiple of batch_interval
                    running_loss_batch += loss.item() *  y.size(0)
                    running_total_batch += y.size(0)
                    if  i%batch_interval==0 and i!=0 :
                        batches_loss = running_loss_batch/running_total_batch
                        print("batch {}, avg training loss {} per sample within batches".format(i,round(batches_loss,3)) )
                        print_time()
                        running_loss_batch, running_total_batch = 0,0
                        self.eval()
                        test_loss, best_loss, best_model_wts = self.eval_model(converter_test, best_loss,best_model_wts, loader_cols, batch_size)
                        self.train()
                    
            # ------print out training loss, accuracy for each epoch
            epoch_loss = running_loss / running_total
            print ('Epoch {} of {}, Training Loss: {}'.format(epoch+1, n_epochs, round(epoch_loss,3)) )
            train_loss_history.append(epoch_loss)

            # ------at each epoch, evaluate the trained model using the test data
            self.eval()
            test_loss, best_loss, best_model_wts = self.eval_model(converter_test, best_loss,best_model_wts, loader_cols, batch_size)
            test_loss_history.append(test_loss)
            print_time()


        return train_loss_history, test_loss_history, best_loss, best_model_wts
