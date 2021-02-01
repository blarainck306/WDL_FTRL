# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import copy
import time

use_cuda = torch.cuda.is_available()

from petastorm import TransformSpec
def get_transform_spec():
  return TransformSpec(func = None, selected_fields = ['features', 'label'])

def print_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    
class WideDeepLoader(Dataset):
    """Helper to facilitate loading the data to the pytorch models.

    Parameters:
    --------
    data: namedtuple with 3 elements - (wide_input_data, deep_inp_data, target)
    """
    def __init__(self, data):

        self.X_wide = data.wide
        self.X_deep = data.deep
        self.Y = data.labels

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y  = self.Y[idx]

        return xw, xd, y

    def __len__(self):
        return len(self.Y)


class WideDeep(nn.Module):
    """ Wide and Deep model. As explained in Heng-Tze Cheng et al., 2016, the
    model taked the wide features and the deep features after being passed through
    the hidden layers and connects them to an output neuron. For details, please
    refer to the paper and the corresponding tutorial in the tensorflow site:
    https://www.tensorflow.org/tutorials/wide_and_deep

    Parameters:
    --------
    - wide_dim (int) : dim of the wide-side input tensor
    - embeddings_input (tuple): 3-elements tuple with the embeddings "set-up" -
    (col_name, unique_values, embeddings dim)
    - continuous_cols (list) : list with the name of the continuum columns
    - deep_column_idx (dict) : dictionary where the keys are column names and the values
    their corresponding index in the deep-side input tensor
    - hidden_layers (list) : list with the number of units per hidden layer
    - encoding_dict (dict) : dictionary with the label-encode mapping
    - n_class (int) : number of classes. Defaults to 1 if logistic or regression
    - dropout (float)
    """

    def __init__(self,
                 wide_dim,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 batch_norm,
                 dropout,
                 n_class):

        super(WideDeep, self).__init__()
        self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        # self.encoding_dict = encoding_dict
        self.n_class = n_class

        if self.hidden_layers:
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

            # -- following hidden layer, and dropout layers (if specified)
            for i,h in enumerate(self.hidden_layers[1:],1):
                setattr(self, 'linear_'+str(i), nn.Linear( self.hidden_layers[i-1], self.hidden_layers[i] , bias = not  self.batch_norm))
                if self.batch_norm:
                    setattr(self, 'bn_'+str(i), nn.BatchNorm1d(self.hidden_layers[i]))
                if self.dropout:
                    setattr(self, 'linear_'+str(i)+'_drop', nn.Dropout(self.dropout[i]))

            # Connect the wide- and dee-side of the model to the output neuron(s)
            self.output = nn.Linear(self.hidden_layers[-1]+self.wide_dim, self.n_class)

        else:
            self.output = nn.Linear(self.wide_dim, self.n_class)


    def compile(self, method="logistic", optimizer="Adam", learning_rate=0.001, momentum=0.0):
        """Wrapper to set the activation, loss and the optimizer.

        Parameters:
        ----------
        method (str) : regression, logistic or multiclass
        optimizer (str): SGD, Adam, or RMSprop
        """
        if method == 'regression':
            self.activation, self.criterion = None, F.mse_loss
        if method == 'logistic': # Avazu
            self.activation, self.criterion = F.sigmoid, F.binary_cross_entropy
        if method == 'multiclass':
            self.activation, self.criterion = F.softmax, F.cross_entropy

        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.method = method


    def forward(self, X_w, X_d):
        """Implementation of the forward pass.

        Parameters:
        ----------
        X_w (torch.tensor) : wide-side input tensor
        X_d (torch.tensor) : deep-side input tensor

        Returns:
        --------
        out (torch.tensor) : result of the output neuron(s)
        """
        # =========Deep Side
        # ---- get embedding concatenated with continuous features: 'deep_inp'
        # -- get embeddings for embedding features: emb (a list of embeddings, one embedding for each feature respectively)
        if self.hidden_layers:
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
            wide_deep_input = torch.cat([x_deep, X_w.float()], 1)
            # self.output: linear function
            # self.activation: sigmoid function for binary classification

            out = self.activation(self.output(wide_deep_input))

        else:
            wide_deep_input = X_w.float()
            out = self.activation(self.output(wide_deep_input))
        return out


    def eval_model(self, converter_test_loader, best_loss,best_model_wts,batch_size=32):
        '''
        This function is called  after each epoch of training on the training data. 
        This function measure the performance of the model using the dev or test data.

        inputs: 
        - test dataset
        outputs:
        - test_loss: test loss for current epoch
        - best_loss: best loss so far
        - best_model_wts: best model weights so far
        '''
        running_loss = 0.0
        running_num_samples = 0.0

        with torch.no_grad():
            with converter_test_loader.make_torch_dataloader(batch_size = batch_size, transform_spec = get_transform_spec(),num_epochs = 1,shuffle_row_groups = False, workers_count = 2) as test_loader:
                # for i, (X_wide, X_deep, target) in enumerate(test_loader):
                for i, row_batch in enumerate(test_loader):
                    X_wide = row_batch['features'][:,:self.wide_dim]
                    X_deep = row_batch['features'][:,self.wide_dim:]
                    target = row_batch['label']

                    X_w = X_wide
                    X_d = X_deep
                    y = (target.float() if self.method != 'multiclass' else target)
                    if use_cuda:
                        X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()
                    # forward:
                    y_pred = self(X_w, X_d)
                    loss = self.criterion(y_pred, y.view(-1,1))

                    running_loss += loss.item() * y.size(0)
                    running_num_samples += y.size(0)

        test_loss = running_loss / running_num_samples # avg loss/sample
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(self.state_dict())
        print('Current test loss: %.4f. So far best test loss: %.4f' % (test_loss,best_loss))
        return test_loss, best_loss, best_model_wts


    def fit(self, converter_train_loader,best_model_wts,best_loss, converter_test_loader,batch_interval, n_epochs = 10, batch_size = 32):
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
        test_loss, best_loss, best_model_wts = self.eval_model(converter_test_loader, best_loss,best_model_wts,batch_size)
        test_loss_history.append(test_loss)
        
        for epoch in range(n_epochs):
            print('-----')
            self.train()
            running_loss = 0
            running_total=0
            running_correct=0

            running_loss_batch = 0
            running_total_batch=0

            # note rows of data will be expressed as a dictionary, with column names as the keys, similar to pandas
            with converter_train_loader.make_torch_dataloader(batch_size = batch_size, transform_spec = get_transform_spec(),num_epochs = 1,shuffle_row_groups = False, workers_count = 2) as train_loader:
                # for i, (X_wide, X_deep, target) in enumerate(train_loader):
                for i, row_batch in enumerate(train_loader):
                    X_wide = row_batch['features'][:,:self.wide_dim]
                    X_deep = row_batch['features'][:,self.wide_dim:]
                    target = row_batch['label']
                    # ----get the inputs and assgin them to cuda if any
                    X_w = X_wide
                    X_d = X_deep
                    y = (target.float() if self.method != 'multiclass' else target)
                    if use_cuda:
                        X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()

                    
                    self.optimizer.zero_grad()
                    # ----forward
                    y_pred =  self(X_w, X_d)
                    loss = self.criterion(y_pred, y.view(-1,1))

                    # ----backward
                    loss.backward()

                    # ---- optimization
                    self.optimizer.step()

                    # ----record 'running_total','running_correct','running_loss'
                    if self.method != "regression":
                        running_total+= y.size(0)
                        if self.method == 'logistic':
                            y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                        if self.method == "multiclass":
                            _, y_pred_cat = torch.max(y_pred, 1)
                        running_loss += loss.item() *  y.size(0)
                        running_correct += float((y_pred_cat == y).sum().data.item())

                    #----print out loss for current batch if i is multiple of batch_interval
                    running_loss_batch += loss.item() *  y.size(0)
                    running_total_batch += y.size(0)
                    if i!=0 and i %batch_interval ==0:
                        batches_loss = running_loss_batch/running_total_batch
                        print("batch {}, avg loss {} per sample within batches".format(i,round(batches_loss,3)) )
                        print_time()

                        running_loss_batch, running_total_batch = 0,0
                        

                    

            # ------print out training loss, accuracy for each epoch
            if self.method != "regression":
                epoch_loss = running_loss / running_total
                print ('Epoch {} of {}, Training Loss: {}'.format(epoch+1, n_epochs, round(epoch_loss,3)) )
            else:
                print ('Epoch {} of {}, Loss: {}'.format(epoch+1, n_epochs,
                    round(loss.data.item(),3)))
            train_loss_history.append(epoch_loss)

            # ------at each epoch, evaluate the trained model using the test data
            self.eval()
            test_loss, best_loss, best_model_wts = self.eval_model(converter_test_loader, best_loss,best_model_wts, batch_size)
            test_loss_history.append(test_loss)
            print_time()


        return train_loss_history, test_loss_history, best_loss, best_model_wts




    def predict(self, dataset):
        """Predict target for dataset.

        Parameters:
        ----------
        dataset (dict): dictionary with the testing dataset -
        X_wide_test, X_deep_test, target

        Returns:
        --------
        array-like with the target for dataset
        """

        X_w = Variable(torch.from_numpy(dataset.wide)).float()
        X_d = Variable(torch.from_numpy(dataset.deep))

        if use_cuda:
            X_w, X_d = X_w.cuda(), X_d.cuda()

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w,X_d).cpu()
        if self.method == "regression":
            return pred.squeeze(1).data.numpy()
        if self.method == "logistic":
            return (pred > 0.5).squeeze(1).data.numpy()
        if self.method == "multiclass":
            _, pred_cat = torch.max(pred, 1)
            return pred_cat.data.numpy()


    def predict_proba(self, dataset):
        """Predict predict probability for dataset.
        This method will only work with method logistic/multiclass

        Parameters:
        ----------
        dataset (dict): dictionary with the testing dataset -
        X_wide_test, X_deep_test, target

        Returns:
        --------
        array-like with the probability for dataset.
        """

        X_w = Variable(torch.from_numpy(dataset.wide)).float()
        X_d = Variable(torch.from_numpy(dataset.deep))

        if use_cuda:
            X_w, X_d = X_w.cuda(), X_d.cuda()

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w,X_d).cpu()
        if self.method == "logistic":
            pred = pred.squeeze(1).data.numpy()
            probs = np.zeros([pred.shape[0],2])
            probs[:,0] = 1-pred
            probs[:,1] = pred
            return probs
        if self.method == "multiclass":
            return pred.data.numpy()

'''
    def get_embeddings(self, col_name):
        """Extract the embeddings for the embedding columns.

        Parameters:
        -----------
        col_name (str) : column we want the embedding for

        Returns:
        --------
        embeddings_dict (dict): dictionary with the column values and the embeddings
        """

        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = self.encoding_dict[col_name]
        inv_dict = {v:k for k,v in col_label_encoding.iteritems()}
        embeddings_dict = {}
        for idx,value in inv_dict.iteritems():
            embeddings_dict[value] = embeddings[idx]

        return embeddings_dict

'''
