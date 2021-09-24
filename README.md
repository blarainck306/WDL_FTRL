[Wide & Deep learning](https://arxiv.org/pdf/1606.07792.pdf) in Pytorch. The idea of wide and deep learning is to combine the memorization of wide model and the genelization of deep model into a single joint model. Please see the details of the structure of the wide and deep learning in  [this paper from Google](https://arxiv.org/pdf/1606.07792.pdf).  
[![Foo](https://1.bp.blogspot.com/-Dw1mB9am1l8/V3MgtOzp3uI/AAAAAAAABGs/mP-3nZQCjWwdk6qCa5WraSpK8A7rSPj3ACLcB/s1600/image04.png)](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
## Wide & Deep Learning in Pytorch
The Wide & Deep Learning model was developed for the prediction of CTR of [this dataset](https://www.kaggle.com/c/avazu-ctr-prediction/overview). The implementation is inspired by [this Github repository](https://github.com/zenwan/Wide-and-Deep-PyTorch). Lots of newly added features for this project include:
- using a special Pytorch data loader via Petastorm library to load data preprocessed by Spark required by each batch from data files in parquet storage format. 
- manually coded the customized forward and backward propagation, optimization algorithm using either OGD or FTRL-proximal for the optimization of the wide part of the model in Python and Pytorch. THe purpose is to utilize the sparsity of the wide feastures to make the training much more efficient. THis is crutial when training the models with big data input. Because the input data to the wide part is 9.5 million dimensional with only 20-30 non-zero features.
- Implemented the wide part of the model to exploit the feature sparsity of the wide input. This is to significantly improve the training speed of the model. The implementation includes the forward and backward progation, optimization algorithm using ongline gradient decent (OGD) or FTRL-proximal. This implementation improves the training speed in a few order of magnitude. And this can seen from the fact that input data to the wide part is 9.5 million dimensional with only 20-30 non-zero entries.
- added the functionality for processing hashed wide features using the [hashing trick](https://alex.smola.org/papers/2009/Weinbergeretal09.pdf)
- separated wide and deep models for training purposes
- added regularzation (L2 and dropout) on the deep part of the model
- added regularization on the wide part of the model
- added batch normalization layer to the deep part of the model
- enabled saving model states/weights, so that close monitoring of the model trainign is possible.
- added better performance training tracking by enabling both epch-based and batch-based intervals.

## Branches
- The 6 branches in this repository are used for training, the branch names are self-explanatory. Among them, 3 branches for wide model with the optimization algorithm of online gradient decent
  - OGD_wideOnly: wide mode only trained using optimization algorithm OGD/SGD
  - OGD_deepOnly: deep model
  - OGD_wideDeep: joint training

; while the other 3 branches for wide model with Follow the Regularized Leader-Proximal (FTRL-proximal)
  - FTRL_wideOnly: wide mode only trained using optimization algorithm  FTRL-proximal
  - FTRL_deepOnly: deep model, almost the same as OGD_deepOnly, it's here for the easiness of combing both wide and deep models in joint training.
  - FTRL_wideDeep: joint training

## Files
- WDL_FTRL_model.py: Wide & Deep Learning in Pytorch with addiontal customized functionality introduced above. 
- utility.py: utlity functions used by the project to be introduced in the next setion. 
- 4_OGD_L2Wide_Combine_addRegu.ipynb: an demo showing the training of a wide & deep model applied to CTR. For the complete scripts including the Jupyter notebook for data ETL, EDA, data preprocessing, data engineering, model training, please refer to this [report](https://docs.google.com/document/d/1bQNWil_nIA_X1sCEoWekLO7SbP3kt6H10hy3DrLovSw/edit?usp=sharing).



## Wide & Deep Learning applied to CTR
The project for Click-through rate (CTR) prediction using this model is introduced in [this Google Doc](https://docs.google.com/document/d/1bQNWil_nIA_X1sCEoWekLO7SbP3kt6H10hy3DrLovSw/edit?pli=1#).
The Colab notebook for data ETL, EDA, data preprocessing, data engineering, model training are in [Google Drive](https://drive.google.com/drive/folders/1zc4k-YZDNHmzNihtHZOjdBArLwfL-DO8?usp=sharing).

