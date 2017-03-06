# Aggregating deep features for visually similar image search

Network used: VGG16  
Layer: after pool5  
Aggregated by: summing over the spatial activation in each channel  

#### Requirements
Keras with Tensorflow as backend. Information on installing Keras can 
be found [here](https://keras.io/#installation)

#### Get Oxford dataset
```
cd data
bash get_data.sh
```

#### The ipython notebook `crow-full-size-paris-whiten-caffe
The ipython notebook `crow-full-size-paris-whiten-caffe contains the 
final version of code. The features extracted from Paris dataset was
pre-computed and included in the repository.