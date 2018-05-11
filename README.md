# slim_cnn_test
Use tensorflow.contrib.slim to training a simple CNN classification model

## Usage:
1.1 git clone this project

1.2 create a directory ./datasets/images/

1.3 generate training images:
```
python3 generate_train_data.py
```

1.4 generate tfrecord file:
```
python3 generate_tfrecord.py \
    --images_path ./datasets/images/ \
    --output_path ./datasets/train.record
```
        
1.5 create a directory ./training/

1.6 train CNN model:
```
python3 train.py \
    --record_path ./datasets/train.record \
    --logdir ./training/
```
        
1.7 visulize the loss curves:
```
tensorboard --logdir /home/.../training/
```

1.8 export frozen inference graph:
```
python3 export_inference_graph.py \
    --input_type image_tensor \
    --trained_checkpoint_prefix ./training/model.ckpt-xxx(num_steps) \
    --output_directory path/to/exported_model_directory
```
    
1.9 evaluate the trained model:
```
python3 evaluate.py \
    --frozen_graph_path path/to/exported_model_directory/frozen_inference_graph.pb
```
