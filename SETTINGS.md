## Data Dictionary

This table contains the attributes from the original ProtoPNet configurations.
These names are being adjusted through the creation of ProtoPNeXt, but they are a useful reference.

| Attribute Name | Type |        Description      |
| ------------- |:-------------:|  --------------------------:|
| val_batch_size | int | Batch size for validation set loader in main.py |
| base_architecture | str | Base architecture for backbone in main.py (Must be VGG, ResNet, or DenseNet) |
| data_path | str | Path to data to train/validate/test the model on |
| img_size | str | Size of images being passed into network |
| train_batch_size | int | Batch size for train loader in main.py |
| experiment_run | int | Name of experiment being run |
| warm_optimizer_lrs["prototype_vectors"] | int | Learning rate for the prototype vectors during the warming stage. |
| warm_optimizer_lrs["add_on_layers"] | int | Learning rate for the add-on layers during the warming stage. |
| prototype_shape | list | Shape of prototypes tensor: total number x number of features x width x height. |
| warm_optimizer_lrs["conv_offset"] | int | Learning rate for the convolutional offset layers during the warming stage. |
| warm_optimizer_lrs["features"] | int | *Learning rate for features during the warming stage. |
| num_warm_epochs | int | Number of warming epochs to be done before joint training. |
| joint_optimizer_lrs["joint_last_layer_lr"] | int | Learning rate for the last layer during joint training. |
| joint_optimizer_lrs["prototype_vectors"] | int | Learning rate for the prototype vectors during joint training. |
| joint_optimizer_lrs["conv_offset"] | int | Learning rate for the convolutional offset layers during joint training. |
| joint_optimizer_lrs["features"] | int | *Learning rate for the features. |
| joint_optimizer_lrs["add_on_layers"] | int | Learning rate for the add-on layers during the warming stage. |
| prototype_activation_function | str | *  |
| subtractive_margin | bool | Whether or not to use a subtractive margin when calculating prototype activations. |
| push_start | int | The first epoch at which you push the prototype vectors. Should be > num_warm_epochs. |
| last_layer_fixed | bool | Whether or not the last layer should be fixed, or able to learn weights. We recommend this to be set to True. |
| warm_pre_offset_optimizer_lrs["prototype_vectors"] | int | The learning rate for the prototype vectors during the second warming stage.  |
| warm_pre_offset_optimizer_lrs["add_on_layers"] | int | The learning rate for the add-on layers during the second warming stage.  |
| warm_pre_offset_optimizer_lrs["features"] | int | The learning rate for the features during the second warming stage.|
| train_push_dir | str | The directory that contains examples that prototypes should be pushed to. We recommend this to be the same as train_dir. |
| joint_lr_step_size | int | The step size for the joint layer optimizer. |
| add_on_layers_type | str | The structure of the add-on layers for the CNN backbone. This is usually "bottleneck", "identity", or "upsample". |
| num_classes | int | The number of classes you are classifying between. |
| push_epochs | list | The list of epochs in which you should push prototypes. By default this is every 10. |
| num_train_epochs | int | The number of total train epochs. Note that you should always push on the last epoch, and it is 0 indexed. |
| train_dir | str | The file directory that contains the training set. |
| train_push_batch_size | int | The batch size for the train push step.  |
| num_secondary_warm_epochs | int | The number of epochs dedicated to the second warm stage. |
| val_dir | str | The directory containing validation set data. |
| last_layer_optimizer_lr | int | The learning rate for the last layer optimizer. |
| coefs["clst"] | int | The coefficient for the clustering loss. |
| coefs["offset_weight_l2"] | int | * |
| coefs["sep"] | int | The coefficient for the separation loss. |
| coefs["orthogonality_loss"] | int | The coefficient for the orthogonality loss. |
| coefs["offset_bias_l2"] | int | The coefficient for the offset distance loss. |
| coefs["l1"] | int | The coefficient for L1 regularization loss on the last layer. |
| coefs["crs_ent"] | int | The coefficient for the cross-entropy loss. |
| gpuid | str | List of GPU ids for running the model on multiple GPUs. |
| m | int | The actual value of the subtractive margin. |
| using_deform | bool | Whether or not the prototypes will be deformable. |
| topk_k | int | The k number of instances to look at when performed top-k pooling over the activations. |
| deformable_conv_hidden_channels | int | The number of hidden channels in the network designed to find the offsets. |
| num_prototypes | int | The total number of prototypes across all classes. |
| dilation | int | ** |
| incorrect_class_connection | int | The weight in the final layer assigned to a connection between incorrect classes. |
| rand_seed | int | A specific random seed to be used. |
| finer_coeff | int | The coefficient for the fine-annotation loss. |
| fa_type | str | The kind of fine-annotation loss that will be calculated; one of "square", "serial", or "l2_norm". |
| finer_dir | str | A separate directory for finely annotated data. |
| save_path | str | The local path that the model will be saved to. |
| num_workers | int | Number of workers for data loading (parallelization). |
| prefetch_factor | int | Prefetch factor for data loading (number of batches prefetched in memory ahead of time). |
| class_specific | bool | Flag for class-specific (recommended True). Separation loss not calculated if not class_specific |

## Directory Naming

| Attribute Name | Type |        Description      |
| ------------- |:-------------:|  --------------------------:|
| data_path | str | The path that contains ALL of the data subfolders (train, val, test, etc.) |
| val_dir | str | The directory containing validation data. |
| train_push_dir | str | The directory containing the data you wish to push onto. |
| gpuid | str | A list of GPU ids for running multiple GPUs at once. |
| modeldir | str | The directory the model is stored in. |
| model | str | The file path to the model you wish to analyze. |
| imgdir | str | The directory of images to be analyzed. |
| img | str | The filename of the image to be analyzed. |
| imgclass | str | The class of the image. |
| savedir | str | The directory to save output to. |

