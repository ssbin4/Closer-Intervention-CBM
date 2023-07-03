# Dataset preprocessing

Download the images from [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) and save them in data/fitz/ directory.

Download the annotations_fitzpatrick17k.csv file from [SKINCON](https://skincon-dataset.github.io/) and save it in data/ directory.

Obtain the pkl files with the binary class labels using the following command.
```
python3 data_processing.py -save_dir ${save_dir} -data_dir data/ -class_label 'binary'
```

For the experiments, we only use the 22 concepts which are present in at least 50 images.
By the following command, we generate the new data existing in 'modify_data_dir' into 'data_dir' directory.
```
python3 generate_new_data.py --exp Concept --data_dir ${data_dir} --modify_data_dir ${modify_data_dir}
```


# Training the models

Download the pretrained models from [Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set](https://drive.google.com/drive/folders/1WscikgfyQWg1OTPem_JZ-8EjbCQ_FHxm).

The training code is based on [Concept Bottleneck Models](https://github.com/yewsiang/ConceptBottleneck).

The following command is an example to train the XtoC model with the data saved in 'data/22concepts/binary' directory.

```
python3 ../experiments.py skincon Concept_XtoC -seed ${seed} -ckpt 1 -log_dir ${log_dir} -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir data/22concepts/ -n_attributes 22 -normalize_loss -b 64  -weight_decay ${weight_decay} -lr ${init_lr} -scheduler_step ${step} -bottleneck -class_label binary
```

# Test-time intervention

You can use the similar command as in the CUB.