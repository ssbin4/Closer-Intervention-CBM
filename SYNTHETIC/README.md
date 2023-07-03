# Generating the data

To generate the synthetic data with input noise, use the following command.

```
python3 data.py -exp GenData -out_dir ${data_dir} -alpha_mean ${alpha_mean} -alpha_var ${alpha_var} -z_var ${z_var} -input_dim ${input_dim} -n_attributes ${n_attributes} -n_classes ${n_classes} -n_groups ${n_groups}
```

You can generate the data with hidden concepts using the following command where 'data_dir' is directory for the previously generated dataset.
```
python3 data.py -exp 'Hidden' -data_dir ${data_dir} -out_dir ${hidden_data_dir} -hidden_ratio ${hidden_ratio}
```

# Training the models

The following command is an example to train XtoC model.

```
python3 ../experiments.py synthetic Concept_XtoC --seed ${seed} -ckpt '' -log_dir ${log_dir} -e 1000 -optimizer sgd -use_attr -weighted_loss multiple -data_dir ${data_dir} -n_attributes ${n_attributes} -input_dim ${input_dim} -n_classes ${n_classes} -normalize_loss -b 64  -weight_decay ${reg} -lr ${init_lr} -scheduler_step ${step} -bottleneck -expand_dim 100 -input_dim ${input_dim} -n_attributes ${n_attributes} -n_classes ${n_classes}
```

# Test-time intervention

You can use the similar command as in the CUB.