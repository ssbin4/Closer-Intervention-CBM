# CUB Dataset
## Dataset preprocessing and training the models

Our code is based on [Concept Bottleneck Models](https://github.com/yewsiang/ConceptBottleneck).
Please check the above repository for dataset preprocessing and training the models.

## Test-time intervention

### Basic usage
Conduct test-time intervention for independent (IND) models using the following command.
```
python3 tti.py -model_dirs ${model_dirs} -model_dirs2 ${model_dirs2} -use_attr -bottleneck -criterion ${criterion} -level {level} -inference_mode ${inference_mode} -n_trials 1 -n_attributes 112 -data_dir2 CUB_processed/CUB_raw -data_dir CUB_processed/class_attr_data_10 -use_sigmoid -class_level -use_invisible -no_intervention_when_invisible 
```

- Criterion is one of ['rand', 'ucp', 'lcp', 'cctp', 'ectp', 'edutp'].
- Intervention level should be represented as one of the ['i+s', 'i+b', 'g+s', 'g+b'].
- Inference mode represents the conceptualization strategy, e.g., 'soft', 'hard', or 'samp'.

### Other training strategies
- SEQ
    ```
    python3 tti.py -model_dirs ${model_dirs} -model_dirs2 ${model_dirs2} -use_attr -bottleneck -criterion ${criterion} -level {level} -inference_mode ${inference_mode} -n_trials 1  -n_attributes 112 -data_dir2 CUB_processed/CUB_raw -data_dir CUB_processed/class_attr_data_10 -use_invisible -class_level -no_intervention_when_invisible
    ```
- JNT
    ```
    python3 tti.py -model_dirs ${model_dirs} -use_attr -bottleneck -criterion ${criterion} -level {level} -inference_mode ${inference_mode} -n_trials 1  -n_attributes 112 -data_dir2 CUB_processed/CUB_raw -data_dir CUB_processed/class_attr_data_10 -use_invisible -class_level -no_intervention_when_invisible
    ```
- JNT+P
    ```
    python3 tti.py -model_dirs ${model_dirs} -use_attr -bottleneck -criterion ${criterion} -level {level} -inference_mode ${inference_mode} -n_trials 1  -n_attributes 112 -data_dir2 CUB_processed/CUB_raw -data_dir CUB_processed/class_attr_data_10 -use_invisible -use_sigmoid  -class_level -no_intervention_when_invisible
    ```

### Comparison between different intervention levels

For the group-wise interventions, comparison values to individual intervention for Figure 5-(b) and Figure 18 are printed at the last lines.

## Cost of intervention

Calculate mean labeling time used for the experiments in Appendix C.
```
python3 generate_new_data.py --exp Cost
```
