# UnifiedQA - CycIC

This repo is a fork of [allenai/unifiedqa](https://github.com/allenai/unifiedqa), with added scripts for converting a CycIC question/answer dataset to the unified QA format and for running a T5-based unifiedQA model against the data.

## Encoding a CycIC dataset

The script `encode_cycic.py` assumes that the data files follow the naming convention `<dev|test|training>_<questions|labels>.jsonl`, for example `training_questions.jsonl`. To encode a cycic dataset at location `./cycic3/` and output it to `./encoded/`, you would use the command:
```
python3 encode_cycic.py --data_dir cycic3 --output_dir encoded
```

## Running the UnifiedQA T5 model:

The scripts under `t5/` simply replicate the behavior of the BART code from the original repo, but with support for T5-based models. To generate predictions for an encoded file using the small pretrained model, you would run:
```
python3 t5/cli_t5.py \
--do_predict \
--model_name 'allenai/unifiedqa-t5-small' \
--output_dir prediction
--predict_batch_size 8
```
