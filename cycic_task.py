import logging
import t5
import os
import json
import functools
import tensorflow as tf
import tensorflow_datasets as tfds

# goal: create a seqio Task for the CycIC dataset

DATA_DIR = f"gs://cycic3/encoded/"

DATASETS = ["cycic3_a"]

def dataset_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
        return {
            "inputs": normalize_text(ex["inputs"]),
            "targets": normalize_text(ex["targets"])
        }

    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

def get_path(data_dir1, split):
    tsv_path = {
        "train": os.path.join(data_dir1, "train.tsv"),
        "dev": os.path.join(data_dir1, "dev.tsv"),
        "test": os.path.join(data_dir1, "test.tsv")
    }
    return tsv_path[split]


def dataset_fn(split, shuffle_files=False, dataset=""):
    # We only have one file for each split.
    del shuffle_files

    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(get_path(DATA_DIR + dataset, split))
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    print(" >>>> about to read csv . . . ")
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # print(" >>>> after reading csv . . . ")
    # Map each tuple to a {"question": ... "answer": ...} dict.
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    # print(" >>>> after mapping . . . ")
    return ds

def load_cycic():
    for dataset in DATASETS:
        print(f" >>>> reading dataset: {dataset}")
        t5.data.set_tfds_data_dir_override(DATA_DIR + dataset)
        t5.data.TaskRegistry.add(
            f"{dataset}_task",
            # Supply a function which returns a tf.data.Dataset.
            dataset_fn=functools.partial(dataset_fn, dataset=dataset),
            splits=["train", "dev", "test"],
            # Supply a function which preprocesses text from the tf.data.Dataset.
            text_preprocessor=[dataset_preprocessor],
            # Use the same vocabulary that we used for pre-training.
            # sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
            # Lowercase targets before computing metrics.
            postprocess_fn=t5.data.postprocessors.lower_text,
            metric_fns=[t5.evaluation.metrics.accuracy],
        )
        print(f" >>>> adding one mixture per dataset: `{dataset}_mixture`")
        t5.data.MixtureRegistry.add(
            f"{dataset}_mixture", [f"{dataset}_task"], default_rate=1.0
        )

load_cycic()

TPU_JOB_NAME = "tpu_worker"
TPU = "pytorch-tpu"
TPU_TOPOLOGY = "v3-8"
GCP_PROJECT = "total-scion-310118"
TPU_ZONE = "us-central1-a"
MODEL_PARALLELISM = 8,
MODEL_DIR = "gs://unifiedqa/models/11B"

# todo: just use the Google-provided mesh_transformer script like they do in the example
# use the module_import flag to import cycic_task which can load the new task
# then use the mixture_or_task flag to activate the cycic task

# note: to make the module import work, run conda develop on the unifiedqa directory, thereby
# adding the files to the search path

def run_cycic():
    model = mtf_model.MtfModel(
            tpu_job_name=FLAGS.tpu_job_name,
            tpu=FLAGS.tpu,
            gcp_project=FLAGS.gcp_project,
            tpu_zone=FLAGS.tpu_zone,
            tpu_topology=FLAGS.tpu_topology,
            model_parallelism=FLAGS.model_parallelism,
            model_dir=FLAGS.model_dir,
            batch_size=FLAGS.batch_size,
            sequence_length={"inputs": FLAGS.input_sequence_length,
                "targets": FLAGS.target_sequence_length}
            )

