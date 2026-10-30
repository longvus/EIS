"""Benchmarks, tasks, costs, episode seeds and PD1 caps of the experiments."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

N_CONFIGS = 32

LCBENCH_TASKS = (
    "APSFailure",
    "Amazon_employee_access",
    "Australian",
    "Fashion-MNIST",
    "KDDCup09_appetency",
    "MiniBooNE",
    "adult",
    "airlines",
    "albert",
    "bank-marketing",
    "blood-transfusion-service-center",
    "car",
    "christine",
    "cnae-9",
    "connect-4",
    "covertype",
    "credit-g",
    "dionis",
    "fabert",
    "helena",
    "higgs",
    "jannis",
    "jasmine",
    "jungle_chess_2pcs_raw_endgame_complete",
    "kc1",
    "kr-vs-kp",
    "mfeat-factors",
    "nomao",
    "numerai28.6",
    "phoneme",
    "segment",
    "shuttle",
    "sylvine",
    "vehicle",
    "volkert",
)

PD1_TASKS = (
    "cifar100_wide_resnet_batch_size_256",
    "cifar10_wide_resnet_batch_size_256",
    "fashion_mnist_max_pooling_cnn_relu_batch_size_256",
    "fashion_mnist_max_pooling_cnn_tanh_batch_size_256",
    "fashion_mnist_simple_cnn_relu_batch_size_256",
    "imagenet_resnet_batch_size_256",
    "imagenet_resnet_batch_size_512",
    "lm1b_transformer_batch_size_2048",
    "mnist_max_pooling_cnn_relu_batch_size_256",
    "mnist_max_pooling_cnn_tanh_batch_size_256",
    "mnist_simple_cnn_relu_batch_size_256",
    "svhn_no_extra_wide_resnet_batch_size_1024",
    "svhn_no_extra_wide_resnet_batch_size_256",
    "translate_wmt_xformer_translate_batch_size_64",
    "uniref50_transformer_batch_size_128",
)

PD1_CAPS = {
    "cifar100_wide_resnet_batch_size_256": 35,
    "cifar10_wide_resnet_batch_size_256": 52,
    "fashion_mnist_max_pooling_cnn_relu_batch_size_256": 35,
    "fashion_mnist_max_pooling_cnn_tanh_batch_size_256": 35,
    "fashion_mnist_simple_cnn_relu_batch_size_256": 43,
    "imagenet_resnet_batch_size_256": 500,
    "imagenet_resnet_batch_size_512": 250,
    "lm1b_transformer_batch_size_2048": 37,
    "mnist_max_pooling_cnn_relu_batch_size_256": 39,
    "mnist_max_pooling_cnn_tanh_batch_size_256": 39,
    "mnist_simple_cnn_relu_batch_size_256": 48,
    "svhn_no_extra_wide_resnet_batch_size_1024": 12,
    "svhn_no_extra_wide_resnet_batch_size_256": 51,
    "translate_wmt_xformer_translate_batch_size_64": 1413,
    "uniref50_transformer_batch_size_128": 21,
}

PD1_PATH = ROOT / "datasets" / "pd1" / "pd1_all_curves_15tasks.npz"

PD1_SHA256 = "f28bca495849dcd8403173fac3be1c21d3395eb496ffbbcce70da46b308950bc"
