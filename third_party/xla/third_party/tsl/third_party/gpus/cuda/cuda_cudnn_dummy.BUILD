licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_ops_infer",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_cnn_infer",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_ops_train",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_cnn_train",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_adv_infer",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_adv_train",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cudnn",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
