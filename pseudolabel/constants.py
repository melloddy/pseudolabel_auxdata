# IMAGE MODEL HYPERPARAMS
DEFAULT_HIDDEN_SIZES = lambda: [  # noqa
    ["50"],
    ["100"],
    ["200"],
    ["400"],
    ["1000"],
    ["2000"],
    ["125", "67"],
    ["250", "125"],
    ["500", "250"],
    ["750", "375"],
    ["1000", "500"],
    ["125", "67", "32"],
    ["250", "125", "67"],
    ["500", "250", "125"],
    ["250", "250", "250"],
    ["125", "125", "125"],
    ["500", "125", "125"],
    ["250", "67", "67"],
    ["125", "34", "34"],
]
DEFAULT_EPOCHS_LR_STEPS = lambda: [(5, 3), (10, 5), (20, 10)]  # noqa
DEFAULT_DROPOUTS = lambda: [0.6, 0.4, 0.2, 0.1, 0.0]  # noqa

IMAGE_MODEL_NAME = "image_model"
IMAGE_MODEL_DATA = "image_model_data"
