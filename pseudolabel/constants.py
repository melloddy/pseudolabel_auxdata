# IMAGE MODEL HYPERPARAMS
DEFAULT_HIDDEN_SIZES = lambda: ["50", "100", "200"]  # noqa
DEFAULT_EPOCH_LR_STEPS = lambda: [(5, 3), (10, 5)]  # noqa
DEFAULT_DROPOUTS = lambda: [0.6, 0.4]  # noqa

IMAGE_MODEL_NAME = "image_model"
IMAGE_MODEL_DATA = "image_model_data"
