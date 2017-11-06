MODEL_REGISTRY = {}

def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator

def get_model(model_name, model_opt):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**model_opt)
    else:
        raise ValueError("Unknown model {:s}".format(model_name))
