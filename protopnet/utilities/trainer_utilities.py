def init_or_update(metrics_dict, key, addition, regularization=False):
    if key not in metrics_dict or metrics_dict[key] is None:
        metrics_dict[key] = 0

    metrics_dict[key] += addition


def get_learning_rates(optimizer, model, detailed=False):
    # WARNING: this function assumes all parameters inside
    #          a complete module has the same LR

    # param_to_name will be a dict of param_id: param_name
    # e.g. {13987308184016: 'backbone.embedded_model.features.0.weight}
    param_to_name = {}
    info_lst = model.named_modules() if detailed else model.named_children()
    for name, module in info_lst:
        for param_name, param in module.named_parameters(recurse=True):
            param_full_name = f"{name}|{param_name}" if name else param_name
            param_to_name[id(param)] = param_full_name

    # if detailed, names would be backbone.fc1|weight
    # else, names would be backbone|fc1.weight
    # and they will be processed later on

    # lr dict will be a dict of layer_name: lr
    # e.g. {'add_on_layers.backbone.blabla': 0.01}
    lr_dict = {}  # module-param-lr
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if id(param) in param_to_name.keys():
                key = f"lr.{param_to_name[id(param)].split('|')[0]}"
            else:
                key = f"Unnamed Parameter {id(param)}"
            lr_dict[key] = param_group["lr"]

    return lr_dict


def predicated_extend(predicate, list1, list2):
    if predicate:
        list1.extend(list2)
    return list1
