from SYNTHETIC.template_model import MLP, End2EndModel


# Independent & Sequential Model
def ModelXtoC(input_dim, n_attributes, expand_dim):
    return MLP(input_dim=input_dim, num_classes=n_attributes, expand_dim=expand_dim)

# Independent Model
def ModelOracleCtoY(n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part
    model = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_attributes, num_classes, expand_dim)

# Joint Model
def ModelXtoCtoY(input_dim, num_classes, n_attributes, expand_dim,
                 use_relu, use_sigmoid):
    model1 = MLP(input_dim=input_dim, num_classes=n_attributes, expand_dim=expand_dim)
    model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return End2EndModel(model1, model2, n_attributes, use_relu, use_sigmoid)