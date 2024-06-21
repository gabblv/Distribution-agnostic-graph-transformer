from nets.graph_transformer_net import GraphTransformerNet

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def gnn_model(model_name, net_params):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models[model_name](net_params)
