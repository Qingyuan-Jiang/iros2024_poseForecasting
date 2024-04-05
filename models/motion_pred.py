# from models.basic import Basic_MLP
from models.pathnet import Path_MLP, PathNet_v2
# from models.vit import PoseTransformer
from models.path_gru import Path_GRU
# from models.path_vit import Path_VIT
# from models.baseline_dct import Baseline_DCT

named_models = {
    # 'Basic_MLP': Basic_MLP,
    'Path_MLP': Path_MLP,
    'Path_GRU': Path_GRU,
    'PathNet_v2': PathNet_v2,
    # 'PoseTransformer': PoseTransformer,
    # 'Path_VIT': Path_VIT,
    # 'Baseline_DCT': Baseline_DCT,
}


def get_model(cfg):
    model_name = cfg['model_name']
    return named_models[model_name](cfg)
