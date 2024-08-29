from meta_learner import MetaModel
from helpers import  load_config


model = MetaModel(load_config("C:\\dis\\ProgressiveTransformersSLP\\MetaConfigs\\Final\\MetaNet_3x128_different_width_config.yaml"))
model.load("C:\\dis\\ProgressiveTransformersSLP\\MetaModels\\Final\\MetaNet_3x128_different_width\\step_7065.pth")


print(sum(p.numel() for p in model.parameters()))