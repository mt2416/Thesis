import subprocess
import numpy as np

# yamls = np.loadtxt("C:\\dis\\ProgressiveTransformersSLP\\Configs\\Final\\to_run.txt", dtype=str)
yamls = np.loadtxt("C:\\dis\\ProgressiveTransformersSLP\\MetaConfigs\\Final\\to_run.txt", dtype=str)

for yaml in yamls:
    print(yaml)
    # subprocess.run(['python', "C:\dis\ProgressiveTransformersSLP\__main__.py"] + ['train', yaml], text=True)
    subprocess.run(['python', "C:\dis\ProgressiveTransformersSLP\meta_learner.py"] + ['train', yaml], text=True)
    print('---')