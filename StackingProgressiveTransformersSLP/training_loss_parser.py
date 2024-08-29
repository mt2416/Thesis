import re 

def parse_training_loss(path):
    with open(path, 'r') as f:
        data = f.read()

    training_losses = re.findall(r'total training loss (\d+\.\d+)', data)
    training_losses = [float(loss) for loss in training_losses]

    return training_losses 

training_losses = parse_training_loss("C:\\dis\\ProgressiveTransformersSLP\\Models\\Final\\T2G2P_2_8(512,1024)\\train.log")

print(training_losses)