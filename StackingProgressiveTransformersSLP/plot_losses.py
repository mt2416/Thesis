import matplotlib.pyplot as plt

# Initialize arrays for storing losses
train_losses = []
val_losses = []
pcks_train = []
pcks_val = []
steps = []

# Open and read the file

with open("C:/dis/ProgressiveTransformersSLP/MetaModels/MetaNet_1x1/validations.txt", 'r') as file:
    for line in file:
        # Split the line by spaces
        parts = line.split()
        
        # Extract train and validation losses
        steps_index = parts.index('Steps') + 1
        train_loss_index = parts.index('train_loss') + 1
        val_loss_index = parts.index('validation_loss') + 1
        pck_train_index = parts.index('pck') + 1
        pck_val_index = parts.index('pck_val') + 1
        
        step = parts[steps_index]
        train_loss = float(parts[train_loss_index])
        val_loss = float(parts[val_loss_index])
        pck_train = float(parts[pck_train_index])
        pck_val = float(parts[pck_val_index])
        
        steps.append(step)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pcks_train.append(pck_train)
        pcks_val.append(pck_val)


# Plotting the losses
fig, axs = plt.subplots(2, 1, figsize=(10,12))

axs[0].plot(steps, train_losses, label='Train Loss')
axs[0].plot(steps, val_losses, label='Validation Loss')

axs[1].plot(steps, pcks_train, label='PCK Train')
axs[1].plot(steps, pcks_val, label='PCK Val')

axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('pck score')
axs[1].legend()
axs[1].grid(True)

plt.show()