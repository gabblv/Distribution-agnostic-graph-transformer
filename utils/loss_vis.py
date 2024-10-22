import matplotlib.pyplot as plt


def visualise_losses(epochs, train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_losses, "o-", color="r", label="Training loss")
    ax.plot(epochs, val_losses, "o-", color="g", label="Validation loss")

    ax.set_title("Learning curves", fontsize=16)
    ax.set_xlabel("Epochs")
    ax.tick_params(axis="x", labelsize=16)
    ax.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(loc="best", prop={"size": 14})

    # plt.show()
