from src.resnet.implement.train import train


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Only needed if you're freezing the script into an executable

    train()