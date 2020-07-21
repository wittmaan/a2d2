from model.vgg16 import VGG16


def train(debug: bool = False):
    model = VGG16()
    model.train("3dbox_weights.hdf5", debug)


if __name__ == "__main__":
    train(debug=True)
