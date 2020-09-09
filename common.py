import os

DFL_DATA_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
DFL_CONFIG = 'resnet50_focal_%d_ap'
DFL_DEVICE = 0

if __name__ == '__main__':
    print('DATA_PATH {}'.format(DFL_DATA_PATH))
    print('CONFIG {}'.format(DFL_CONFIG))
    print('DEVICE {}'.format(DFL_DEVICE))

