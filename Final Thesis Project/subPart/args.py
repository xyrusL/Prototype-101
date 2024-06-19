import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument('--use_static_image_mode', action='store_true')
    args = parser.parse_args()

    return args