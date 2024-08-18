import os

ENV_ASSET_DIR_V2 = os.path.join(os.path.dirname(__file__), 'assets')


def full_v2_path_for(file_name):
    return os.path.join(ENV_ASSET_DIR_V2, file_name)
