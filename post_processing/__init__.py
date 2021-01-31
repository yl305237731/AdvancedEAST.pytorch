from .east_representer import EastRepresenter


def get_post_processing(config):
    try:
        cls = eval(config['type'])(**config['args'])
        return cls
    except:
        return None