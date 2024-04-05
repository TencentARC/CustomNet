from ldm.util import create_carvekit_interface, load_and_preprocess

def load_preprocess_model():
    carvekit = create_carvekit_interface()
    return carvekit

def preprocess_image(models, input_im):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array.
    '''
    input_im = load_and_preprocess(models, input_im)
    return input_im