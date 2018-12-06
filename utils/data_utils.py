import errno
import json
import os
import _pickle as pickle
import tensorflow as tf
from keras import backend, Sequential

def export(model: Sequential, history=None, tokenizer=None, output_dir=None, **kwargs):
    """Exports the model, training history and relevant files to an output directory

    :param model: keras model to be exported
    :param history: training history (optional)
    :param tokenizer: tokenizer used to convert texts to sequences (optional)
    :param output_dir: path to the desired output directory (defaults to /output/{i})
    :param kwargs: training parameters that you want to remember the values of (exported to json (key - value pairs))
    """
    # Set the learning phase to Test = 0 since model is already trained
    backend.set_learning_phase(0)

    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(project_root, 'output')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        i = 0
        while os.path.exists(os.path.join(output_dir, str(i))):
            i = i + 1

        output_dir = os.path.join(output_dir, str(i))
        os.makedirs(output_dir)

    model.save(f'{output_dir}/model.h5')

    with backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            f'{output_dir}/protobuf/',
            inputs={'sequence': model.input},
            outputs={'sentiments': model.output}
        )
        
    if history:
        with open(f'{output_dir}/history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

    if tokenizer:
        with open(f'{output_dir}/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

    if kwargs:
        with open(f'{output_dir}/kwargs.json', 'w') as f:
            json.dump(kwargs, f)
