import errno
import os

from keras import backend
from keras.models import load_model
from tensorflow.python.saved_model import builder, tag_constants
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def


def export_h5_to_pb(model_path=None, export_path=None):
    """Converts a keras h5 model to a protobuf model

        # Arguments
            model_path: path to the directory containing the model
                (if not specified, the sub-folder last added to the output folder is selected)
            export_path: path to the desired export directory
                (if not specified, a sub-folder is created in the model_path directory)
    """

    # Set the learning phase to Test = 0 since model is already trained
    backend.set_learning_phase(0)

    if model_path is None:
        if not os.path.exists('../output'):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'output')

        i = 0
        while os.path.exists(f'../output/{str(i)}'):
            i = i + 1

        model_path = f'../output/{str(i-1)}'

    if export_path is None:
        export_path = f'{model_path}/protobuf'

    # Load the Keras model
    model = load_model(model_path + '/model.h5')

    saved_model_builder = builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'sequences': model.input},
                                      outputs={'sentiment': model.output})

    with backend.get_session() as sess:
        saved_model_builder.add_meta_graph_and_variables(sess=sess,
                                                         tags=[tag_constants.SERVING],
                                                         signature_def_map={'predict': signature})

    saved_model_builder.save()
