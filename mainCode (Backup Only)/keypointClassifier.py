import numpy as np
import tensorflow as tf

# Define the KeyPointClassifier class
class KeyPointClassifier(object):
    # Initialize the class
    def __init__(self, model_path='subPart/recognitionModel.tflite', num_threads=1):
        # Load the TensorFlow Lite subPart
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        # Allocate tensors for the subPart
        self.interpreter.allocate_tensors()

        # Get the details of the input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # Define the call method for the class
    def __call__(self, landmark_list):
        # Get the index of the input tensor
        input_details_tensor_index = self.input_details[0]['index']

        # Set the tensor for the interpreter
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))

        # Invoke the interpreter
        self.interpreter.invoke()

        # Get the index of the output tensor
        output_details_tensor_index = self.output_details[0]['index']

        # Get the result from the output tensor
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Get the index of the maximum value in the result
        result_index = np.argmax(np.squeeze(result))

        # Return the result index
        return result_index