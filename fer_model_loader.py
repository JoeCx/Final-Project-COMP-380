""" fer_model_loader.py """

from torchvision import models
import torch

class FERModelLoader:
    """
    Initializes and FER model with the designated pretrained weights from the training program.
    """
    def __init__(self, weights_file_path):
        """
        Initializes the FERModelLoader class.

        Args:
            weights_file_paths (dict): A dictionary mapping model keys to their weight file paths.
        """
        self.weights_file_path = weights_file_path
        self.num_classes = 7

        self.model = None

        # Set device to a CUDA-compatible gpu
        # Else use CPU to allow general usability and MPS if user has Apple Silicon
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu')

        self.model_initializer()

    def model_initializer(self):
        """
        Initializes ResNet18 model to a fully connected layer to output classes.
        Lastly, loads pretrained weights into the initialized model.

        Returns:
            None
        """
        # Initialize a fresh model with weights = None, so there are no weights
        self.model = models.resnet50(weights=None)

        # Initialize the model to have 7 outputs(number of emotions to be identified)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, self.num_classes)

        self.model = self.model.to(self.device)

        self.load_model_weights()

        # set model to evaluation mode
        self.model.eval()

    def load_model_weights(self):
        """
        Loads the model with the pre-trained weights.

        This method retrieves the model weights file path from the self.weights_file_path 
        and loads them into the model instance, self.model.

        Args:
            None

        Returns:
            None
        """

        # Load weights from file path into the model
        try:
            self.model.load_state_dict(
                torch.load(self.weights_file_path, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print(f"Weights File for FER Model Does Not Exist.")

    def get_model(self):
        """
        Returns:
            torch.nn.Module: The FER ResNet model.
        """
        return self.model
