""" fer_training_program.py """

import dill
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

class FERTrainingProgram:
    """
    Loads in data from the FER 2013 dataset and uses it to train one res net 18
    model for facial emotion recognition. The best model is saved according to
    the accuracy after each epoch.
    """
    def __init__(self):
        """
        Initializes the FERTrainingProgram class.

        Args:
            None
        """
        # Settings for model
        self.batch_size = 64
        self.epochs = 30
        self.learning_rate = 0.001
        self.num_classes = 7
        self.model_path = "fer_resnet18.pth"

        # Define transformation for images
        self.transformation = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
            transforms.Resize((224, 224)), # 224 by 224 pixels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.training_data = datasets.ImageFolder("dataset/train", transform=self.transformation)
        self.class_names = self.training_data.classes
        self.testing_data = datasets.ImageFolder("dataset/test", transform=self.transformation)

        self.training_loader = DataLoader(
            self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.testing_loader = DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.load_model(self.num_classes)

        self.train_model(self.learning_rate, self.epochs)

    def load_model(self, num_classes):
        """
        Loads model to be trained and saved with FER images

        Return: ResNet18 model
        """
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)

        return model

    def train_model(self, learning_rate, epochs):
        """
        Trains ResNet model with FER images and saves the best model according to
        the accuracy after each epoch.

        Return: None
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training Loop
        best_accuracy = 0
        total_loss = 0
        for epoch in range(epochs):
            self.model.train()
            for images, labels in self.training_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate
            self.model.eval()
            correct, total = 0, 0
            predictions, actuals = [], []
            with torch.no_grad():
                for images, labels in self.testing_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    predictions += predicted.cpu().tolist()
                    actuals += labels.cpu().tolist()

            accuracy = correct / total
            average_loss = total_loss / len(self.training_loader)
            print(f"Epoch {epoch+1}: Accuracy = {accuracy:.4f}, Average Loss = {average_loss:.4f}")

            if accuracy > best_accuracy:
                previous_best = best_accuracy
                best_accuracy = accuracy
                self.save_model()
                print(f"Best model saved, accuracy increased from {previous_best:.4f} to {best_accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(actuals, predictions, target_names=self.class_names))

    def save_model(self):
        """
        Saves the trained model and transformation to their respective files
        
        Returns: None
        """
        torch.save(self.model.state_dict(), self.model_path)
        with open("fer_transformation.pth", "wb") as f:
            dill.dump(self.transformation, f)

        print(f"FER Model weights saved to {self.model_path}.")
        print("Transformation saved to fer_transformation.pth.")

if __name__ == "__main__":
    print("Running Facial Emotion Recognition Training Program")
    FERTrainingProgram()
    print("Training Complete!")
