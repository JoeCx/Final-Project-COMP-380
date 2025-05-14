""" fer_training_program.py """

import dill
import torch
import torch.nn as nn
import torch.optim as optim
import functools
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

class PixelCorruptor:
    def __init__(self, amount):
        self.corruption_amount = amount

    def __call__(self, img_tensor):
        channels, height, width = img_tensor.shape
        number_of_pixels = height * width
        number_of_corruptions = int(self.corruption_amount * number_of_pixels)

        # pick which flat indices to corrupt
        idx = torch.randperm(number_of_pixels)[:number_of_corruptions]
        for channel in range(channels):
            flat = img_tensor[channel].view(-1)
            flat[idx] = torch.rand(number_of_corruptions)
            img_tensor[channel] = flat.view(height, width)
        return img_tensor

class PixelMasker:
    def __init__(self, amount):
        self.missing_amount = amount

    def __call__(self, img_tensor):
        channels, height, width = img_tensor.shape
        number_of_pixels = height * width
        number_of_missing = int(self.missing_amount * number_of_pixels)

        idx = torch.randperm(number_of_pixels)[:number_of_missing]
        for channel in range(channels):
            flat = img_tensor[channel].view(-1)
            flat[idx] = 0.0
            img_tensor[channel] = flat.view(height, width)
        return img_tensor

class FERTrainingProgram:
    """
    Loads in data from the FER 2013 dataset and uses it to train one res net 18
    model for facial emotion recognition. The best model is saved according to
    the accuracy after each epoch.
    """
    def __init__(self, epochs, learning_rate, corruption_amount, missing_amount, model_path, output_file):
        """
        Initializes the FERTrainingProgram class.

        Args:
            None
        """
        # Settings for model
        self.batch_size = 64
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.corruption_amount = corruption_amount
        self.missing_amount = missing_amount
        self.num_classes = 7
        self.model_path = model_path
        self.output_file = output_file

        # Define transformation for images
        self.transformation = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
            transforms.Resize((224, 224)), # 224 by 224 pixels
            transforms.ToTensor(),
            PixelCorruptor(amount=self.corruption_amount),
            PixelMasker(amount=self.missing_amount),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.training_data = datasets.ImageFolder("dataset/train", transform=self.transformation)
        self.class_names = self.training_data.classes
        self.testing_data = datasets.ImageFolder("dataset/test", transform=self.transformation)

        self.training_loader = DataLoader(
            self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.testing_loader = DataLoader(self.testing_data, batch_size=self.batch_size, num_workers=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.load_model()

        self.train_model()

    def load_model(self):
        """
        Loads model to be trained and saved with FER images

        Return: ResNet18 model
        """
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        model = model.to(self.device)

        return model

    def train_model(self):
        """
        Trains ResNet model with FER images and saves the best model according to
        the accuracy after each epoch.

        Return: None
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training Loop
        best_f1_score = 0
        with open(self.output_file, "w") as output_file:
            for epoch in range(self.epochs):
                total_loss = 0
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
                f1 = f1_score(actuals, predictions, average="macro")
                average_loss = total_loss / len(self.training_loader)
                output_file.write(f"Epoch {epoch+1}: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}, Average Loss = {average_loss:.4f}\n")

                if f1 > best_f1_score:
                    previous_best = best_f1_score
                    best_f1_score = f1
                    self.save_model()
                    output_file.write(f"Best model saved, F1 score increased from {previous_best:.4f} to {best_f1_score:.4f}\n")

            output_file.write("\nClassification Report:\n")
            output_file.write(classification_report(actuals, predictions, target_names=self.class_names))

    def save_model(self):
        """
        Saves the trained model and transformation to their respective files
        
        Returns: None
        """
        torch.save(self.model.state_dict(), self.model_path)
        with open(f"transformation_{self.model_path}", "wb") as f:
            dill.dump(self.transformation, f)

        print(f"FER Model weights saved to {self.model_path}.")
        print(f"Transformation saved to transformation_{self.model_path}.")

if __name__ == "__main__":
    print("Running Facial Emotion Recognition Training Program")
    FERTrainingProgram(
        epochs=20, learning_rate=0.00005, corruption_amount=0.1,
        missing_amount=0.1, model_path="fer_resnet18.pth", output_file="original_model.txt")
    print("Training Complete!")
