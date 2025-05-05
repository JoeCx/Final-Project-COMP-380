""" fer_epoch_testing.py """

from fer_training_program import FERTrainingProgram

print("Running Facial Emotion Recognition Training Program with Various Amounts of Epochs")

epoch_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

for epoch_amount in epoch_list:
    print(f"Training with {epoch_amount} Epochs")
    model_path = f"fer_{epoch_amount}_epoch.pth"
    output_file = f"{epoch_amount}_epoch_output.txt"
    FERTrainingProgram(
        epochs=epoch_amount, learning_rate=0.001, model_path=model_path, output_file=output_file)
    print(f"{epoch_amount} Epochs - Training Complete!")

print("Complete - Facial Emotion Recognition Training Program with Various Amounts of Epochs")