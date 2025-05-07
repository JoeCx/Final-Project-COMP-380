""" fer_missing_testing.py """

from fer_training_program import FERTrainingProgram

if __name__ == "__main__":
    print("Running Facial Emotion Recognition Training Program with Various Amounts of Missing Data")

    missing_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for missing in missing_list:
        print(f"Training with missing data: {missing}")
        model_path = f"fer_{missing}_missing.pth"
        output_file = f"{missing}_missing_output.txt"
        FERTrainingProgram(
            epochs=15, learning_rate=0.01,
            corruption_amount=0, missing_amount=missing,
            model_path=model_path, output_file=output_file)
        print(f"{missing} Missing Data - Training Complete!")

    print("Complete - Facial Emotion Recognition Training Program with Various Amounts of Missing Data")