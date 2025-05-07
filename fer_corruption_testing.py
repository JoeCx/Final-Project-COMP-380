""" fer_corruption_testing.py """

from fer_training_program import FERTrainingProgram

if __name__ == "__main__":
    print("Running Facial Emotion Recognition Training Program with Various Amounts of Corruption")

    corruption_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for corruption in corruption_list:
        print(f"Training with corruption: {corruption}")
        model_path = f"fer_{corruption}_corruption.pth"
        output_file = f"{corruption}_corruption_output.txt"
        FERTrainingProgram(
            epochs=15, learning_rate=0.01,
            corruption_amount=corruption, missing_amount=0,
            model_path=model_path, output_file=output_file)
        print(f"{corruption} Corruption - Training Complete!")

    print("Complete - Facial Emotion Recognition Training Program with Various Amounts of Corruption")