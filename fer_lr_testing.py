""" fer_lr_testing.py """

from fer_training_program import FERTrainingProgram

if __name__ == "__main__":
    print("Running Facial Emotion Recognition Training Program with Various Amounts of Learning Rates")

    lr_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

    for learning_rate in lr_list:
        print(f"Training with Learning Rate: {learning_rate}")
        model_path = f"fer_{learning_rate}_lr.pth"
        output_file = f"{learning_rate}_lr_output.txt"
        FERTrainingProgram(
            epochs=15, learning_rate=learning_rate, model_path=model_path, output_file=output_file)
        print(f"{learning_rate} Learning Rate - Training Complete!")

    print("Complete - Facial Emotion Recognition Training Program with Various Amounts of Learning Rates")