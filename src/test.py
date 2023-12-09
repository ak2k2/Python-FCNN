from fcnn import predict
from helpers import (
    evaluate_predictions,
    load_network_from_file,
    read_configuration_and_data_file,
)


def test_pipeline(
    network_file_path: str,
    test_file_path: str,
    output_file_path: str,
):
    trained_network, Ni, Nh, No = load_network_from_file(network_file_path)

    _, _, n_outputs, test_data = read_configuration_and_data_file(test_file_path)

    true_labels = [row[-n_outputs:] for row in test_data]
    predictions = [predict(trained_network, row, n_outputs) for row in test_data]

    if isinstance(predictions[0], int):  # for single output classification
        predictions = [[prediction] for prediction in predictions]  # wrap in list

    class_metrics, micro, macro = evaluate_predictions(true_labels, predictions)

    with open(output_file_path, "w") as f:
        # Write the class metrics
        for metrics in class_metrics:
            # Unpack A, B, C, D, accuracy, precision, recall, F1 from metrics
            A, B, C, D, accuracy, precision, recall, F1 = metrics
            # Format each line as space-separated values
            formatted_line = f"{A} {B} {C} {D} {accuracy:.3f} {precision:.3f} {recall:.3f} {F1:.3f}\n"
            f.write(formatted_line)

        # Write the micro-averaged metrics
        micro_str = " ".join(f"{metric:.3f}" for metric in micro)
        f.write(f"{micro_str}\n")

        # Write the macro-averaged metrics
        macro_str = " ".join(f"{metric:.3f}" for metric in macro)
        f.write(f"{macro_str}\n")


if __name__ == "__main__":
    network_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/KAPOOR_TRAINED_GRADES.txt"
    test_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/GRADES_test.txt"
    output_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/grades/KAPOOR_TESTED_GRADES.txt"

    # network_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/KAPOOR_TRAINED_NNWDBC.txt"
    # test_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/WDBC_test.txt"
    # output_file_path = "/Users/armaan/Desktop/Fall-2023 Classes/Sable-Artificial-Intelligence/NN/breast_cancer/KAPOOR_TESTED_NNWDBC.txt"

    trained_network = test_pipeline(
        network_file_path=network_file_path,
        test_file_path=test_file_path,
        output_file_path=output_file_path,
    )

    print(f"Succesfully tested network and saved results to: '{output_file_path}'")
