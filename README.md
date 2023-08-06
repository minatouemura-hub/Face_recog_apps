## Facial Type Diagnosis App using VGG16 Fine-Tuning

This repository contains an application that performs facial type diagnosis for women using the fine-tuning technique with the VGG16 model. The application consists of three main scripts:

1. `save_beautiful.py`: This script collects training data by utilizing web scraping techniques. It gathers facial images and processes them, storing the processed data in the "photo" directory.

2. `recog.py`: In this script, the collected data is used to fine-tune the VGG16 model. The fine-tuned model is then trained on the processed data.

3. `app.py`: This script creates a user-friendly application using Gradio, allowing users to interact with the trained model and perform facial type diagnosis.

### Usage

To run the application, follow these steps:

1. Execute `save_beautiful.py` to collect and process the training data.

2. Run `recog.py` to fine-tune the VGG16 model using the processed data and train the model.

3. Finally, start the application using `app.py` to use the trained model for facial type diagnosis.

### Requirements

Make sure you have the following dependencies installed before running the application:

- Python (version 3.9)
- Gradio
- Keras
- Other necessary libraries (refer to the `requirements.txt` file)

### License

This project is licensed under the [MIT License](LICENSE).

### Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to create a pull request or open an issue.

### Acknowledgments

The application is built upon the VGG16 model and utilizes web scraping techniques to collect the training data. Special thanks to the creators of Gradio and the original VGG16 model for their contributions to the project.
