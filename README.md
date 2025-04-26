😄 Smile Detection from Images
📜 Project Description
This project builds an AI-based system that can detect whether a person is smiling or not from an image.
We use a Convolutional Neural Network (CNN), trained on the CelebA dataset, to perform binary classification:

1 → Smiling

0 → Not Smiling

This project is part of the final submission for the course Image and Video (DS3273).


🏗️ Model Choice
The model used is a custom CNN architecture called SmileNet, consisting of:

3 Convolutional Layers (with ReLU activations and MaxPooling)

A Fully Connected Classifier

Final Activation: Sigmoid (for binary output)

Loss Function: Binary Cross Entropy (BCE Loss)
Optimizer: Adam Optimizer

📚 Dataset
Dataset: CelebA (https://www.kaggle.com/datasets/jessicali9530/celeba-dataset  /   https://archive.org/details/celeba)

We specifically use the 'Smiling' attribute from list_attr_celeba.txt.
project_root/
├── celebA/
│   ├── img_align_celeba/         # Folder containing all images
│   └── list_attr_celeba.txt      # Attribute file



🛠️ Installation Instructions
Clone the repository:
git clone https://github.com/22f2000427/Image-Video-Course-Assignment.git
cd Image-Video-Assignment

Install dependencies:
pip install -r requirements.txt
If requirements.txt is missing, manually install:

pip install torch torchvision pandas pillow


Prepare dataset:
Download CelebA.

Place the dataset correctly in the celebA/ folder as shown above.

🚀 How to Run the Code
1. Train the model
To train the SmileNet model:
python train.py
This will start the training process and save the model weights after completion.

2. Predict on a single image
To predict if a given image has a smiling face:
python predict.py path_to_image.jpg
Predicted: Smiling / Non Smiling


🗂️ Project Structure
├── model.py            # SmileNet CNN model
├── train.py            # Training loop
├── predict.py          # Inference (single image prediction)
├── dataset.py          # Dataset and DataLoader setup
├── config.py           # Hyperparameters (epochs, batch size, etc.)
├── interface.py        # Standardized interface file
├── README.md           # Project documentation (this file)
├── requirements.txt    # Required Python packages
└── celebA/             # Folder containing dataset (not uploaded to GitHub)



The code uses torchvision.transforms to resize images to 64x64 and convert them to tensors.





