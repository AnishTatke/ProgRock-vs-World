# Audio Genre Classification : Prog-Rock vs World

This project is part of a course(Machine Learning[CAP6610]) offered at the University of Florida. The project focuses on learning, development and analysis of various Deep Learning architectures which can be used for classification of songs into two genres.

## Description

The goal of this project is to develop machine learning models that can accurately classify audio files by differentiating between one genre and the rest. The models will be trained on a dataset of audio samples and evaluated based on their classification performance. The class labels are thus defined as
- Progressive Rock
- Non-Progressive Rock(All other genres)

Hence, our professor calls this project as "Progressive Rock vs the World"
## Features
- Data: We use a dataset of various different songs along with Progressive Rock genre. The types of songs included in the Non-Progressive Rock class label include Rock, Pop, Bollywood, Jazz and several other genres
- Preprocessing:  Features such as spectrograms, mel-frequency cepstral coefficients (MFCCs), and chroma features correctly target various differences in tone, rhythm and pitch across both the class labels.
- Model Implementation: We propose two approaches to solving this challenge.

    1. Models on Snippets

        - **Data Preparation**

            For computational efficiency, we split the data into 10-second snippets and extract features discussed above for each snippet. Thus our input tensor is of shape
            `BATCH_SIZE, 1, 160, 216`.
        
        - **CNN-FCN**
        
            This approach starts with implementing a 2D Convolutional Neural Network(CNN) model on the features extarcted from each snippets. We implement various regularization techniques like Batch-Normalization, Dropout and Early Stopping to retreive the best model performance(least overfit) which generalizes equally across both the models.

        - Classification on Songs

            Once we have our predictions over the snippets, we aim to collect these predictions for each song by using a infrastructure to aggregate or learn the predictions over each snippet of that song. 
            
            To map all snippets for each songs together in the order in which the snippets appears in the song, we frame a metadata object for each snippet which contains the song name and snippet index. This helps us collect all snippets predictions for each song in the correct order.
            
            `{song_name: str, snippet_idx: int}`

            We propose three ways to gain the classification on songs from our results over snippets
            
            - **Voting and Aggregation**: We aggregate over all snippet prediction scores for each song to find a prediction over songs. We also experiment with voting over the snippet predictions to infer a predictions over a song.

            - **RNN**: We train our input arrays of variable lengths in a simple RNN implementation.

            - **LSTM**: We use the Pytorch's LSTM layer to learn sequences over long-term.

    2. Models on Songs

        - **1D-CNN model**
        
            In this approach, we extract features from the whole song as compared to previous approach. For homogenous feature representation, we set a max length of 120,000 for each song. Thus all songs less that this length are zero-padded to 120,000. Our input tensor is of shape `BATCH_SIZE, 160, 120000`.

            We perform 1D CNN layers on the input data. We observe that the use Max-Pooling helps the model target relevant features while discarding noise while Batch Normalization helps for faster convergence and reduces overfitting.

        - **FC-FD model**

            Our final approach targets a unique way of capturing the distribution of the sequences of samples. To achieve this, we calculate the mean and covariance of our extracted features across all songs. We end up with a input tensor of shape `BATCH_SIZE, 17088`.

            We finally send this inputs through a Fully Connected Neural Network.

- Training and Evaluation: The models are trained on various Batch Size (16, 32, 64) and trained for several epoch(10, 15, 20, 50)

- Visualization: The project will include visualizations of the model performance, such as confusion matrices and learning curves.

## Installation

1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Prepare the dataset: Download the audio dataset and organize it into appropriate folders based on genre labels.

2. Preprocess the audio data: Run the preprocessing script to extract the desired features from the audio files.

3. Train the models: Run the training script to train the machine learning models on the preprocessed data.

4. Evaluate the models: Run the evaluation script to assess the performance of the trained models.

5. Visualize the results: Use the provided visualization scripts to generate visualizations of the model performance.


