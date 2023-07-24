# deep-learning-challnge
The purpose of this analysis was to create a deep learning model using TensorFlow and Keras to predict whether an organization funded by Alphabet Soup will be successful based on various features in the dataset. The goal was to achieve a predictive accuracy higher than 75% by optimizing the model through data preprocessing, hyperparameter tuning, and adjusting the neural network's architecture.

Results
Data Preprocessing
Target Variable: The target variable for our model is "IS_SUCCESSFUL," which indicates whether the organization is successful (1) or not (0).
Feature Variables: The feature variables include "APPLICATION_TYPE," "AFFILIATION," "CLASSIFICATION," "USE_CASE," "ORGANIZATION," "STATUS," "INCOME_AMT," "SPECIAL_CONSIDERATIONS," and "ASK_AMT."
Variables to Remove: We dropped the "EIN" and "NAME" columns as they are neither targets nor features and don't contribute to the model's predictive power.
Compiling, Training, and Evaluating the Model
Neural Network Architecture: For our model, we selected three hidden layers with 100, 50, and 25 neurons, respectively. The activation function used for the hidden layers was ReLU, which helps introduce non-linearity to the model and allows it to learn complex patterns in the data. The output layer has one neuron with a sigmoid activation function for binary classification, as it yields a probability between 0 and 1.

Target Model Performance: After several rounds of training and hyperparameter tuning, we achieved a model accuracy of approximately 78%, which surpasses our target of 75%. The model is showing promising results in predicting the success of funded organizations.

Steps Taken to Increase Model Performance:

Data Preprocessing: We handled missing values, performed one-hot encoding for categorical variables, and standardized the numerical features using Scikit-learn's StandardScaler.
Hyperparameter Tuning: We experimented with different learning rates, batch sizes, and optimizers, ultimately settling on Adam with a learning rate of 0.001, which yielded the best results for our model.
Model Architecture: By increasing the number of neurons and adding hidden layers, the model's capacity to learn complex patterns improved, contributing to the overall performance.
Early Stopping: We implemented early stopping with patience to prevent overfitting and achieve better generalization on unseen data.
Summary
The deep learning model developed for Alphabet Soup demonstrates good predictive performance, achieving an accuracy of 78%. By optimizing the model through data preprocessing, hyperparameter tuning, and adjusting its architecture, we were able to meet and exceed our target performance.

Recommendation:

For further improvements, we recommend experimenting with different model architectures, such as using a more complex neural network like a Convolutional Neural Network (CNN) for image data or a Recurrent Neural Network (RNN) for sequential data. Moreover, incorporating feature engineering techniques and exploring ensemble methods could help boost the model's performance for this classification problem.

Overall, the deep learning model provides valuable insights for Alphabet Soup in identifying organizations with a higher likelihood of success, aiding in strategic funding decisions and maximizing their impact in the philanthropic domain.

*** CHATGPT supported me writing the code and ReadME. Thank you
