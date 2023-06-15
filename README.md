# Project4
Project4- Jessica Tabak, Holly Martin, Caleb Thornsbury, Vanat Tham


## Caleb

My part of this project was to help us decide what models to use and then tune the parameters to create models with the best accuracy for this dataset. 


After looking at the original Random Forest Classifier that was brought up by Holly, I was tasked to try and see if we could tune the parameters to increase our accuracy. 
Doing a quick google search I was able to find that there are multiple ways to tune a model, which can help improve the accuracy. One of the simplest ways I found to tune the
parameters of a random forest classifier was by increasing the max features parameters. The one max feature parameter I decided to include in the revised version of the random 
forest classifier is called n_estimators. Now, with every change in parameters you make to a code some downsides follow, but with n_estimators the only downside was increased 
running time of the model. What this feature does is change the max number of trees created before taking the average, looking at the picture in the back you can see that by simply 
increasing the number of trees in a model that can improve the accuracy. We changed the numbers of trees to 500 which got our accuracy to go from the 71% that Holly observed to 85% 
from the improved model. 

My next part in the project was to create an auto-optimized Neural Network model, this was auto-optimized to find the best parameters that also gave the best accuracy for the model.
To do this I took what we learned in class for auto-optimization and then changed the units and activations for the layers to be more in tune with what we wanted from the model. I 
ended up deciding to have 2 hidden layers, one with units from 10 to 1000 and activation of ReLU. ReLU stands for Rectified Linear Unit(s) and its basic function is to introduce the 
property of non-linearity to a deep learning model. What that means is that any negative number put into the equations will result in 0 and every positive number gets returned as what
was entered. Looking at the fist picture in the top middle left you can see that the ReLU function is linear acting, which makes it the easiest type of function to optimize in neural 
networks. For the second hidden layer I had units from 10 to 100 with the activation of tanh. Looking into the tanh activation its short for Hyperbolic Tangent Function, the picture on 
the far right is a graph of the tang function, this function is important for Neural Networks because it helps with the backpropagation process. Backpropagation involves taking the error 
rate of a forward propagation and feeding this loss backward through the neural network layers to fine-tune the weights. This process happens in the changing of epochs during the running of  
Neural Network models to create better models vs the previous. It is also a good function to be used for multi-layer processes because it delivers better training performance. When creating 
the auto-optimization code we also coded to have the RMSE calculated. RMSE Is the Root Mean Standard Error, this calculation is important because it shows the accuracy between the trained d
ata and the test data entered into the model, a good RMSE score falls between .2 and .5.The model was also coded to run a total of 10 trials. Looking at the data from the 10 trials it can be 
seen that the best RMSE was found to be .394 which was trial 7/10. This trial also had 200 units, which is saying that the best Neural Network model for this dataset needs to include 200 units 
between the hidden layers.

The last model I worked on was the Decision Tree originally made by Holly. This time we had a model around the 69% accuracy mark and I was tasked with tuning the parameters to see if I 
could create a model with better accuracy. I again returned to google for help in this situation and found a lot of information about changing parameters for a Decision Tree Classifier. 
The first step I did was try default settings, with machine learning if you don't specify none or default settings it wont use them to help optimize. After doing this the accuracy still
only increased to around like 75%. So the next step I took was to look at the max features, like I did for the random forest classifier. This time i decided to look at the max depth of the
trees.The max depth was set to 2, which only allows for 2 branches to be made from each tree so I decided to also change that to default which is none, allowing for each tree to make as many
branches that it needed, this increased the accuracy to our final percentage of 83%. The picture in the back is the actual decision tree model that was created that has 83% accuracy, you can 
also see that the tuning to the model in the code block is all none/default, helping optimize the model.






## Jessica

To get the project going, I wanted a dataset that was relevant to me.   I wanted to take all that I've learned in the course to analyze something with significance.  I also wanted a clean data set that wasn't riddled with excessive unusable data.  From Kaggle I chose the data set on maternal risks as it fit what I was looking for, precisely.  

After downloading the data set, I cleaned it up and renamed it, then exported it to kaggle to be used as a url link in the project.  I then created a datadase using spark.sql.  I exported the database, too, to kaggle.   I ran queries to find the target parameters from which I would begin exploratory data analysis.  I partioned the data and exported as parquet.   I saved the database, the csv and the parquet files to my google drive to be used in the subsequent notebooks. 


------NEXT STEPS-----

I started the deep learning portion by creating an inital notebook for exploration.  

The patients featured in the study were from various age groups. Age designations were given based on National Institute of Health (NIH) guide (NIH, 2022). More than half of the patients were pre-menopausal adults (62.9% or 638 patients, 18-44 years old), followed by adolescents (15.2%, 13-17 years old) and menopausal adults (13.6%, 45-55 years old, [NIH, 2021]). There were a small number of post-menopausal adults (3.9%, 56-64 years old), children (3.8%, 10-12 years old) and older adults (0.5% or 5 patients, 65 and older).

Age plays an important role in pregnancy health. Women who get pregnant later in life have higher risks for complications, e.g. after age 40 there is an increased risk for preeclampsia and risk for the health of the fetus (The American College of Obstetricians and Gynecologists, 2023). Menopause transition could start around the age of 44 but it varies among individuals depending on factors such as life stles and ethnicity. During first 12 months after a woman's last period, there is still a possibility of natural pregnancy (NIH, 2021). Then after menopause when the hormonal levels are too low and ovaries do not release any eggs, women could get pregnant via artificial methods such as in vitro fertilisation (IVF).

In order to build a model, we need to ensure that there are no correlations among variables, i.e. no multicolinearity. First, we can create a pairplot using seaborn to give us an overview of variable relationships.

Next I created the "opening notebook" which gave us a starting point from which to further delve.  The pie charts were used as a quick way to view the data.  The results were then converted to a score to be used in further model deployments.  Next, using an example from google, I ran seaborn density plots from columns created in the database/datframe.  

First, looking at the kernel density plots (diagonal), there are a few variables where risk levels differed across the health condition spectrum. For diastolic blood pressure (BP), the peak density for high risk patients falls in higher range of diastolic BP (around 100) while most low/mid risk patients have lower diastolic BP (60-80). Similar pattern exists in systolic BP although the differentiation is less pronounced (two smaller peaks of high risk are at same systolic BP level with mid/low risk). Also, most patients with high risk are older in age (most around 40 years old) than patients with mid/low risk (peak density at 20 years old).

To sum up for kernel density plots, visually there are bigger differences in health conditions between high and mid/low group versus between mid and low group.

Next, the pairplot showed that there might be correlations between diastolicBP/systolicBP, age/diastolicBP, age/systolicBP while for other pairs of variables there seems to be no strong relationships. To confirm, Pearson correlation coefficient values between every two variables were calculated.

Next I made a decision tree illustrating a heatmap of Pearson correlation coefficient values checking for multicolinearity.   Since there wasn't any, all the independent variables ccould be used for data modeling. Decision tree is a type of supervised learning algorithms, suitable for both continuous and categorical output variables. Here decision tree classification is used to illustrate the impact of health conditions on risk level (categorical output).  

The next step was creating a Gini graph.  Gini refers to the Gini index or Gini coefficient, which is a statistical measure commonly used to quantify the level of inequality within a distribution of data and variables other variables. It is named after the Italian statistician Corrado Gini.

The Gini coefficient is a number between 0 and 1, where 0 represents perfect equality (i.e., everyone has an equal share) and 1 represents maximum inequality (i.e., a single individual has all the share, while others have none). It is often depicted as a Lorenz curve, which is a graphical representation of the cumulative distribution function of the ranked values.

In the context of healthcare inequality, the Gini coefficient measures the deviation from a perfectly equal distribution. It is calculated as the area between the Lorenz curve and the line of perfect equality (a straight line connecting the origin and the top-right corner of the plot) divided by the total area under the line of perfect equality.

A Gini coefficient of 0 indicates perfect equality, where everyone has the same income or wealth, while a Gini coefficient of 1 represents extreme inequality, where one individual has all the income or wealth, and others have none.

The Gini coefficient is widely used in economics, sociology, and other fields to study and compare income inequality, and various indicators. It provides a concise measure to assess the inequality within a population and can be used to track changes in inequality over time or compare inequality across different targets and variables

Healthcare: It can measure the disparity in health outcomes - as used here.

Hyperparameters (max_depth, min_samples_split) were adjusted reasonably to achieve an accuracy of over 70% for the classification. Key factors that can predict risk level are Blood sugar (BS), systolicBP, and BodyTemp.

BS was at root note. When BS > 7.95 and SystolicBP > 135.0 the risk level is most likely high (Gini impurity of 0.0 in the leaf node; all 67 training instances were sorted into high risk); otherwise when SystolicBP ≤ 135.0, the outcome is also possibly high risk (gini = 0.508).

When BS ≤ 7.95, the predicted outcome is mostly low/mid risk except when systolicBP > 132.5 (gini = 0.213, predicted outcome is high risk). When BS > 0.7055 and systolicBP ≤ 125.0, the prediction is low risk; otherwise when systolicBP > 125 the prediction is mid risk (gini = 0.0). Alternatively, when BS ≤ 7.055 and BodyTemp > 99.5, the prediction is also mid risk.

This model is most accurate for patients that are pre-menopausal adults, adolescents, or menopausal adults (when combined accounts for 91.7% of total number of patients). Since post-menopausal adults, children and older adults have a much smaller sample size in comparison, it is possible that above preditions do not apply to these age groups.


------NEXT STEPS-----
##*After completing the initial notebook for analysis, I began working on the advanced ML notebook.  First I ran a machine learning technique called cross-validation to estimate the performance of an XGBoost classifier on a dataset (X_dev, y_dev). Here's a step-by-step explanation:

Import necessary modules: It starts by importing the required modules: Pipeline from sklearn.pipeline, cross_val_score from sklearn.model_selection.

Define the pipeline: Then it sets up a Pipeline that first scales the features using a MinMaxScaler and then applies an XGBClassifier to the scaled features.

Setup Cross Validation: It initializes a stratified K-fold cross-validation object with 10 splits. Stratified K-fold CV is a type of CV where the proportion of each class is preserved in each fold to ensure that one class of data is not overrepresented especially when the target variable is imbalanced.

Cross-Validation: It uses cross_val_score to perform the stratified K-fold cross-validation on the dataset using the pipeline. This involves splitting the data into 10 parts, then running the pipeline on 9 parts and validating on the 1 left out part, repeating this process for each part. The cross_val_score returns a list of scores, one for each fold.

Print Accuracy Scores: The accuracy score for each fold and the mean accuracy across all folds are printed.

Train and Predict: The pipeline is then fitted to the entire dataset and used to generate class probabilities with the predict_proba method. The predicted class is the one with the highest probability.

Confusion Matrix: A confusion matrix is generated from the true class labels and the predicted class labels. The confusion matrix is a 2D array where the entries C[i, j] are the number of observations known to be in group i but predicted to be in group j. It's displayed as a heatmap.

The normalize='all' argument in the confusion_matrix function ensures that the values displayed in the confusion matrix are the proportions of the corresponding class occurrences rather than the raw counts, giving an understanding of the classifier performance in terms of proportions or percentages.

This code block provides a robust way to estimate the performance of a model with the benefit of preprocessing and model in a pipeline and performing cross-validation to mitigate overfitting.

##*After that, I executed a Python function that defines a configurable neural network model using Keras. Here's a breakdown of what it does:

The function neural_network takes three parameters:

input_dim is the dimensionality of the input data (i.e., the number of features your data has).
output_dim is the dimensionality of the output data (i.e., the number of classes in your classification problem).
layers is a list of tuples. Each tuple contains two elements: the number of neurons (also called units) for the layer, and the dropout fraction for the layer.
model = Sequential() initializes a new Sequential model. A Sequential model is a linear stack of layers in which you can just add layers on top of each other.

The function then iterates over the layers list. For each layer, it does the following:

It unpacks the tuple into the number of neurons and the dropout fraction (neurons, dropout_frac = layer).
If it's the first layer (i == 0), it adds a Dense layer with the specified number of neurons, the 'relu' activation function, and specifies that the input dimension is input_dim. This is needed for the first layer so that Keras knows the input shape of the data.
If it's not the first layer, it adds a Dense layer with the specified number of neurons and the 'relu' activation function.
After each Dense layer, it adds a Dropout layer with the specified dropout fraction. Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to 0 during training.
After all the hidden layers have been added, it adds a final Dense layer with a number of units equal to output_dim and the 'softmax' activation function. This is the output layer of the model. In a classification problem, output_dim is often equal to the number of classes, and the 'softmax' activation function is used to output a probability distribution over the classes.

Finally, the function compiles the model with the categorical crossentropy loss function (which is suitable for multi-class classification problems), the accuracy metric, and the Adam optimizer. It then returns the compiled model.

This function provides a flexible way to create a neural network with an arbitrary number of layers, each with its own number of neurons and dropout rate. This allows you to experiment with different architectures more easily.



##**Then I used TensorFlow's Keras API to create a custom callback class called PlotLosses. This was used to visualize the training and validation losses after each epoch during the training of a neural network model.

Here's a breakdown of what it does:

class PlotLosses(tf.keras.callbacks.Callback): defines a new class PlotLosses that inherits from the tf.keras.callbacks.Callback class, meaning it can be used as a callback during model training in Keras.

def on_train_begin(self, logs={}): is a method that is called at the beginning of training. It initializes several instance variables to store the epoch number (self.i), the epoch numbers (self.x), the training losses (self.losses), and the validation losses (self.val_losses). It also sets up a new figure for plotting (self.fig) and a list to store the logs (self.logs).

def on_epoch_end(self, epoch, logs={}): is a method that is called at the end of each epoch. It updates the instance variables with information from the completed epoch (the epoch number and the training and validation losses). It then clears the output of the Jupyter cell (clear_output(wait=True)) and plots the training and validation losses so far. The wait=True argument means that it will wait to clear the output until new output is available to replace it. It then displays the plot using plt.show().

plot_losses = PlotLosses() creates an instance of the PlotLosses class. You can then pass plot_losses as a callback when you call the fit method to train your model.

In summary, this code allows you to visualize the progress of your model's training process by plotting the training and validation losses after each epoch. This is useful for monitoring the model's learning process and diagnosing issues like overfitting (if the validation loss starts to increase while the training loss continues to decrease).

 ##** Then, I performed a stratified k-fold cross-validation using a neural network model to train and evaluate a multiclass classification model. Here are the specific steps:

The necessary libraries and functions are imported. These are required for k-fold cross-validation, metrics calculation (accuracy, confusion matrix), the XGBoost algorithm, stratified k-fold cross-validation, data scaling (MinMaxScaler), and time handling.

An instance of StratifiedKFold is created with n_splits=10, meaning the data will be divided into 10 parts, maintaining the same class proportion in each split as the complete dataset.

The number of folds or splits to be generated from the given data is calculated with skf.get_n_splits(X_dev, y_dev).

Inside the loop for train_index, val_index in skf.split(X_dev, y_dev.astype("category")):, stratified k-fold cross-validation is performed. For each iteration, the function generates a new set of indices for the training and validation splits.

The targets y_dev are transformed into one-hot encoded format using tf.keras.utils.to_categorical(y_dev).

For each iteration, the code scales the training and validation data using MinMaxScaler(), which scales and translates each feature individually so that it lies between a given range on the training set (usually 0 to 1).

A neural network model is created with the function neural_network(), using the number of columns of X_train as the input dimension and the number of columns of y_dev_ohe as the output dimension.

The model is then trained using the scaled training data, with the scaled validation data used for validation in each epoch. The training is set to run for 150 epochs, with a batch size of 64. A custom Keras callback plot_losses is used, which is expected to provide live plotting of losses during the model training.

The model's predictions for the validation set are obtained by model.predict(X_val_scaled).

Then, it calculates the accuracy of the model's predictions using accuracy_score() function and prints it out. The accuracy is calculated by comparing the model's predicted class labels with the true class labels. Here, np.argmax(y_val,axis=1) and np.argmax(y_val_hat,axis=1) are used to convert the one-hot encoded labels back into class labels.

A confusion matrix is plotted to visualize the performance of the classifier. This matrix indicates the number of correct and incorrect predictions made by the classifier, broken down by each class. This is done using the sns.heatmap() function from the seaborn library which creates a heatmap representation of data.

Finally, a delay of 3 seconds is introduced before the next iteration using time.sleep(3) to probably let the user view the plotted confusion matrix before it gets replaced.

Overall, this script allows one to use stratified k-fold cross-validation to train and evaluate a neural network model, providing a live plot of losses during training and visualizing the model's performance using a confusion matrix after each fold.

##* - Next I used the Synthetic Minority Over-sampling Technique (SMOTE) from the imbalanced-learn library to balance an imbalanced dataset.

Imbalanced datasets are a common problem in machine learning, especially in classification tasks, where the classes are not equally represented. For example, in a binary classification problem, you might have 100 instances of class A and only 10 instances of class B. This imbalance can cause a model to predict poorly, especially for the minority class.

The SMOTE technique works by generating new instances of the minority class. These new instances are not just copies of existing instances, but are created by interpolating between existing instances.

The specific steps in the code are:

The distribution of the classes in the original dataset is printed (Counter(y)).

A SMOTE object is created with a specified random state for reproducibility (SMOTE(random_state=42)).

The SMOTE object is used to fit the original feature and target data (X and y), and create a resampled feature and target dataset (X_res and y_res). The resampled data should have balanced classes.

The distribution of the classes in the resampled dataset is printed (Counter(y_res)).

This is typically done before training a machine learning model, to ensure that the model is trained on a balanced dataset and performs well across all classes.

##* After that, I used Hyperparameter tuning to further execute the ML model.  SMOTE is the process of selecting the best combination of hyperparameters for a machine learning model. The goal is to find the hyperparameter values that optimize the model's performance on a given metric, such as accuracy or F1 score.

In the provided code, the GridSearchCV class from scikit-learn is used for hyperparameter tuning. It performs an exhaustive search over a specified parameter grid and evaluates the model's performance using cross-validation.

After fitting the GridSearchCV model on the training data, the best hyperparameters are obtained using the best_params_ attribute. The code snippet below demonstrates how to access the best hyperparameters:

The output will display the best parameter values found during the grid search.

The importance of hyperparameter tuning is to find the optimal configuration of hyperparameters that can lead to better model performance. By tuning hyperparameters, you can improve the model's ability to generalize and make accurate predictions on unseen data.

Understanding the specific meaning and impact of each hyperparameter requires knowledge about the algorithm being used. In the case of the RandomForestClassifier, some of the commonly tuned hyperparameters are:

n_estimators: The number of decision trees in the random forest. Increasing the number of trees can improve performance, but it also increases training time.
max_depth: The maximum depth of each decision tree. Increasing the depth can lead to overfitting, while limiting the depth can prevent overfitting but may reduce model performance.
min_samples_split: The minimum number of samples required to split an internal node. It controls the trade-off between underfitting and overfitting.
By tuning these hyperparameters, you can explore different trade-offs between model complexity and performance. The best hyperparameters found during the tuning process are expected to yield improved model performance compared to the default hyperparameter values.

##* - Then I calculated feature importance, sepaated risk level into 3 groups, created a Histogram of various risk levels, performed a ROC score (Reciver Operating Chacteristic) analysis, a Precision Recall Curve, and further analyzed the contributing features in relation to the risk level. I ran a classification report to The main goal is to accurately predict the categorical class labels of new, unseen instances, based on the patterns it has learned from the training data.

##* _ As one of my last steps I ran a Python code that uses libraries like Numpy and Matplotlib to visualize the predicted labels from a Graph Convolutional Network (GCN) model. Here's what each part does:

1. It first imports the necessary libraries: `numpy` for numerical operations and `matplotlib.pyplot` for plotting.

2. It assumes that `output` contains the predictions from the GCN model. These predictions are probably probabilities or logits for each class, outputted for each node in a graph.

3. The `argmax(dim=0)` function is used to select the class with the highest probability for each node, thus performing a multi-class classification task. The output, `predicted_labels`, is a tensor containing the class label with the highest score for each node.

4. The next block of code converts the tensor of `predicted_labels` into a 1D numpy array. If `predicted_labels` is a 0-dimensional tensor (a single value), it's converted to a numpy array with a single element. Otherwise, it's converted directly to a numpy array. This is necessary for compatibility with Matplotlib, which is used for plotting.

5. Finally, the code generates a scatter plot using Matplotlib. Each point in the plot represents a node, with the x-coordinate representing the node index and the y-coordinate representing the predicted class label for that node. The color of each point is determined by its predicted label, with the colormap 'viridis' being used.

6. After setting up the details for the plot (like figure size, labels for x and y axes, the title, and a color bar), the plot is displayed using `plt.show()`.

In summary, this code is used to visualize the classification results of a GCN model by plotting the predicted labels for each node in a scatter plot.


