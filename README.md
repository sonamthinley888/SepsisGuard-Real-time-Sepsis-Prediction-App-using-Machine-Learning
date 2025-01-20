# SepsisGuard: Machine Learning-based Sepsis Prediction

SepsisGuard is a machine learning-based prediction model designed to predict the likelihood of sepsis in patients using clinical data. The project includes both a Tkinter-based graphical user interface (GUI) and a Streamlit-based web application to facilitate real-time patient data input and provide sepsis predictions.

## Features

- **Machine Learning Model**: Utilizes a variety of machine learning algorithms, including Logistic Regression, Decision Trees, Random Forest, SVM, and Gradient Boosting, to predict the likelihood of sepsis.
- **Real-time Predictions**: The app allows users to input clinical data such as plasma glucose, blood pressure, and other vital metrics to receive real-time predictions of sepsis risk.
- **Interactive Visualization**: Built using Matplotlib and Seaborn, the app showcases interactive visualizations of the data distribution, correlations, and model results.
- **GUI & Web Application**: Provides both a desktop GUI built with Tkinter and a web app built with Streamlit for easy deployment and access.

## Tech Stack

- **Python**: The primary programming language used for machine learning and data preprocessing.
- **scikit-learn**: For building and evaluating machine learning models.
- **PyTorch**: Utilized for deep learning-related tasks (if applicable).
- **Streamlit**: Used for building the interactive web application for predictions.
- **Tkinter**: GUI library for building the desktop application.
- **Matplotlib & Seaborn**: For data visualization and analysis.
- **Pandas & NumPy**: For data manipulation and preprocessing.

## Installation

To run the project on your local machine, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/sonamthinley888/SepsisGuard.git
    cd SepsisGuard
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit web app:
    ```bash
    streamlit run app.py
    ```

4. To use the Tkinter-based GUI, run the following command:
    ```bash
    python sepsis_predictor_gui.py
    ```

## Usage

- **Streamlit Web Application**: After running the `app.py` file, visit the local URL provided by Streamlit (usually `http://localhost:8501/`) to access the web interface for predicting sepsis.
  
- **Tkinter GUI**: Use the Tkinter-based desktop app to enter patient data such as plasma glucose, blood pressure, and other parameters, and click "Predict Sepsis" to see the prediction result.

## Model Evaluation

The performance of the models is evaluated using classification metrics such as accuracy, precision, recall, and F1-score. Cross-validation was employed to select the best-performing model for predicting the risk of sepsis.

## Contributing

Feel free to fork this repository, open issues, and submit pull requests if you have suggestions or improvements to make.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by healthcare projects aimed at using machine learning for improving patient outcomes.
- Thanks to the open-source community for providing the tools and libraries used in this project.

