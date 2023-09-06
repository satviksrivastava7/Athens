import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title='Athens.ai',
                   layout='wide', page_icon='athenslogo.png')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

title_container = st.container()
col1, col2 = st.columns([1, 20])
with title_container:
    with col1:
        st.image('athenslogo.png', width=64)
    with col2:
        st.markdown('<h1 style="padding-left: 20px; padding-bottom: 50px;">Athens.ai</h1>',
                    unsafe_allow_html=True)

def build_model(df, model_choice):
    # ... (previous code remains the same) ...
    target = df.columns[-1]

    # Check if the target variable is continuous or categorical
    if df[target].nunique() <= 10:  # Assuming 10 unique values or less is categorical
        is_classification = True
    else:
        is_classification = False

    if is_classification:
        # Classification problem
        label = df[target].unique()
        df[target] = LabelEncoder().fit_transform(df[target])

        # Drop rows with missing values
        df = df.dropna()

        # Split data into X (features) and Y (target)
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]
    else:
        # Regression problem
        # Drop rows with missing values in any column
        df = df.dropna()

        # Split data into X (features) and Y (target)
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100)

    # Model selection based on user choice
    if is_classification:
        if model_choice == 'Logistic Regression':
            model = LogisticRegression(random_state=parameter_random_state)
        elif model_choice == 'KNN':
            model = KNeighborsClassifier(n_neighbors=parameter_n_neighbors)
        else:
            model = SVC(kernel='linear', C=1.0)
    else:
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'KNN Regression':
            model = KNeighborsRegressor(n_neighbors=parameter_n_neighbors)
        else:
            model = SVR(kernel='linear', C=1.0)

    # Fit the model
    model.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    # Predictions on training set
    Y_pred_train = model.predict(X_train)
    st.markdown('**2.1. Training set**')
    if is_classification:
        st.write('Accuracy Score:')
        st.info(accuracy_score(Y_train, Y_pred_train))
    else:
        st.write('R-squared Score:')
        st.info(r2_score(Y_train, Y_pred_train))
        st.write('Mean Squared Error:')
        st.info(mean_squared_error(Y_train, Y_pred_train))

    # Predictions on test set
    Y_pred_test = model.predict(X_test)
    st.markdown('**2.2. Test set**')
    if is_classification:
        st.write('Accuracy Score:')
        st.info(accuracy_score(Y_test, Y_pred_test))
    else:
        st.write('R-squared Score:')
        st.info(r2_score(Y_test, Y_pred_test))
        st.write('Mean Squared Error:')
        st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(model)
    # Create a scatter plot separated by classifier's decision boundary (for classification models)
    if is_classification and len(df.columns) >= 3:
        X_vis = df.iloc[:, :2]  # Use only the first two columns for visualization
        clf = model
        clf.fit(X_train.iloc[:, :2], Y_train)  # Fit the model using the 2-feature dataset

        h = .02  # Step size in the mesh
        x_min, x_max = X_vis.iloc[:, 0].min() - 1, X_vis.iloc[:, 0].max() + 1
        y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.rc('axes', axisbelow=True)
        plt.grid(linestyle='--', alpha=0.6)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues, alpha=0.8)  # Use Blues colormap for the decision boundary

        scatter = plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=Y, cmap=plt.cm.Greens, edgecolor='k')

        handles, labels = scatter.legend_elements()
        legend_labels = [str(label[i]) for i in range(len(label))]  # Get label values as strings
        legend = plt.legend(handles, legend_labels, title="Class Labels")

        plt.xlabel(X_vis.columns[0])
        plt.ylabel(X_vis.columns[1])
        plt.title("Dataset Separated by Classifier's Decision Boundary")

        st.subheader("4. Results")
        st.pyplot(plt)
    elif not is_classification and len(df.columns) >= 2:
        # Regression model scatter plot
        Y_pred = model.predict(X_test)

        plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')
        plt.rc('axes', axisbelow=True)
        plt.grid(linestyle='--', alpha=0.6)
        plt.scatter(Y_test, Y_pred, c='green', marker='o', edgecolor='k')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")

        st.subheader("Results")
        st.pyplot(plt)
    else:
        st.warning("The dataset does not have enough features for plotting.")

st.info('Upload a CSV file to train a classification model and visualize results')

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    model_choice = st.sidebar.selectbox('Select the classification model', ['Logistic Regression', 'KNN', 'SVM'])
    if model_choice == 'KNN':
        parameter_n_neighbors = st.sidebar.slider('Number of neighbors (n_neighbors)', 1, 20, 5)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)


if uploaded_file is not None:
    st.subheader('1. Dataset')
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Preprocessed Dataset**')
    st.write(df)

    build_model(df, model_choice)

else:
    st.write("""
    # Athens.ai
    ### Train, Test, Predict...
    This app allows you to train different classification models (Logistic Regression, KNN, SVM) on your dataset and visualize the results.
    Try adjusting the hyperparameters and choose a model to train!
    
    The web application is under development. Many other features like training regression models, image classifications model, etc. will be added soon.
    
    """)

    st.write("""
    ## About Me
    Hello there, I am Satvik Srivastava, an aspiring mathematician and computer scientist with a passion for problem-solving and innovation. \n
    I am a junior in University of Petroleum and Energy Studies, currently pursuing my Bachelors in Computer Science and Engineering. 
    I am also specializing a major in Artificial Intelligence and Machine learning, along with a minor in IOT (Internet of Things). \n   
    I enjoy tackling complex challenges and inventing new solutions. What draws me to the fascinating world of AI and machine learning is the incredible potential to harness data for groundbreaking insights. 
    AI and ML, to me, are like the modern-day magic that can transform raw information into valuable knowledge. I'm on a quest to explore this dynamic field and contribute to its exciting future. 
    Join me on this journey as we unlock the mysteries of technology and science together!
    """)
