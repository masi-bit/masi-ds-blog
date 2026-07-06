import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="My Data Science Blog", layout="wide")

# --- SIDEBAR NAVIGATION (Your "Menu") ---
st.sidebar.title("📚 My DS Library")
page = st.sidebar.radio("Choose an article:", ["Home", "How Linear Regression Works", "Why Overfitting Hurts", "🌳 Decision Trees Explained", "🧩 K-Means Clustering Demo"])

# --- ARTICLE 1: HOME ---
if page == "Home":
    st.title("👋 Welcome Masimpe Muleya's Data Science Blog!")
    st.write("I'm a student learning Data Science. This is my interactive blog where you can play with the algorithms as you read!")
    
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400", caption="Data is beautiful!")

# --- ARTICLE 2: LINEAR REGRESSION (INTERACTIVE!) ---
elif page == "How Linear Regression Works":
    st.title("📈 How Linear Regression Works")
    st.write("Linear regression finds the 'line of best fit' through your data points.")
    
    # Data Science Magic: Let the user control the data!
    col1, col2 = st.columns(2)
    with col1:
        noise_level = st.slider("Add random noise to the data:", 1, 30, 10)
        num_points = st.slider("Number of data points:", 10, 100, 50)
    
    # Generate fake data
    np.random.seed(42)
    X = np.linspace(0, 10, num_points)
    y = 2 * X + 1 + np.random.normal(0, noise_level, num_points)
    
    # Plot it using Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data points')
    # Calculate and draw the best-fit line
    coeffs = np.polyfit(X, y, 1)
    line = np.poly1d(coeffs)
    ax.plot(X, line(X), color='red', linewidth=3, label=f'Y = {coeffs[0]:.2f}X + {coeffs[1]:.2f}')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    with col2:
        st.pyplot(fig)
    
    st.success(f"✅ The model found the line: Y = {coeffs[0]:.2f} * X + {coeffs[1]:.2f}")
    st.caption("👆 Move the 'Noise' slider up and down to see how the line changes!")

# --- ARTICLE 3: OVERFITTING ---
elif page == "Why Overfitting Hurts":
    st.title("🎯 Why Overfitting Hurts")
    st.write("A model that memorizes training data fails on new data. Here is a visual demonstration:")
    
    # Generate training and test sets
    x_train = np.linspace(0, 10, 15)
    y_train = np.sin(x_train) + np.random.normal(0, 0.3, 15)
    
    x_test = np.linspace(0, 10, 100)
    y_test = np.sin(x_test) + np.random.normal(0, 0.3, 100)
    
    # Overly complex polynomial (degree 14) to show overfitting
    degree = st.slider("Pick a polynomial degree (1=simple, 14=overfit):", 1, 14, 10)
    coeffs = np.polyfit(x_train, y_train, degree)
    poly_func = np.poly1d(coeffs)
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(x_train, y_train, color='blue', label='Training Data')
    ax2.plot(x_test, poly_func(x_test), color='red', linewidth=3, label=f'Degree {degree} Model')
    ax2.plot(x_test, np.sin(x_test), color='green', linestyle='--', label='True underlying function')
    ax2.set_ylim(-2, 2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
    
    if degree >= 8:
        st.warning("⚠️ Look at that wiggly red line! It perfectly hits the blue dots but goes crazy in between. That's overfitting!")
    else:
        st.info("ℹ️ The red line follows the green dashed line nicely. This model will generalize well to new data.")
        st.pyplot(fig2)

        if degree >= 8:
            st.warning("⚠️ Look at that wiggly red line! It perfectly hits the blue dots but goes crazy in between. That's overfitting!")
        else:
            st.info("ℹ️ The red line follows the green dashed line nicely. This model will generalize well to new data.")

		# --- ARTICLE 4: DECISION TREES (INTERACTIVE!) ---
elif page == "🌳 Decision Trees Explained":
    st.title("🌳 Interactive Decision Tree (Stump)")
    st.write("A Decision Tree splits data by asking a 'Yes/No' question. Move the slider to find the best split point!")
    
    import numpy as np
    
    # Generate fake data
    np.random.seed(42)
    X = np.random.uniform(0, 10, 150)
    # True rule: if X > 5, it's Red (Class 1), else Blue (Class 0)
    true_labels = (X > 5).astype(int)
    # Add a little noise (mislabel some points to make it realistic)
    noise_idx = np.random.choice(len(X), 15, replace=False)
    true_labels[noise_idx] = 1 - true_labels[noise_idx]
    
    # User slider
    threshold = st.slider("📍 Choose the split value (X = ?)", 0.0, 10.0, 5.0, 0.1)
    
    # Predict based on the slider
    pred_labels = (X > threshold).astype(int)
    accuracy = np.mean(pred_labels == true_labels) * 100
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Color the dots based on their TRUE labels
    colors = ['blue' if lbl == 0 else 'red' for lbl in true_labels]
    ax.scatter(X, np.zeros_like(X), c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Draw the split line
    ax.axvline(x=threshold, color='black', linestyle='--', lw=2, label=f'Split at X = {threshold:.1f}')
    
    ax.set_yticks([])
    ax.set_xlabel("X Value")
    ax.set_title(f"Decision Stump: Accuracy = {accuracy:.1f}%")
    ax.legend()
    ax.set_xlim(0, 10)
    
    st.pyplot(fig)
    
    if accuracy > 85:
        st.success(f"🎉 Great split! You got {accuracy:.1f}% accuracy!")
    else:
        st.info(f"🧐 Keep sliding! Try to get above 85% accuracy. Current: {accuracy:.1f}%")
    
    st.caption("👆 Slide the bar left and right. The black line moves. Try to find the spot that separates the red and blue dots the best!")
# --- ARTICLE 5: K-MEANS CLUSTERING ---
elif page == "🧩 K-Means Clustering Demo":
    st.title("🧩 K-Means Clustering Demo")
    st.write("I will generate random groups of points and use K-Means to group them by similarity!")
    
    # Import the ML libraries
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    # User controls
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Clusters (K):", 2, 5, 3)
        n_samples = st.slider("Number of Points:", 100, 500, 200)
    
    # Generate random data and apply K-Means
    X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.8, random_state=42)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.7)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', marker='X', s=200, label='Centroids')
    ax.set_title(f"K-Means Clustering (K={n_clusters})")
    ax.legend()
    st.pyplot(fig)
    
    st.success(f"✅ K-Means successfully grouped the data into {n_clusters} distinct clusters!")
    st.caption("👆 Slide the 'K' value to see how the algorithm groups the data differently!")