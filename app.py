import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="My Data Science Blog", layout="wide")

# --- SIDEBAR NAVIGATION (Your "Menu") ---
st.sidebar.title("📚 My DS Library")
page = st.sidebar.radio("Choose an article:", ["Home", "How Linear Regression Works", "Why Overfitting Hurts"])

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

# --- ARTICLE 4: DECISION TREES ---
elif page == "How Decision Trees Work":
    st.title("🌳 How Decision Trees Work")
    st.write("A Decision Tree asks a series of 'Yes/No' questions to make a prediction. Think of it like 20 Questions!")

    st.write("**How it splits data:**")
    st.write("The algorithm looks at all the features and asks: *'Which question splits my data into the purest groups?'*")

    # Simple visual using matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw a simple tree structure
    ax.text(0.5, 1.0, "❓ Is X > 5?", ha='center', va='center', bbox=dict(boxstyle="round", facecolor="lightblue"))
    ax.text(0.25, 0.6, "Yes ➡️ Class A", ha='center', va='center', bbox=dict(boxstyle="round", facecolor="lightgreen"))
    ax.text(0.75, 0.6, "No ➡️ Class B", ha='center', va='center', bbox=dict(boxstyle="round", facecolor="salmon"))

    # Hide axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.2)
    ax.axis('off')

    st.pyplot(fig)

    st.info("💡 **Key takeaway:** Decision Trees are easy to understand, but they can overfit easily if you let them grow too deep!")