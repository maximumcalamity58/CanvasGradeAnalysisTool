# Project README
# Grade Prediction Dashboard 

This project demonstrates the generation of synthetic student grade data, the training of a predictive model using a simple neural network, and the visualization of both the data and model predictions on an interactive dashboard.

The primary components of the project include:  
- Synthetic data generation of student assignments and grades  
- Preprocessing and normalization of data  
- Training a predictive neural network model using PyTorch  
- Exporting the model to ONNX format  
- Running a local HTTP server to serve an interactive web dashboard that shows current student performance and predicts future grades

## Steps to Set Up and Run the Project

### Prerequisites  
- Python 3.12 installed  
- Pip package manager  
- (Optional) Visual Studio Code (VS Code) with Python plugin  
- Git (if you are cloning directly from GitHub)

## Two Ways to Run the Project

### 1. Using VS Code
1. **Download and Extract**: Download the project from GitHub and unzip it into your desired directory.  
2. **Open in VS Code**: Launch VS Code and open the project folder.  
3. **Install Python Plugin**: If not already installed, add the Python plugin to VS Code.  
4. **Trust the Folder**: If prompted, choose to trust the workspace folder.  
5. **Run the Script**: Execute tpython main` script within VS Code. This will:  
   - Automatically install all required dependencies (this may take a few minutes)  
   - Train the model on the generated synthetic data  
   - Export the trained model to ONNX format  
   - Launch a local HTTP server  
   - Open the web dashboard in your default browser

**Note:** The initial run may take 3-5 minutes as dependencies are installed and the model is trained. Afterward, the dashboard should appear, allowing you to interact with the predictions.

### 2. Using Command Prompt (cmd)
1. **Download and Extract**: Download the project from GitHub and unzip it into your desired directory.  
2. **Navigate in cmd**: Open a command prompt and ucd path/to/project/fol` to move into the project directory.  
3. **Run the Script**: Execute the commapython main`. This will run the main script, which:  
   - Installs all required dependencies  
   - Trains the model on generated data  
   - Exports the model to ONNX format  
   - Hosts a local server and opens the dashboard in your default browser

Again, allow 3-5 minutes for dependencies to install and the model to train before the dashboard appears.

## What the Project Does

1. **Generates Example Data**: It creates synthetic student performance data with a variety of assignment types (e.g., quizzes, tests, projects).  
2. **Trains a Predictive Model**: Using PyTorch, it trains a neural network to predict future student assignment grades based on current grades.  
3. **Dashboard Visualization**: With a local HTTP server, it serves a web interface that visualizes current class performance, at-risk students, and projected outcomes.  
4. **Interactive Sliders**: The dashboard includes sliders to adjust the number of current assignments considered, updating predictions on the fly.

## Additional Notes

- The ONNX model allows for model portability and integration with web technologies via ONNX Runtime.  
- The charts and visualizations leverage libraries like Chart.js.  
- All data is generated randomly, providing a sandbox for experimenting with model training and inference.  
- The provided code and instructions are meant to ensure a quick start: simply run the main script and explore the dashboard.
