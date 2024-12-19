import subprocess
import sys
import os
import webbrowser

# Function to install missing packages
def install_missing_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
required_packages = ['pandas', 'numpy', 'json', 'torch', 'scikit-learn']
install_missing_packages(required_packages)

# Your original code starts here
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
num_students = 150
num_assignments = 40
assignment_types = ['quiz', 'test', 'homework', 'project', 'participation']
assignment_weights = {
    'test': 0.3,
    'participation': 0.15,
    'quiz': 0.15,
    'project': 0.25,
    'homework': 0.15
}

min_current = 0
max_current = 40
max_input_len = 40
max_output_len = 40

# Generate Synthetic Grades
assignments = []
for i in range(1, num_assignments+1):
    assignment = {
        'name': f'Assignment {i}',
        'type': np.random.choice(assignment_types),
        'challenge_rating': round(np.random.uniform(0.5, 1.5), 2)
    }
    assignments.append(assignment)

data = []
students_dict = {}
for student_id in range(1, num_students+1):
    skill_level = np.random.uniform(0.5, 1.5)
    student_assignments = []
    for assignment in assignments:
        base_grade = np.random.normal(85, 10)
        adjusted_grade = base_grade * skill_level / assignment['challenge_rating']
        final_grade = max(0, min(adjusted_grade, 100))
        student_assignments.append({
            'assignment_name': assignment['name'],
            'assignment_type': assignment['type'],
            'grade': round(final_grade)
        })
        data.append({
            'student_id': student_id,
            'assignment_name': assignment['name'],
            'assignment_type': assignment['type'],
            'grade': round(final_grade)
        })
    student_assignments = sorted(student_assignments, key=lambda x: x['assignment_name'])
    students_dict[student_id] = student_assignments

df = pd.DataFrame(data)
df.to_csv('generated_grades.csv', index=False)

# Compute weighted actual final grade for all 40 assignments
students_data = []
for student_id, assignments_list in students_dict.items():
    weighted_sum = 0
    total_weight = 0
    for a in assignments_list:
        w = assignment_weights[a['assignment_type']]
        weighted_sum += a['grade'] * w
        total_weight += w
    final_grade_weighted = weighted_sum / total_weight

    students_data.append({
        "id": student_id,
        "name": f"Student {student_id}",
        "all_assignments": assignments_list,
        "final_grade": round(final_grade_weighted, 2)
    })

with open('../data/students.json', 'w') as f:
    json.dump(students_data, f, indent=4)

# Prepare training data for variable currentCount (0 to 40)
X_list = []
Y_list = []
for s in students_data:
    all_grades = [a['grade'] for a in s['all_assignments']]
    for currentCount in range(min_current, max_current+1):
        input_grades = all_grades[:currentCount]
        future_grades = all_grades[currentCount:]

        # Pad input to length 40
        input_padded = input_grades + [0]*(max_input_len - currentCount)
        # Pad output to length 40
        output_padded = future_grades + [0]*(max_output_len - len(future_grades))

        X_list.append(input_padded)
        Y_list.append(output_padded)

X = np.array(X_list)
Y = np.array(Y_list)

combined = np.hstack((X, Y))
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

X_scaled = combined_scaled[:, :max_input_len]
Y_scaled = combined_scaled[:, max_input_len:max_input_len+max_output_len]

with open('../data/scaler.json', 'w') as f:
    json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class VariableLengthModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(VariableLengthModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = max_input_len
output_size = max_output_len
model = VariableLengthModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 400
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

torch.save(model.state_dict(), '../data/grade_prediction_model.pth')
dummy_input = torch.randn(1, input_size)
torch.onnx.export(
    model,
    dummy_input,
    "../data/grade_prediction_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)

# Open index.html in the default web browser
index_path = os.path.abspath('../html/index.html')
if os.path.exists(index_path):
    print(f"Opening {index_path} in your default web browser...")
    browser = webbrowser.get()
    browser.open(f'file://{index_path}', new=2)
else:
    print("index.html not found. Please make sure it exists in the same directory.")
