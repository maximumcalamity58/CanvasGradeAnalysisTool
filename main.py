import subprocess
import sys
import os
import time
import webbrowser
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

# Function to install missing packages
def install_missing_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['pandas', 'numpy', 'torch', 'scikit-learn', 'onnx']
install_missing_packages(required_packages)

# Normal imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

###############################################################################
# 1. CONFIGURATION
###############################################################################
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

num_students = 150
num_assignments = 50

# We'll create random assignment names.
# Example: "Assignment 1 (1900001)", "Assignment 2 (1900002)", etc.
assign_names = [
    f"Assignment {i} (1900{100 + i})" for i in range(1, num_assignments+1)
]

###############################################################################
# 2. GENERATE THE CSV
# Structure:
#   Row 0: Student, ID, SIS User ID, Section, Assign1, Assign2, ..., Assign50
#   Row 1: (null, null, null, null, "Manual Posting", "Manual Posting", ...)
#   Row 2: ("Points Possible", null, null, null, p1, p2, ..., p50)
#   Row 3..: each student's row
###############################################################################
header_row = ["Student","ID","SIS User ID","Section"] + assign_names

# Second row: first 4 = null, then "Manual Posting" for each assignment
second_row = ["","","",""] + ["Manual Posting"]*num_assignments

# Third row: ["Points Possible", "", "", ""] + [random points for each assignment]
third_row = ["Points Possible","","",""]
for _ in range(num_assignments):
    possible = np.random.randint(8,21)  # random integer between 8..20
    third_row.append(str(possible))

# Build student rows
student_rows = []
for s in range(1, num_students+1):
    name = f"Student{s}"
    student_id = str(900000000 + s)
    sis_user_id = str(1000000 + s)
    section = f"2024Fall_ICS603s{s}"

    row_vals = [name, student_id, sis_user_id, section]
    # For each assignment, pick a random score
    for i in range(num_assignments):
        p = float(third_row[4 + i])  # parse the "Points Possible" from row 2
        # random uniform from 0..p
        score = np.random.uniform(0, p)
        row_vals.append(f"{score:.2f}")
    student_rows.append(row_vals)

csv_path = os.path.join(data_dir, 'grades.csv')
with open(csv_path, 'w', encoding='utf-8') as f:
    # Row 0
    f.write(",".join(header_row) + "\n")
    # Row 1
    f.write(",".join(second_row) + "\n")
    # Row 2
    f.write(",".join(third_row) + "\n")
    # Rows 3.. for each student
    for row in student_rows:
        f.write(",".join(row) + "\n")

print(f"Created sample CSV at {csv_path}")

###############################################################################
# 3. PARSE THE CSV, BUILD STUDENTS.JSON
###############################################################################
df_raw = pd.read_csv(csv_path, header=None)

# row 0 -> col names
# row 1 -> null/manual posting
# row 2 -> "Points Possible"/null/null/null/...
# row 3+ -> student data

header0 = df_raw.iloc[0].values
points_possible_row = df_raw.iloc[2].values
students_raw = df_raw.iloc[3:].reset_index(drop=True)

assignment_cols = header0[4:]  # after the first 4 columns
points_possible_vals = points_possible_row[4:]

cleaned_students = []
for i in range(len(students_raw)):
    row = students_raw.iloc[i].values
    student_name = str(row[0])
    student_id = str(row[1])
    sis_id = str(row[2])
    section = str(row[3])

    raw_scores = row[4:]
    numeric_scores = []
    for j, val in enumerate(raw_scores):
        try:
            numeric_scores.append(float(val))
        except:
            numeric_scores.append(0.0)

    all_assignments = []
    sum_earned = 0.0
    sum_possible = 0.0
    for k in range(len(assignment_cols)):
        try:
            possible = float(points_possible_vals[k])
        except:
            possible = 0.0
        earned = numeric_scores[k]
        if possible>0:
            sum_earned += earned
            sum_possible += possible

        all_assignments.append({
            "assignment_name": assignment_cols[k],
            "points_earned": earned,
            "points_possible": possible
        })

    final_grade = (sum_earned / sum_possible * 100.0) if sum_possible>0 else 0.0
    cleaned_students.append({
        "id": student_id,
        "name": student_name,
        "sis_id": sis_id,
        "section": section,
        "all_assignments": all_assignments,
        "final_grade": round(final_grade,2)
    })

with open(os.path.join(data_dir, 'students.json'), 'w') as f:
    json.dump(cleaned_students, f, indent=4)
print("Wrote data/students.json")

###############################################################################
# 4. TRAIN A PARTIAL-ASSIGNMENTS MODEL
###############################################################################
max_input_len = len(assignment_cols)
max_output_len = max_input_len

X_list = []
Y_list = []
for s in cleaned_students:
    all_grades = [a["points_earned"] for a in s["all_assignments"]]
    for currentCount in range(max_input_len+1):
        inp = all_grades[:currentCount]
        fut = all_grades[currentCount:]
        inp_padded = inp + [0]*(max_input_len - currentCount)
        fut_padded = fut + [0]*(max_output_len - len(fut))
        X_list.append(inp_padded)
        Y_list.append(fut_padded)

X = np.array(X_list)
Y = np.array(Y_list)
combined = np.hstack([X,Y])
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

X_scaled = combined_scaled[:, :max_input_len]
Y_scaled = combined_scaled[:, max_input_len:]

with open(os.path.join(data_dir, 'scaler.json'), 'w') as f:
    json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel(max_input_len, max_output_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_tensor)
    loss = criterion(pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1)%10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")

# Save model + ONNX
torch.save(model.state_dict(), os.path.join(data_dir, 'grade_prediction_model.pth'))

dummy_input = torch.randn(1, max_input_len)
torch.onnx.export(
    model,
    dummy_input,
    os.path.join(data_dir, 'grade_prediction_model.onnx'),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}},
    opset_version=11
)

print("Model training + export complete.")

###############################################################################
# 5. SERVE THE DASHBOARD
###############################################################################
def start_server():
    os.chdir('.')  # serve the current folder
    handler = SimpleHTTPRequestHandler
    with TCPServer(("", 8000), handler) as httpd:
        print("Serving on port 8000...")
        httpd.serve_forever()

threading.Thread(target=start_server, daemon=True).start()
url = "http://localhost:8000/html/index.html"
print("Opening", url)
webbrowser.open(url)

# keep alive
while True:
    time.sleep(1)
