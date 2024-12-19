let students = [];
let session;
let scaler = {};

const assignment_weights = {
  test: 0.3,
  participation: 0.15,
  quiz: 0.15,
  project: 0.25,
  homework: 0.15
};

const total_assignments = 40;
const max_input_len = 40;
const max_output_len = 40;
let currentStudent = null;
let chart = null;
let oddsChartInstance = null;
let globalCurrentCount = 30; // default

async function loadResources() {
  try {
    const studentResponse = await fetch("../data/students.json");
    students = await studentResponse.json();

    const scalerResponse = await fetch("../data/scaler.json");
    scaler = await scalerResponse.json();

    session = await ort.InferenceSession.create("../data/grade_prediction_model.onnx");
    console.log("Model loaded successfully.");

    populateStudentList();
    setupGlobalSlider();
    await updateAllStudentsHighlight(globalCurrentCount); // initial highlight and sidebar update
  } catch (error) {
    console.error("Failed to load data or model:", error);
  }
}

function calculateWeightedAverage(assignments) {
  let weightedSum = 0;
  let totalWeight = 0;
  for (const a of assignments) {
    const weight = assignment_weights[a.assignment_type] || 0;
    weightedSum += a.grade * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

async function predictForStudent(student, currentCount) {
  const allAssignments = student.all_assignments;
  const firstPart = allAssignments.slice(0, currentCount);
  const futureCount = total_assignments - currentCount;

  const inputGrades = firstPart.map(a => a.grade);
  // Pad input to length 40
  const paddedInput = inputGrades.concat(Array(max_input_len - currentCount).fill(0));

  const scaledInput = paddedInput.map((v, i) =>
    (v - scaler.mean[i]) / scaler.scale[i]
  );

  const inputTensor = new ort.Tensor("float32", new Float32Array(scaledInput), [1, max_input_len]);
  let predictionsScaled;
  try {
    const results = await session.run({ input: inputTensor });
    predictionsScaled = results.output.data;
  } catch (error) {
    console.error(`Prediction failed for ${student.name}:`, error);
    return null;
  }

  const predictedFutureScaled = Array.from(predictionsScaled).slice(0, futureCount);
  const predictedFuture = predictedFutureScaled.map((val, i) =>
    val * scaler.scale[max_input_len+i] + scaler.mean[max_input_len+i]
  );

  const predictedAssignments = allAssignments.map((a, i) => {
    if (i < currentCount) return { ...a };
    return { ...a, grade: predictedFuture[i - currentCount] };
  });

  return predictedAssignments;
}

async function updatePredictionDisplay(student, currentCount) {
  const details = document.getElementById("details");
  const predictedAssignments = await predictForStudent(student, currentCount);
  if (!predictedAssignments) {
    details.innerHTML = "<p>Prediction failed.</p>";
    document.getElementById('gradeChart').style.display = 'none';
    document.getElementById('oddsChart').style.display = 'none';
    return;
  }

  const allAssignments = student.all_assignments;
  const currentPart = allAssignments.slice(0, currentCount);

  const currentGrade = calculateWeightedAverage(currentPart);
  const predictedFinalGrade = calculateWeightedAverage(predictedAssignments);
  const actualFinalGrade = student.final_grade;

  details.innerHTML = `
    <h3>${student.name}</h3>
    <p>Current Grade (first ${currentCount}): ${currentGrade.toFixed(2)}</p>
    <p>Predicted Final Grade: ${predictedFinalGrade.toFixed(2)}</p>
    <p>Actual Final Grade: ${actualFinalGrade.toFixed(2)}</p>
  `;

  updateChart(allAssignments, predictedAssignments, currentCount);

  const odds_of_failing = oddsOfFailing(predictedFinalGrade, currentCount);
  updateOddsChart(odds_of_failing);
}

function oddsOfFailing(predictedFinalGrade, currentCount) {
  const total_assignments = 40; // ensure this matches your actual total
  // sigmaStart: uncertainty at the beginning (0 assignments done)
  // sigmaEnd: uncertainty near the end (all assignments done)
  const sigmaStart = 20;
  const sigmaEnd = 0.1;

  // progress goes from 0 (no assignments done) to 1 (all done)
  const progress = currentCount / total_assignments;

  // Interpolate sigma between start and end
  // At currentCount=0, sigma=20 (very uncertain)
  // At currentCount=40, sigma≈0.1 (very certain)
  const sigma = sigmaStart + (sigmaEnd - sigmaStart) * progress;

  // Logistic function centered at 60
  const exponent = (predictedFinalGrade - 60) / sigma;
  const odds = 1 / (1 + Math.exp(exponent));

  return odds;
}


function updateChart(allAssignments, predictedAssignments, currentCount) {
  const ctx = document.getElementById('gradeChart').getContext('2d');
  document.getElementById('gradeChart').style.display = 'block';

  const combinedGrades = predictedAssignments.map(a => a.grade);
  const labels = allAssignments.map((_, i) => `A${i+1}`);

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Grades',
        data: combinedGrades,
        fill: false,
        tension: 0.3,
        pointRadius: 4,
        segment: {
          borderColor: ctx => {
            const index = ctx.p0DataIndex;
            return index < currentCount - 1 ? 'red' : 'blue';
          }
        }
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0,
          max: 110
        }
      }
    }
  });
}

function updateOddsChart(odds_of_failing) {
  const canvas = document.getElementById("oddsChart");
  const ctx = canvas.getContext("2d");

  // Clear the canvas before redrawing
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Destroy the existing chart instance
  if (oddsChartInstance) {
    oddsChartInstance.destroy();
  }

  const failPercent = odds_of_failing * 100;
  const passPercent = 100 - failPercent;

  // Create a new doughnut chart
  oddsChartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Fail Chance", "Pass Chance"],
      datasets: [
        {
          data: [failPercent, passPercent],
          backgroundColor: ["#dc3545", "#28a745"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false, // Prevent scaling issues
    },
  });
}

function populateStudentList() {
  const studentList = document.getElementById("studentList");
  studentList.innerHTML = "";
  students.forEach(student => {
    const li = document.createElement("li");
    li.dataset.studentId = student.id;
    studentList.appendChild(li);
  });
}

async function updateAllStudentsHighlight(currentCount) {
  const studentListItems = document.querySelectorAll('#studentList li');
  for (const li of studentListItems) {
    const studentId = parseInt(li.dataset.studentId);
    const student = students.find(s => s.id === studentId);

    const predictedAssignments = await predictForStudent(student, currentCount);

    const currentPart = student.all_assignments.slice(0, currentCount);
    const currentGrade = currentCount > 0 ? calculateWeightedAverage(currentPart) : 0;

    li.classList.remove('at-risk', 'at-risk-predicted');

    if (!predictedAssignments) {
      li.textContent = `${student.name} - Current: ${currentGrade.toFixed(2)} - Actual: ${student.final_grade.toFixed(2)}`;
      continue;
    }

    const predictedFinalGrade = calculateWeightedAverage(predictedAssignments);

    li.textContent = `${student.name} - Current: ${currentGrade.toFixed(2)} - Actual: ${student.final_grade.toFixed(2)}`;

    const actualFinalGrade = student.final_grade; // store once for convenience
    if (predictedFinalGrade < 60) {
      if (actualFinalGrade < 60 && currentGrade > 60) {
        li.classList.add('at-risk-doublefail');
      } else if (actualFinalGrade >= 60) {
        li.classList.add('at-risk-predicted');
      } else {
        li.classList.add('at-risk');
      }
    }
  }

  // If there's a selected student, update their details
  if (currentStudent) {
    updatePredictionDisplay(currentStudent, currentCount);
  }
}

function selectStudent(student) {
  currentStudent = student;
  updatePredictionDisplay(currentStudent, globalCurrentCount);
}

function setupGlobalSlider() {
  const globalSlider = document.getElementById('globalCurrentCountSlider');
  const globalSliderValue = document.getElementById('globalCurrentCountValue');

  globalSlider.value = globalCurrentCount;
  globalSliderValue.textContent = globalSlider.value;

  globalSlider.addEventListener('input', async () => {
    globalCurrentCount = parseInt(globalSlider.value);
    globalSliderValue.textContent = globalCurrentCount.toString();
    await updateAllStudentsHighlight(globalCurrentCount);
  });
}

document.addEventListener("click", function(e) {
  if (e.target && e.target.tagName === "LI" && e.target.dataset.studentId) {
    const studentId = parseInt(e.target.dataset.studentId);
    const student = students.find(s => s.id === studentId);
    selectStudent(student);
  }
});

loadResources();
