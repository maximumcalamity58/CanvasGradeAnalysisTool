document.addEventListener("DOMContentLoaded", () => {
  const total_assignments = 40;
  const max_input_len = 40;
  let scaler = {};
  let session;
  let classPerformanceChart = null;
  let gradeDistributionChart = null;

  const assignment_weights = {
    test: 0.3,
    participation: 0.15,
    quiz: 0.15,
    project: 0.25,
    homework: 0.15,
  };

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

  async function initializeDashboard() {
    try {
      const studentsResponse = await fetch("../data/students.json");
      const students = await studentsResponse.json();

      const scalerResponse = await fetch("../data/scaler.json");
      scaler = await scalerResponse.json();

      session = await ort.InferenceSession.create("../data/grade_prediction_model.onnx");
      console.log("Model loaded successfully.");

      const metrics = calculateMetrics(students, total_assignments);

      document.getElementById("averageGrade").textContent = metrics.averageGrade.toFixed(2);

      await populateRiskStudents(students, total_assignments);

      renderClassPerformanceChart(metrics.performance);
      renderGradeDistributionChart(metrics.gradeDistribution);

      setupGlobalSlider(students);
    } catch (error) {
      console.error("Failed to initialize dashboard:", error);
    }
  }

  function calculateMetrics(students, currentCount) {
    const totalStudents = students.length;
    let totalGrade = 0;
    let passing = 0;
    let failing = 0;
    let predictedFailing = 0;

    const gradeDistribution = new Array(11).fill(0);

    students.forEach((student) => {
      const assignments = student.all_assignments.slice(0, currentCount);
      const currentGrade = calculateWeightedAverage(assignments);

      totalGrade += currentGrade;

      if (currentGrade < 60) {
        failing++;
      } else {
        passing++;
      }

      if (student.predicted_final_grade < 60) {
        predictedFailing++;
      }

      const gradeIndex = Math.min(10, Math.floor(currentGrade / 10));
      gradeDistribution[gradeIndex]++;
    });

    return {
      averageGrade: totalGrade / totalStudents,
      performance: { passing, failing, predictedFailing },
      gradeDistribution,
    };
  }

  async function populateRiskStudents(students, currentCount) {
    const redList = document.getElementById("redStudents");
    const blueList = document.getElementById("blueStudents");
    redList.innerHTML = "";
    blueList.innerHTML = "";

    for (const student of students) {
      // Predict assignments and grades up to currentCount
      const predictedAssignments = await predictForStudent(student, currentCount);
      const currentGrade = calculateWeightedAverage(
        student.all_assignments.slice(0, currentCount)
      );
      const predictedFinalGrade = calculateWeightedAverage(predictedAssignments);

      const li = document.createElement("li");
      li.textContent = `${student.name} - Grade: ${currentGrade.toFixed(2)}`;
      li.style.cursor = "pointer";

      li.addEventListener("click", () => {
        alert(`Clicked on: ${student.name}`);
      });

      if (predictedFinalGrade < 60) {
        console.log(currentGrade + " " + predictedFinalGrade)
      }

      // Classify students as red or blue
      if (currentGrade < 60) {
        li.classList.add("red-student");
        redList.appendChild(li);
      } else if (predictedFinalGrade < 60) {
        li.classList.add("blue-student");
        blueList.appendChild(li);
      }
    }
  }

  async function predictForStudent(student, currentCount) {
    const allAssignments = student.all_assignments.slice(0, currentCount);
    const futureCount = total_assignments - currentCount;
    const inputGrades = allAssignments.map((a) => a.grade);

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
    const predictedFuture = predictedFutureScaled.map(
      (val, i) => val * scaler.scale[max_input_len + i] + scaler.mean[max_input_len + i]
    );

    return allAssignments.concat(predictedFuture.map((grade) => ({ grade })));
  }

  function renderClassPerformanceChart(performance) {
    const ctx = document.getElementById("classPerformanceChart").getContext("2d");
    if (classPerformanceChart) {
      classPerformanceChart.destroy();
    }

    classPerformanceChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Passing", "Failing", "Predicted to Fail"],
        datasets: [
          {
            data: [
              performance.passing,
              performance.failing,
              performance.predictedFailing,
            ],
            backgroundColor: ["#28a745", "#dc3545", "#007bff"],
          },
        ],
      },
    });
  }

  function renderGradeDistributionChart(gradeDistribution) {
    const ctx = document.getElementById("gradeDistributionChart").getContext("2d");

    // Destroy the existing chart instance if it exists
    if (gradeDistributionChart) {
      gradeDistributionChart.destroy();
    }

    // Create a new chart instance
    gradeDistributionChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: gradeDistribution.map((_, i) => `${i * 10}-${i * 10 + 9}`),
        datasets: [
          {
            label: "Number of Students",
            data: gradeDistribution,
            backgroundColor: "#007bff",
          },
        ],
      },
      options: {
        maintainAspectRatio: false, // Ensures the chart respects container size
        responsive: true,          // Makes the chart adapt to container size
        plugins: {
          legend: {
            display: true,        // Show the legend
            position: "bottom",   // Position the legend at the bottom
          },
        },
        scales: {
          y: {
            beginAtZero: true,    // Ensure the y-axis starts at zero
          },
        },
      },
    });
  }

  function setupGlobalSlider(students) {
    const slider = document.getElementById("globalCurrentCountSlider");
    const sliderValue = document.getElementById("globalCurrentCountValue");

    slider.addEventListener("input", async () => {
      const currentCount = parseInt(slider.value);
      sliderValue.textContent = currentCount.toString();

      const metrics = calculateMetrics(students, currentCount);
      renderClassPerformanceChart(metrics.performance);
      renderGradeDistributionChart(metrics.gradeDistribution);
      document.getElementById("averageGrade").textContent = metrics.averageGrade.toFixed(2);

      await populateRiskStudents(students, currentCount);
    });
  }

  initializeDashboard();
});
