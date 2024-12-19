document.addEventListener("DOMContentLoaded", () => {
  const total_assignments = 40;
  const max_input_len = 40;
  let scaler = {};
  let session;
  let students = [];
  let globalCurrentCount = 30;
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
      students = await studentsResponse.json();

      const scalerResponse = await fetch("../data/scaler.json");
      scaler = await scalerResponse.json();

      session = await ort.InferenceSession.create("../data/grade_prediction_model.onnx");
      console.log("Model loaded successfully.");

      await updateDashboard(globalCurrentCount);
      setupGlobalSlider();
    } catch (error) {
      console.error("Failed to initialize dashboard:", error);
    }
  }

  async function predictForStudent(student, currentCount) {
    const allAssignments = student.all_assignments.slice(0, currentCount);
    const inputGrades = allAssignments.map((a) => a.grade);

    // Pad input to length 40
    const paddedInput = inputGrades.concat(Array(max_input_len - currentCount).fill(0));

    // Scale input using the scaler
    const scaledInput = paddedInput.map((v, i) =>
      (v - scaler.mean[i]) / scaler.scale[i]
    );

    const inputTensor = new ort.Tensor("float32", new Float32Array(scaledInput), [1, max_input_len]);

    try {
      const results = await session.run({ input: inputTensor });
      const predictionsScaled = results.output.data;

      // Rescale predictions to original grade range
      const predictedFuture = Array.from(predictionsScaled).map(
        (val, i) => val * scaler.scale[i] + scaler.mean[i]
      );

      return predictedFuture;
    } catch (error) {
      console.error(`Prediction failed for ${student.name}:`, error);
      return null;
    }
  }

  async function updateDashboard(currentCount) {
    const totalStudents = students.length;
    let totalGrade = 0;
    let passing = 0;
    let failing = 0;
    let predictedFailing = 0;
    const gradeDistribution = new Array(11).fill(0);

    const redList = document.getElementById("redStudents");
    const blueList = document.getElementById("blueStudents");
    redList.innerHTML = "";
    blueList.innerHTML = "";

    for (const student of students) {
      const predictedGrades = await predictForStudent(student, currentCount);

      if (!predictedGrades) continue;

      // Map predictions back to assignments
      const actualAssignments = student.all_assignments.slice(0, currentCount);
      const predictedAssignments = student.all_assignments.map((a, i) => {
        if (i < currentCount) return { ...a };
        return { ...a, grade: predictedGrades[i - currentCount] };
      });

      // Calculate current and predicted grades
      const currentGrade = calculateWeightedAverage(actualAssignments);
      const predictedFinalGrade = calculateWeightedAverage(predictedAssignments);

      totalGrade += currentGrade;

      if (currentGrade < 60) {
        failing++;
      } else {
        passing++;
      }

      if (predictedFinalGrade < 60) {
        if (currentGrade >= 60) {
          console.log(`${student.name} classified as "Predicted to Fail"`);
          predictedFailing++;
        }
      }

      const gradeIndex = Math.min(10, Math.floor(currentGrade / 10));
      gradeDistribution[gradeIndex]++;

      // Update risk lists
      const isDoubleFail = currentGrade >= 60 && predictedFinalGrade < 60;
      const li = document.createElement("li");
      li.textContent = `${student.name} - Current Grade: ${currentGrade.toFixed(2)}`;

      if (currentGrade < 60) {
        li.classList.add("red-student");
        redList.appendChild(li);
      } else if (predictedFinalGrade < 60) {
        li.classList.add("yellow-student");
        blueList.appendChild(li);
      }

      // Add click event listener to navigate to student list
      li.addEventListener("click", () => {
          // Store the student ID in localStorage
          localStorage.setItem("selectedStudentId", student.id);

          // Navigate to the student list page
          window.location.href = "../html/student_list.html";
      });
    }

    console.log(`Sidebar Data: Passing: ${passing}, Failing: ${failing}, Predicted to Fail: ${predictedFailing}`);

    document.getElementById("averageGrade").textContent = (totalGrade / totalStudents).toFixed(2);

    renderClassPerformanceChart({ passing, failing, predictedFailing });
    renderGradeDistributionChart(gradeDistribution);
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
            data: [performance.passing, performance.failing, performance.predictedFailing],
            backgroundColor: ["#28a745", "#dc3545", "#ecec00"],
          },
        ],
      },
    });
  }

  function renderGradeDistributionChart(gradeDistribution) {
    const ctx = document.getElementById("gradeDistributionChart").getContext("2d");
    if (gradeDistributionChart) {
      gradeDistributionChart.destroy();
    }

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
        maintainAspectRatio: false,
        responsive: true,
        plugins: {
          legend: {
            display: true,
            position: "bottom",
          },
        },
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
    });
  }

  function setupGlobalSlider() {
    const slider = document.getElementById("globalCurrentCountSlider");
    const sliderValue = document.getElementById("globalCurrentCountValue");

    slider.addEventListener("input", async () => {
      globalCurrentCount = parseInt(slider.value);
      sliderValue.textContent = globalCurrentCount.toString();
      await updateDashboard(globalCurrentCount);
    });
  }

  initializeDashboard();
});
