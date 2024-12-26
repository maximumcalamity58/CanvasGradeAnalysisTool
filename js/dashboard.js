let students = [];
let session = null;
let scalerData = null;
let maxAssignments = 0;

let gradeDistChart = null;
let passFailChart = null;

document.addEventListener("DOMContentLoaded", init);

async function init() {
  try {
    // 1) load students + scaler + onnx
    const res = await fetch("../data/students.json");
    students = await res.json();

    const res2 = await fetch("../data/scaler.json");
    scalerData = await res2.json();

    session = await ort.InferenceSession.create("../data/grade_prediction_model.onnx");
    console.log("ONNX model loaded.");

    // figure out how many assignments from the data
    maxAssignments = students[0].all_assignments.length;

    // set up slider
    const slider = document.getElementById("assignmentsSlider");
    const sliderVal = document.getElementById("sliderValue");
    slider.max = maxAssignments.toString();
    slider.value = "10"; // or any default
    sliderVal.textContent = slider.value;

    slider.addEventListener("input", async () => {
      sliderVal.textContent = slider.value;
      await updateDashboard(parseInt(slider.value));
    });

    // initial
    await updateDashboard(parseInt(slider.value));

  } catch (err) {
    console.error("Failed to init:", err);
  }
}

async function updateDashboard(currentCount) {
  let sumCurrent = 0;
  let passCount = 0;
  let failCount = 0;
  const dist = new Array(11).fill(0);

  const riskList = document.getElementById("riskList");
  riskList.innerHTML = "";

  for (const s of students) {
    const predictedAssignments = await predictAssignments(s, currentCount);
    if (!predictedAssignments) continue;

    // "current" portion is predictedAssignments[0..currentCount)
    const currentPart = predictedAssignments.slice(0, currentCount);
    const currentGrade = computeGrade(currentPart);

    sumCurrent += currentGrade;

    const bucket = Math.min(10, Math.floor(currentGrade / 10));
    dist[bucket]++;

    if (currentGrade < 60) {
      failCount++;
      const li = document.createElement("li");
      li.textContent = `${s.name} - ${currentGrade.toFixed(2)}%`;
      riskList.appendChild(li);
    } else {
      passCount++;
    }
  }

  const avg = sumCurrent / students.length;
  document.getElementById("avgGrade").textContent = avg.toFixed(2);

  renderGradeDist(dist);
  renderPassFail(passCount, failCount);
}

function computeGrade(assignments) {
  let sumEarned = 0;
  let sumPossible = 0;
  for (const a of assignments) {
    sumEarned += a.points_earned;
    sumPossible += a.points_possible;
  }
  return sumPossible > 0 ? (sumEarned / sumPossible * 100) : 0;
}

async function predictAssignments(student, currentCount) {
  // The model outputs a full 50 predicted grades (for partial input).
  // We take the first "currentCount" from the real data, the rest from model.
  const scores = student.all_assignments.map(a => a.points_earned);
  // build input
  const inputSeq = scores.slice(0, currentCount);
  const padded = inputSeq.concat(Array(maxAssignments - currentCount).fill(0));

  // scale
  const scaled = [];
  for (let i = 0; i < padded.length; i++) {
    const mean_i = scalerData.mean[i];
    const scale_i = scalerData.scale[i];
    const val = padded[i];
    scaled.push((val - mean_i) / scale_i);
  }

  const inputTensor = new ort.Tensor("float32", Float32Array.from(scaled), [1, maxAssignments]);
  try {
    const results = await session.run({ input: inputTensor });
    const predArr = results.output.data;
    // re-scale
    const predictedAll = [];
    for (let i = 0; i < maxAssignments; i++) {
      const reVal = predArr[i] * scalerData.scale[maxAssignments + i] + scalerData.mean[maxAssignments + i];
      predictedAll.push(reVal);
    }
    // merge with real data
    const out = [];
    for (let i = 0; i < maxAssignments; i++) {
      if (i < currentCount) {
        out.push({
          ...student.all_assignments[i],
          points_earned: scores[i] // real
        });
      } else {
        out.push({
          ...student.all_assignments[i],
          points_earned: predictedAll[i]
        });
      }
    }
    return out;
  } catch (err) {
    console.error(`Prediction error for ${student.name}:`, err);
    return null;
  }
}

function renderGradeDist(dist) {
  const ctx = document.getElementById("gradeDistChart").getContext("2d");
  if (gradeDistChart) gradeDistChart.destroy();

  const labels = dist.map((_, i) => {
    if (i < 10) return `${i*10}-${i*10+9}`;
    return "100";
  });

  gradeDistChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Num Students",
          data: dist,
          backgroundColor: "#007bff"
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true }
      }
    }
  });
}

function renderPassFail(passCount, failCount) {
  const ctx = document.getElementById("passFailChart").getContext("2d");
  if (passFailChart) passFailChart.destroy();

  passFailChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Pass", "Fail"],
      datasets: [
        {
          data: [passCount, failCount],
          backgroundColor: ["#28a745", "#dc3545"]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false
    }
  });
}
