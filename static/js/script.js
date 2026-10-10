
document.addEventListener("DOMContentLoaded", function () {
    const emailInput = document.getElementById("emailText");
    const charCount = document.getElementById("charCount");
    const form = document.getElementById("analyzerForm");
    const clearButton = document.getElementById("clearButton");
    const analyzeButton = document.getElementById("analyzeButton");
    const historyBody = document.getElementById("historyBody");
    const clearHistoryButton = document.getElementById("clearHistory");

    const HISTORY_KEY = "mailguard_scan_history";
    let history = [];

    // Safely load saved scan history from this browser.
    try {
        const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
        if (Array.isArray(saved)) {
            history = saved;
        }
    } catch (error) {
        history = [];
        console.warn("Could not read scan history:", error);
    }

    // Update the email character counter.
    function updateCharCount() {
        if (!emailInput || !charCount) return;

        charCount.textContent =
            emailInput.value.length.toLocaleString() + " / 20000";
    }

    if (emailInput) {
        emailInput.addEventListener("input", updateCharCount);
        updateCharCount();
    }

    // Sample emails for testing.
    const samples = {
        spam: "URGENT! Congratulations! You have won a $5,000 cash prize. Click here immediately to claim your reward before this offer expires!",
        ham: "Dear Students, this is a reminder that the project review will be held on Monday at 10:00 AM in the department seminar hall. Please bring your project report and presentation slides. Regards, Department Coordinator."
    };

    document.querySelectorAll("[data-sample]").forEach(function (button) {
        button.addEventListener("click", function () {
            if (!emailInput) return;

            const sampleType = button.dataset.sample;
            if (samples[sampleType]) {
                emailInput.value = samples[sampleType];
                updateCharCount();
                emailInput.focus();
            }
        });
    });

    // Clear the email input.
    if (clearButton) {
        clearButton.addEventListener("click", function () {
            if (emailInput) {
                emailInput.value = "";
                updateCharCount();
                emailInput.focus();
            }
        });
    }

    // Avoid accidental double submission.
    if (form && analyzeButton) {
        form.addEventListener("submit", function () {
            if (!emailInput || !emailInput.value.trim()) return;

            analyzeButton.disabled = true;
            analyzeButton.textContent = "Analyzing...";

            // The browser will submit the form to Flask.
            // Flask renders the actual model prediction.
        });
    }

    // Save only a short preview and the model's actual rendered result.
    // Full email messages are not stored in browser history.
    function saveRenderedResult() {
        const resultState = document.querySelector(
            ".result-state.spam-state, .result-state.ham-state"
        );

        if (!resultState || !emailInput) return;

        const predictionHeading = resultState.querySelector("h2");
        const confidenceElement = document.querySelector(".confidence-row strong");

        if (!predictionHeading || !confidenceElement) return;

        const prediction = predictionHeading.textContent.trim();
        const confidenceText = confidenceElement.textContent.trim();
        const confidence = Number.parseFloat(confidenceText.replace("%", ""));

        if (!["SPAM", "HAM"].includes(prediction) || !Number.isFinite(confidence)) {
            return;
        }

        const email = emailInput.value.trim();
        if (!email) return;

        // Prevent the same page result from being recorded repeatedly.
        const signature = prediction + "|" + confidenceText + "|" + email;
        try {
            if (sessionStorage.getItem("mailguard_last_signature") === signature) {
                return;
            }
            sessionStorage.setItem("mailguard_last_signature", signature);
        } catch (error) {
            console.warn("Session storage is unavailable:", error);
        }

        history.unshift({
            preview: email.replace(/\s+/g, " ").slice(0, 100),
            prediction: prediction,
            confidence: confidence,
            time: new Date().toLocaleString()
        });

        history = history.slice(0, 50);
        saveHistory();
    }

    function saveHistory() {
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        } catch (error) {
            console.warn("Could not save scan history:", error);
        }

        renderHistory();
        updateStatistics();
    }

    // Build table cells with textContent to prevent HTML injection.
    function renderHistory() {
        if (!historyBody) return;

        historyBody.replaceChildren();

        if (history.length === 0) {
            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 4;
            cell.className = "history-empty";
            cell.textContent =
                "No scans yet. Analyze an email to start your history.";

            row.appendChild(cell);
            historyBody.appendChild(row);
            return;
        }

        history.forEach(function (item) {
            const row = document.createElement("tr");
            const previewCell = document.createElement("td");
            const resultCell = document.createElement("td");
            const confidenceCell = document.createElement("td");
            const timeCell = document.createElement("td");
            const badge = document.createElement("span");

            previewCell.textContent = item.preview;
            previewCell.title = item.preview;

            badge.className =
                "history-result " +
                (item.prediction === "SPAM" ? "spam" : "ham");
            badge.textContent = item.prediction;
            resultCell.appendChild(badge);

            confidenceCell.textContent =
                Number(item.confidence).toFixed(2) + "%";
            timeCell.textContent = item.time;

            row.append(
                previewCell,
                resultCell,
                confidenceCell,
                timeCell
            );

            historyBody.appendChild(row);
        });
    }

    function updateStatistics() {
        const totalElement = document.getElementById("totalScans");
        const spamElement = document.getElementById("spamCount");
        const hamElement = document.getElementById("hamCount");

        const spamTotal = history.filter(
            item => item.prediction === "SPAM"
        ).length;

        const hamTotal = history.filter(
            item => item.prediction === "HAM"
        ).length;

        if (totalElement) totalElement.textContent = history.length;
        if (spamElement) spamElement.textContent = spamTotal;
        if (hamElement) hamElement.textContent = hamTotal;
    }

    // Clear saved browser history.
    if (clearHistoryButton) {
        clearHistoryButton.addEventListener("click", function () {
            if (!window.confirm("Clear all saved scan history?")) return;

            history = [];

            try {
                localStorage.removeItem(HISTORY_KEY);
                sessionStorage.removeItem("mailguard_last_signature");
            } catch (error) {
                console.warn("Could not clear browser storage:", error);
            }

            renderHistory();
            updateStatistics();
        });
    }

    // Model metrics from the current evaluation report.
    const chartCanvas = document.getElementById("modelChart");

    if (chartCanvas && typeof Chart !== "undefined") {
        new Chart(chartCanvas, {
            type: "bar",
            data: {
                labels: [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ],
                datasets: [
                    {
                        label: "Naive Bayes",
                        data: [97.9965, 98.8462, 92.7798, 95.7169],
                        backgroundColor: "#9aa9d6",
                        borderRadius: 5
                    },
                    {
                        label: "Logistic Regression",
                        data: [98.3449, 99.6154, 93.5018, 96.4618],
                        backgroundColor: "#7e91ed",
                        borderRadius: 5
                    },
                    {
                        label: "SVM",
                        data: [99.4774, 100, 97.8339, 98.9051],
                        backgroundColor: "#4568e8",
                        borderRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: "index"
                },
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            usePointStyle: true,
                            pointStyle: "rectRounded",
                            padding: 18,
                            font: {
                                size: 10
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return context.dataset.label + ": " +
                                    context.parsed.y.toFixed(2) + "%";
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            callback: value => value + "%"
                        },
                        grid: {
                            color: "#edf0f7"
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    } else if (chartCanvas) {
        console.warn("Chart.js did not load. Check your internet connection.");
    }

    // Restore existing history, then add this page's rendered result if present.
    renderHistory();
    updateStatistics();
    saveRenderedResult();

    // Update active sidebar link while navigating sections.
    document.querySelectorAll(".nav-link").forEach(function (link) {
        link.addEventListener("click", function () {
            document.querySelectorAll(".nav-link").forEach(function (item) {
                item.classList.remove("active");
            });

            link.classList.add("active");
        });
    });
});