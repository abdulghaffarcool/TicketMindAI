
const API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("ticket-form");
const ticketText = document.getElementById("ticket-text");
const charCount = document.getElementById("char-count");
const classifyBtn = document.getElementById("classify-btn");
const message = document.getElementById("message");
const resultPanel = document.getElementById("result-panel");

ticketText.addEventListener("input", () => {
    charCount.textContent = `${ticketText.value.length} / 5000`;
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = ticketText.value.trim();

    if (!text) {
        message.textContent = "Please enter a customer message.";
        return;
    }

    classifyBtn.disabled = true;
    classifyBtn.textContent = "Analyzing...";
    message.textContent = "";
    resultPanel.hidden = true;

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        const result = await response.json();

        if (!response.ok) {
            const detail = typeof result.detail === "string"
                ? result.detail
                : "Please check your input and try again.";

            throw new Error(detail);
        }

        document.getElementById("intent").textContent = result.intent;

        const confidence = Number(result.confidence);
        document.getElementById("confidence").textContent =
            `${(confidence * 100).toFixed(2)}%`;

        document.getElementById("confidence-bar").style.width =
            `${Math.max(0, Math.min(100, confidence * 100))}%`;

        const statusElement = document.getElementById("review-status");
        const statusLabels = {
            auto_accepted: "Auto accepted",
            needs_review: "Needs review",
            manual_classification: "Manual classification"
        };

        statusElement.textContent =
            statusLabels[result.review_status] ?? result.review_status;

        statusElement.className = "status";

        if (result.review_status === "needs_review") {
            statusElement.classList.add("review");
        } else if (result.review_status === "manual_classification") {
            statusElement.classList.add("manual");
        }

        const notes = {
            auto_accepted: "The prediction meets the configured confidence threshold.",
            needs_review: "A human should review this prediction before relying on it.",
            manual_classification: "Manual classification is recommended."
        };

        document.getElementById("result-note").textContent =
            notes[result.review_status] ??
            "Review the prediction before relying on it.";

        resultPanel.hidden = false;
    } catch (error) {
        message.textContent =
            `Could not classify the ticket: ${error.message}. Make sure the API is running.`;
    } finally {
        classifyBtn.disabled = false;
        classifyBtn.textContent = "Classify Ticket →";
    }
});