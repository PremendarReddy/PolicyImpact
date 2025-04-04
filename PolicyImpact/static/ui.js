document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("#chat-form");
    const chatBox = document.querySelector("#chat-box");
    const inputField = document.querySelector("#policy-text");
    const darkModeToggle = document.querySelector("#toggle-dark-mode");

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        const policyText = inputField.value.trim();
        if (policyText === "") return;

        appendMessage("user", policyText);
        appendMessage("bot", "Analyzing... Please wait.");

        fetch(form.action, {
            method: "POST",
            body: new FormData(form),
        })
        .then(response => response.text())
        .then(data => {
            let tempDiv = document.createElement("div");
            tempDiv.innerHTML = data;
            const aiResponse = tempDiv.querySelector("#ai-response")?.innerText || "Error fetching response.";
            const explanation = tempDiv.querySelector("#ai-explanation")?.innerText || "No explanation available.";

            removeLastMessage();
            appendMessage("bot", aiResponse);
            appendMessage("bot", `🤖 **Why?** ${explanation}`);
        })
        .catch(error => {
            console.error("Error:", error);
            removeLastMessage();
            appendMessage("bot", "⚠️ An error occurred. Please try again.");
        });

        inputField.value = "";
    });

    function appendMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        messageDiv.innerHTML = `<span>${message}</span>`;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeLastMessage() {
        const messages = document.querySelectorAll(".message.bot");
        if (messages.length > 0) {
            messages[messages.length - 1].remove();
        }
    }

    darkModeToggle.addEventListener("click", function () {
        document.body.classList.toggle("dark-mode");
    });
});
