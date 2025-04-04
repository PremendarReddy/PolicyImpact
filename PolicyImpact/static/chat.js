function sendPolicy(model) {
    let inputField = document.getElementById("policy-input");
    let chatBox = document.getElementById("chat-content");

    let userInput = inputField.value.trim();
    if (userInput === "") return;

    chatBox.innerHTML += `<div class="text-end"><b>You:</b> ${userInput}</div>`;
    inputField.value = "";

    fetch("/api/analyze_policy_chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ policy: userInput, model: model })
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<div class="text-start"><b>AI:</b> ${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}
