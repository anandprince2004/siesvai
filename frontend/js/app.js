const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");

const COLD_START_MESSAGE_DELAY_MS = 6000;

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
}
input.addEventListener("input", autoGrow);

function addMessage(text, sender) {
  if (welcome && welcome.parentElement) {
    welcome.remove();
  }
  const row = document.createElement("div");
  row.className = "msg-row " + sender;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.id = "typing-row";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function removeTypingIndicator() {
  const row = document.getElementById("typing-row");
  if (row) row.remove();
}

function showColdStartNotice() {
  const row = document.getElementById("typing-row");
  if (!row) return;
  const bubble = row.querySelector(".bubble");
  if (bubble) {
    bubble.innerHTML =
      'Waking up the server — this can take up to a minute after ' +
      'a period of inactivity. Thanks for your patience! ' +
      '<div class="typing-dots" style="margin-top:6px;"><span></span><span></span><span></span></div>';
  }
}

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed) return;

  addMessage(trimmed, "user");
  input.value = "";
  autoGrow();
  sendBtn.disabled = true;
  addTypingIndicator();

  const coldStartTimer = setTimeout(showColdStartNotice, COLD_START_MESSAGE_DELAY_MS);

  try {
    const response = await fetch(SIESVAI_CONFIG.API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: trimmed }),
    });

    if (!response.ok) {
      throw new Error("Server responded with " + response.status);
    }

    const data = await response.json();
    clearTimeout(coldStartTimer);
    removeTypingIndicator();
    addMessage(data.answer, "bot");

  } catch (err) {
    clearTimeout(coldStartTimer);
    removeTypingIndicator();
    addMessage(
      "Sorry, I couldn't reach the server. Please check your connection and try again in a moment.",
      "bot"
    );
    console.error("SIESVAI chat error:", err);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener("click", () => sendMessage(input.value));

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(input.value);
  }
});

document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => sendMessage(chip.dataset.q));
});

input.focus();
