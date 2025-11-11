/* ========== STATE ========== */
let sessions = [];
let currentSessionId = null;

let thinkingInterval = null; // animation timer
let cancelFetch = false; // user pressed stop
let activeFetchId = null; // session id of running fetch
let abortController = null; // to actually abort fetch

/* ========== DOM CACHE ========== */
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendButton");
const chatContainer = document.getElementById("chatContainer");
const chatHistoryDiv = document.getElementById("chatHistory");
const sessionTitleBar = document.getElementById("sessionTitleBar");
const sessionTitle = document.getElementById("sessionTitle");

/* ========== LOCAL-STORAGE HELPERS ========== */
function loadSessions() {
  const data = localStorage.getItem("sessions");
  sessions = data ? JSON.parse(data) : [];
}
function saveSessions() {
  localStorage.setItem("sessions", JSON.stringify(sessions));
}

/* ========== SESSION MANAGEMENT ========== */
function startNewSession() {
  const newSession = {
    id: Date.now().toString(),
    name: "New Chat",
    messages: [],
    created: new Date().toISOString(),
  };
  sessions.unshift(newSession);
  currentSessionId = newSession.id;
  saveSessions();
  updateSidebar();
  renderSession();
}

function updateSidebar() {
  chatHistoryDiv.innerHTML = "";
  if (!sessions.length) {
    const msg = document.createElement("div");
    msg.className = "chat-item";
    msg.style.color = "#888";
    msg.textContent = "No sessions yet.";
    chatHistoryDiv.appendChild(msg);
    return;
  }
  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = "chat-item" + (s.id === currentSessionId ? " active" : "");
    item.textContent = s.name;
    item.title = s.name;
    item.onclick = () => {
      currentSessionId = s.id;
      updateSidebar();
      renderSession();
    };
    chatHistoryDiv.appendChild(item);
  });
}

/* ========== RENDER CHAT ========== */
function renderSession() {
  const session = sessions.find((s) => s.id === currentSessionId);
  sessionTitleBar.style.display = session ? "" : "none";
  sessionTitle.textContent = session ? session.name : "";
  chatContainer.innerHTML = "";

  if (!session || !session.messages.length) {
    chatContainer.innerHTML = `
            <div class="welcome-message">
                <h1 class="welcome-title">Forecasting Assistant</h1>
                <p class="welcome-subtitle">
                  Your AI-powered Forecasting companion. Get accurate financial Prediction of different Products </p>
            </div>`;
    resetSendBtnIfNecessary();
    return;
  }

  session.messages.forEach((m) => {
    addMessageToDOM(m.content, m.role);
    if (
      m.role === "assistant" &&
      m.content.startsWith("🤖 Thinking") &&
      session.id === activeFetchId
    )
      startThinkingAnimation(m); // resume animation only in active session
  });
  resetSendBtnIfNecessary();
}

/* ========== ADD SINGLE MESSAGE TO DOM ========== */
function addMessageToDOM(content, type) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${type}`;

  const avatar = document.createElement("div");
  avatar.className = `message-avatar ${type}-avatar`;
  avatar.textContent = type === "user" ? "You" : "SA";

  const body = document.createElement("div");
  body.className = "message-content";
  if (type === "assistant") body.innerHTML = marked.parse(content);
  else body.textContent = content;

  wrapper.append(avatar, body);
  chatContainer.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* ========== THINKING ANIMATION ========== */
function startThinkingAnimation(msgObj) {
  stopThinkingAnimation();
  let dots = 0;
  thinkingInterval = setInterval(() => {
    if (cancelFetch) {
      stopThinkingAnimation();
      return;
    }
    if (dots > 3) dots = 0;
    msgObj.content = `🤖 Thinking${".".repeat(
      dots++
    )} <span class="loader-dot"></span>`;
    saveSessions();
    if (currentSessionId === activeFetchId) renderSession();
  }, 500);
}
function stopThinkingAnimation() {
  if (thinkingInterval) {
    clearInterval(thinkingInterval);
    thinkingInterval = null;
  }
}

/* ========== BUTTON STATES ========== */
function showStopButton() {
  sendButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M6 6h12v12H6z" stroke="currentColor" stroke-width="2"/>
      </svg>`;
  sendButton.onclick = stopFetch;
}
function showSendButton() {
  sendButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M7 11L12 6L17 11M12 18V7"
              stroke="currentColor" stroke-width="2"
              stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`;
  sendButton.onclick = sendMessage;
}
function resetSendBtnIfNecessary() {
  if (activeFetchId !== currentSessionId) showSendButton();
}

/* ========== SEND MESSAGE ========== */
function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  let session = sessions.find((s) => s.id === currentSessionId);
  if (!session) {
    startNewSession();
    session = sessions.find((s) => s.id === currentSessionId);
  }

  // push user message
  session.messages.push({ role: "user", content: text });
  if (session.messages.filter((m) => m.role === "user").length === 1)
    session.name = text.length > 40 ? text.slice(0, 40) + "…" : text;

  // thinking placeholder
  const thinkMsg = {
    role: "assistant",
    content: '🤖 Thinking <span class="loader-dot"></span>',
  };
  session.messages.push(thinkMsg);

  // prep UI
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendButton.disabled = false;
  cancelFetch = false;
  activeFetchId = session.id;
  saveSessions();
  updateSidebar();
  renderSession();
  startThinkingAnimation(thinkMsg);
  showStopButton();

  // fire off request
  fetchOllamaResponse(text, session.id, thinkMsg);
}

/* ========== STOP / CANCEL ========== */
function stopFetch() {
  cancelFetch = true;
  if (abortController) abortController.abort();
}

/* ========== FETCH OLLAMA ========== */
async function fetchOllamaResponse(userInput, sessionId, thinkingMsg) {
  try {
    abortController = new AbortController();

    const res = await fetch("http://localhost:8001/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userInput,
      }),
      signal: abortController.signal,
    });

    const data = await res.json();

    if (!res.ok) {
      updateMessage(thinkingMsg, "Error from assistant.");
      return;
    }

    // ✅ Your backend returns { response: "...", source: "llm_natural" }
    const finalText = data.response;

    updateMessage(thinkingMsg, finalText);
    saveToSession(sessionId, userInput, finalText);
  } catch (error) {
    console.log("Chat aborted or failed", error);
    updateMessage(thinkingMsg, "Assistant stopped.");
  }
}


function finalizeAssistantMsg(sessionId, thinkMsg, newContent) {
  stopThinkingAnimation();
  activeFetchId = null;
  showSendButton();
  const session = sessions.find((s) => s.id === sessionId);
  if (!session) return;
  thinkMsg.content = newContent; // overwrite placeholder
  saveSessions();
  if (sessionId === currentSessionId) renderSession();
}
function insertStoppedMsg(sessionId, thinkMsg) {
  stopThinkingAnimation();
  activeFetchId = null;
  showSendButton();
  thinkMsg.content = "⏹️ Response stopped by user.";
  saveSessions();
  if (sessionId === currentSessionId) renderSession();
}

/* ========== EXAMPLE CLICK ========== */
function sendExample(txt) {
  chatInput.value = txt;
  chatInput.dispatchEvent(new Event("input"));
  sendMessage();
}

/* ========== TEXTAREA UX ========== */
chatInput.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 200) + "px";
});
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

/* ========== INIT ========== */
loadSessions();
if (!sessions.length) startNewSession();
else {
  currentSessionId = sessions[0].id;
  updateSidebar();
  renderSession();
}
