const messages = document.getElementById("messages");
const q = document.getElementById("q");
const send = document.getElementById("send");
const spinner = document.getElementById("spinner");
const sendText = document.getElementById("sendText");
const topk = document.getElementById("topk");
const topkVal = document.getElementById("topkVal");
const cites = document.getElementById("cites");

function getSelectedSources() {
  const chips = [...document.querySelectorAll(".chip")];
  return chips.filter(c => c.classList.contains("active")).map(c => c.dataset.source);
}

function addMsg(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
}

function setLoading(on) {
  send.disabled = on;
  spinner.classList.toggle("hidden", !on);
  sendText.textContent = on ? "Thinking" : "Send";
}

topk.addEventListener("input", () => topkVal.textContent = topk.value);

document.getElementById("chips").addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip")) return;
  e.target.classList.toggle("active");
});

document.querySelectorAll(".example").forEach(btn => {
  btn.addEventListener("click", () => {
    q.value = btn.textContent;
    q.focus();
  });
});

async function ask() {
  const question = q.value.trim();
  if (!question) return;

  addMsg("user", question);
  q.value = "";
  setLoading(true);

  const payload = {
    question,
    sources: getSelectedSources(),
    top_k: parseInt(topk.value, 10),
  };

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.text();
      addMsg("bot", `Server error (${res.status}):\n${err}`);
      setLoading(false);
      return;
    }

    const data = await res.json();
    addMsg("bot", data.answer || "(empty)");

    if (data.citations && data.citations.length) {
      cites.innerHTML = "";
      data.citations.forEach(c => {
        const div = document.createElement("div");
        div.className = "cite";
        div.textContent = `${c.source} — ${c.pdf_file} p.${c.page}`;
        cites.appendChild(div);
      });
    } else {
      cites.textContent = "No citations returned.";
    }

  } catch (e) {
    addMsg("bot", `Network error:\n${e}`);
  } finally {
    setLoading(false);
  }
}

send.addEventListener("click", ask);
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});