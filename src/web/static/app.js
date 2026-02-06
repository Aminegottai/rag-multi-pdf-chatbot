const messages = document.getElementById("messages");
const q = document.getElementById("q");
const send = document.getElementById("send");
const spinner = document.getElementById("spinner");
const sendText = document.getElementById("sendText");
const topk = document.getElementById("topk");
const topkVal = document.getElementById("topkVal");
const cites = document.getElementById("cites");

const modeChips = document.getElementById("modeChips");
const baseSettings = document.getElementById("baseSettings");
const uploadSettings = document.getElementById("uploadSettings");

const pdfFile = document.getElementById("pdfFile");
const uploadBtn = document.getElementById("uploadBtn");
const docIdSpan = document.getElementById("docId");

let mode = "base";     // "base" or "upload"
let docId = null;

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

function getSelectedSources() {
  const chips = [...document.querySelectorAll("#chips .chip")];
  return chips.filter(c => c.classList.contains("active")).map(c => c.dataset.source);
}

document.getElementById("chips").addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip")) return;
  e.target.classList.toggle("active");
});

function switchMode(newMode){
  mode = newMode;
  [...modeChips.querySelectorAll(".chip")].forEach(x => x.classList.remove("active"));
  modeChips.querySelector(`[data-mode="${newMode}"]`).classList.add("active");

  if (newMode === "base") {
    baseSettings.classList.remove("hidden");
    uploadSettings.classList.add("hidden");
  } else {
    baseSettings.classList.add("hidden");
    uploadSettings.classList.remove("hidden");
  }
}

modeChips.addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip")) return;
  switchMode(e.target.dataset.mode);
});

document.querySelectorAll(".example").forEach(btn => {
  btn.addEventListener("click", () => {
    q.value = btn.textContent;
    q.focus();
  });
});

uploadBtn.addEventListener("click", async () => {
  if (!pdfFile.files || pdfFile.files.length === 0) {
    addMsg("bot", "Please choose a PDF file first.");
    return;
  }

  const file = pdfFile.files[0];
  const fd = new FormData();
  fd.append("file", file);

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";

  try {
    const res = await fetch("/upload", { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok) {
      addMsg("bot", `Upload failed (${res.status}): ${JSON.stringify(data)}`);
      return;
    }

    docId = data.doc_id;
    docIdSpan.textContent = docId;

    addMsg("bot", `PDF uploaded and indexed (chunks=${data.chunks}). Now ask questions about THIS uploaded PDF only.`);

    // Auto switch to upload mode for clarity
    switchMode("upload");

  } catch (e) {
    addMsg("bot", `Upload error: ${e}`);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload & Index";
  }
});

async function ask() {
  const question = q.value.trim();
  if (!question) return;

  addMsg("user", question);
  q.value = "";
  setLoading(true);

  let url = "/chat";
  let payload = {
    question,
    top_k: parseInt(topk.value, 10),
    sources: getSelectedSources(),
  };

  if (mode === "upload") {
    if (!docId) {
      addMsg("bot", "Upload a PDF first (Upload & Index). Then I can answer from that PDF.");
      setLoading(false);
      return;
    }
    url = "/chat_doc";
    payload = {
      doc_id: docId,
      question,
      top_k: parseInt(topk.value, 10),
    };
  }

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      addMsg("bot", `Server error (${res.status}):\n${JSON.stringify(data)}`);
      return;
    }

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