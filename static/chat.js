/* Osprey AI chat: a question/answer thread with the ship loading animation,
   Stop, and clickable [n] citations. Used by the chat page and by the side
   chat on each article page.

   createOspreyChat({
     thread, empty, form, input, send,   // elements
     answerUrl,                           // GET endpoint returning {html} | {cancelled} | {error}
     statusUrl, cancelUrl, frames, logo,
     extraParams: () => ({...}),          // added to each answer request
     citationTarget: (wrap, n) => el,     // element that citation [n] should reveal
     scroller,                            // element that scrolls the thread (default: the window)
     hint,                                // optional element for "getting ready" status
   })
*/
function createOspreyChat(opts) {
  const FRAME_MS = 140;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const { thread, empty, form, input, send } = opts;
  const extraParams = opts.extraParams || (() => ({}));
  (opts.frames || []).forEach(src => { new Image().src = src; });   // preload the animation

  let current = null;   // the question being answered: {controller, rid}

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  function scrollDown() {
    if (opts.scroller) opts.scroller.scrollTo({ top: opts.scroller.scrollHeight, behavior: "smooth" });
    else window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }
  function setRunning(on) {
    send.classList.toggle("running", on);
    send.setAttribute("aria-label", on ? "Stop" : "Send");
  }
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  function addUserMessage(text) {
    const msg = el("div", "msg msg-user");
    msg.append(el("div", "bubble", text));
    thread.append(msg);
  }
  function addAssistantMessage() {
    const msg = el("div", "msg msg-ai");
    const avatar = el("span", "brand-badge avatar");
    const img = el("img"); img.src = opts.logo; img.alt = "";
    avatar.append(img);
    const body = el("div", "msg-body");
    msg.append(avatar, body);
    thread.append(msg);
    return body;
  }

  // Turn [n] in the answer into buttons that reveal and highlight what n refers to.
  function linkCitations(wrap) {
    const answer = wrap.querySelector(".answer-text");
    if (!answer || !opts.citationTarget) return;
    const parts = answer.textContent.split(/\[(\d+)\]/);
    answer.replaceChildren();
    parts.forEach((part, i) => {
      if (i % 2 === 0) { answer.append(document.createTextNode(part)); return; }
      const target = opts.citationTarget(wrap, part);
      if (!target) { answer.append(document.createTextNode("[" + part + "]")); return; }
      const cite = el("button", "cite", part);
      cite.type = "button";
      cite.title = "Show " + part;
      cite.addEventListener("click", () => {
        const details = target.closest("details");
        if (details) details.open = true;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.remove("flash"); void target.offsetWidth; target.classList.add("flash");
      });
      answer.append(cite);
    });
  }

  function showOutcome(body, question, heading, text) {
    body.replaceChildren();
    const box = el("div", "msg-status");
    box.append(el("strong", null, heading));
    if (text) box.append(el("span", "muted", " " + text));
    const retry = el("button", "retry-button", "Ask again");
    retry.type = "button";
    retry.addEventListener("click", () => ask(question));
    box.append(retry);
    body.append(box);
  }

  async function ask(question) {
    if (current) return;
    if (empty) empty.hidden = true;
    addUserMessage(question);
    const body = addAssistantMessage();

    const thinking = el("div", "thinking");
    const frames = opts.frames || [];
    let ship = null;
    if (frames.length) { ship = el("img", "loading-ship"); ship.src = frames[0]; ship.alt = ""; thinking.append(ship); }
    const label = el("p", "thinking-title", "Reading and writing the answer…");
    const secs = el("p", "thinking-detail", "0 s");
    thinking.append(label, secs);
    body.append(thinking);
    scrollDown();

    const controller = new AbortController(), signal = controller.signal;
    const rid = crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2);
    current = { controller, rid };
    setRunning(true);

    const t0 = Date.now(), timers = [];
    timers.push(setInterval(() => { secs.textContent = Math.floor((Date.now() - t0) / 1000) + " s · usually 15–30 s"; }, 1000));
    if (ship && !reduceMotion) {
      let frame = 0;
      timers.push(setInterval(() => { frame = (frame + 1) % frames.length; ship.src = frames[frame]; }, FRAME_MS));
    }

    try {
      for (;;) {                                  // wait for the search engine if the app just started
        const s = await (await fetch(opts.statusUrl, { signal })).json();
        if (s.ready) break;
        label.textContent = "Getting ready (" + s.status + ")…";
        await sleep(2000);
        if (signal.aborted) throw new DOMException("Aborted", "AbortError");
      }
      label.textContent = opts.workingLabel || "Reading and writing the answer…";
      const params = new URLSearchParams({ q: question, rid, ...extraParams() });
      const data = await (await fetch(opts.answerUrl + "?" + params, { signal })).json();
      if (data.html) {
        const wrap = el("div", "ai-content");
        wrap.innerHTML = data.html;               // server-rendered, auto-escaped template
        linkCitations(wrap);
        body.replaceChildren(wrap);
      } else if (data.cancelled) {
        showOutcome(body, question, "Stopped.", "");
      } else {
        showOutcome(body, question, "Something went wrong.", data.error || "No answer was returned.");
      }
    } catch (err) {
      if (err.name === "AbortError") showOutcome(body, question, "Stopped.", "");
      else showOutcome(body, question, "Something went wrong.", String(err));
    } finally {
      timers.forEach(clearInterval);
      current = null;
      setRunning(false);
      scrollDown();
    }
  }

  function stop() {
    if (!current) return;
    current.controller.abort();
    navigator.sendBeacon(opts.cancelUrl, new URLSearchParams({ rid: current.rid }));
  }
  function clear() {
    stop();
    thread.querySelectorAll(".msg").forEach(m => m.remove());
    if (empty) empty.hidden = false;
    input.focus();
  }

  form.addEventListener("submit", e => {
    e.preventDefault();
    if (current) { stop(); return; }
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    autosize();
    ask(q);
  });
  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  input.addEventListener("input", autosize);
  (empty ? empty.querySelectorAll(".suggestion") : []).forEach(b => b.addEventListener("click", () => ask(b.textContent.trim())));
  // Leaving the page stops the question on the server too.
  window.addEventListener("pagehide", stop);

  // Show while the search engine is still loading.
  if (opts.hint) {
    (async function watchEngine() {
      try {
        const s = await (await fetch(opts.statusUrl)).json();
        opts.hint.textContent = s.ready ? "" : "Getting ready: " + s.status;
        if (!s.ready) setTimeout(watchEngine, 3000);
      } catch (_) { setTimeout(watchEngine, 5000); }
    })();
  }

  return { ask, stop, clear };
}
