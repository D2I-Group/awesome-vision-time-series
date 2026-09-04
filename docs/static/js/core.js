/* Shared page machinery for D2I project pages.
   Page-specific table rendering lives in main.js and registers itself with
   registerRender() so the language toggle can redraw it. */
const MC = (function () {
  "use strict";

  const $  = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };

  /* Strings the tables build at runtime; static copy carries its own data-zh. */
  const ZH = Object.create(null);
  let lang = document.documentElement.getAttribute("data-lang") || "en";
  const tr = (s) => (lang === "zh" && ZH[s] ? ZH[s] : s);

  const renderers = [];
  const registerRender = (fn) => { renderers.push(fn); fn(); };
  const redraw = () => renderers.forEach((f) => f());

  function applyLang(next) {
    lang = next;
    const root = document.documentElement;
    root.setAttribute("data-lang", lang);
    root.setAttribute("lang", lang === "zh" ? "zh-CN" : "en");
    localStorage.setItem("mc-lang", lang);
    $$("[data-zh]").forEach((n) => {
      if (n.dataset.en === undefined) n.dataset.en = n.innerHTML;
      n.innerHTML = lang === "zh" ? n.dataset.zh : n.dataset.en;
    });
    redraw();
  }

  /* ---------------------------------------------------------- chrome */
  function initChrome() {
    const nav = $("#nav");

    $("#lang-toggle").addEventListener("click", () => applyLang(lang === "zh" ? "en" : "zh"));

    $("#theme-toggle").addEventListener("click", () => {
      const root = document.documentElement;
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("mc-theme", next);
    });

    const onScroll = () => nav.setAttribute("data-stuck", String(window.scrollY > 8));
    addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    const menuBtn = $("#menu-toggle");
    const setMenu = (open) => {
      nav.setAttribute("data-menu", open ? "open" : "closed");
      menuBtn.setAttribute("aria-expanded", String(open));
    };
    menuBtn.addEventListener("click", () => setMenu(nav.getAttribute("data-menu") !== "open"));
    $$(".nav-links a").forEach((a) => a.addEventListener("click", () => setMenu(false)));

    const copyBtn = $("#copy-bib");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText($("#bibtex").textContent).then(() => {
          copyBtn.textContent = tr("Copied");
          copyBtn.dataset.done = "true";
          setTimeout(() => { copyBtn.textContent = tr("Copy"); delete copyBtn.dataset.done; }, 1800);
        });
      });
    }
  }

  /* ----------------------------------------------------- table helpers */
  /* Best and second-best within a set of values; ties share a rank, the way
     the papers mark their tables. `better` is "lower" or "higher". */
  function rankMarks(values, better) {
    const finite = values.filter(Number.isFinite);
    const sorted = [...new Set(finite)].sort((a, b) => (better === "higher" ? b - a : a - b));
    return values.map((v) => (v === sorted[0] ? "best" : v === sorted[1] ? "second" : ""));
  }

  const fmt = (v, d = 3) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d));

  /* A number over a slim meter; `log` for quantities spanning decades. */
  function meter(label, frac) {
    const pct = (3 + Math.max(0, Math.min(1, frac)) * 97).toFixed(1);
    return `<span class="v">${label}</span><span class="track"><i style="width:${pct}%"></i></span>`;
  }

  function bindChoices(id, key, apply) {
    $$(`#${id} button`).forEach((b) => {
      b.addEventListener("click", () => {
        $$(`#${id} button`).forEach((x) => x.removeAttribute("aria-pressed"));
        b.setAttribute("aria-pressed", "true");
        apply(b.dataset[key]);
      });
    });
  }

  function start() {
    initChrome();
    if (lang === "zh") applyLang("zh"); else redraw();
  }

  return { $, $$, el, ZH, tr, get lang() { return lang; },
           registerRender, rankMarks, fmt, meter, bindChoices, start };
})();
