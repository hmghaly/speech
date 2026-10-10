// Bootstrap errors remain readable when the dictionary cannot be fetched.
const fallback = {
  "zh-Hans": {
    "error.interfaceUnavailable": "界面文字无法加载，请重新加载页面。",
    "error.saveFailed": "保存失败，请保持页面打开并重试。",
  },
  en: {
    "error.interfaceUnavailable":
      "Interface text could not be loaded. Reload this page.",
    "error.saveFailed": "Save failed. Keep the page open and retry.",
  },
};
let messages = { "page-error": fallback, "form-error": fallback };
let templates = {};
export const locale = document.documentElement.lang === "en" ? "en" : "zh-Hans";

// The DOM ID selects the translation; a state selects changing button/status copy.
export function bind(element, state, params = {}, target = "textContent") {
  if (!element?.id) throw new Error("Translated elements require a stable ID.");
  const entry = messages[element.id];
  const translated = entry?.[locale] ?? entry?.en;
  const message =
    typeof translated === "string" ? translated : translated?.[state];
  if (typeof message !== "string")
    throw new Error("Missing translation: " + element.id + " / " + state);
  const attribute =
    target === "textContent" ? "data-i18n" : "data-i18n-" + target;
  element.setAttribute(attribute, element.id);
  element.dataset.i18nState = state;
  const text = message.replace(/\{(\w+)\}/g, (_, name) => params[name] ?? "");
  if (target === "textContent") element.textContent = text;
  else element.setAttribute(target, text);
  return element;
}

// Repeated rows expand an ID template using the stored item's stable identifier.
export function textNode(tag, state, params = {}, id, template) {
  const element = document.createElement(tag);
  element.id = id;
  if (template) messages[id] = templates[template];
  return bind(element, state, params);
}

export function content(tag, value, id) {
  const element = document.createElement(tag);
  element.id = id;
  element.dataset.content = "research";
  element.textContent = value;
  return element;
}

export function focusHeading(element) {
  element.tabIndex = -1;
  element.focus({ preventScroll: true });
}

export function showError(element, error, { focus = true } = {}) {
  const key = "error." + error.code;
  const variants = messages[element.id]?.[locale];
  bind(element, variants?.[key] ? key : "error.saveFailed", error.params);
  element.hidden = false;
  if (focus) {
    focusHeading(element);
    const bounds = element.getBoundingClientRect();
    if (bounds.top < 24 || bounds.bottom > innerHeight - 24)
      element.scrollIntoView({ block: "center", behavior: "instant" });
  }
}

export function status(element, key, params = {}) {
  bind(element, key, params);
}

export function time(seconds) {
  const value = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  return (
    String(Math.floor(value / 60)).padStart(2, "0") +
    ":" +
    String(value % 60).padStart(2, "0")
  );
}

// Loads one tiny dictionary and wires the shared shell before each page controller starts.
export async function setup(store) {
  try {
    const response = await fetch(
      new URL("../i18n/messages.json", import.meta.url),
    );
    if (!response.ok) throw new Error();
    const dictionary = await response.json();
    if (dictionary.schemaVersion !== 2 || !dictionary.messages?.["page-error"])
      throw new Error();
    messages = dictionary.messages;
    templates = dictionary.templates;
  } catch {
    throw Object.assign(new Error("interfaceUnavailable"), {
      code: "interfaceUnavailable",
    });
  }
  for (const element of document.querySelectorAll("[data-i18n]"))
    bind(element, element.dataset.i18nState || element.id, {
      year: new Date().getFullYear(),
    });
  for (const target of ["placeholder", "aria-label"]) {
    for (const element of document.querySelectorAll(
      "[data-i18n-" + target + "]",
    ))
      bind(element, element.dataset.i18nState || element.id, {}, target);
  }
  const remote = store.capabilities.persistence === "remote";
  bind(
    document.querySelector("#storage-notice"),
    remote ? "remote.notice" : "local.notice",
  );
  bind(
    document.querySelector("#storage-detail"),
    remote ? "remote.detail" : "local.detail",
  );
  await store.ready();
}
