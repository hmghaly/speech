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
const languagePreference = "speech.interface.language";

// Prefer an explicit language choice while keeping Chinese as the starting language.
function interfaceLocale() {
  const requested = new URLSearchParams(location.search).get("lang");
  if (["en", "zh-Hans"].includes(requested)) return requested;
  try {
    const saved = localStorage.getItem(languagePreference);
    if (["en", "zh-Hans"].includes(saved)) return saved;
  } catch {
    // A URL choice still works when browser preferences cannot be stored.
  }
  return document.documentElement.lang === "en" ? "en" : "zh-Hans";
}

export const locale = interfaceLocale();
document.documentElement.lang = locale;

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

// Reuse the desktop links in a modal sidebar with native keyboard focus handling.
function setupNavigation() {
  const toggle = document.querySelector("#menu-toggle");
  const menu = document.querySelector("#mobile-menu");
  const navigation = document.querySelector(".header-actions > .nav");
  const header = navigation.parentElement;
  toggle.addEventListener("click", () => {
    menu.querySelector(".mobile-menu-links").append(navigation);
    menu.showModal();
    toggle.setAttribute("aria-expanded", "true");
    document.documentElement.classList.add("menu-open");
  });
  menu.addEventListener("close", () => {
    header.prepend(navigation);
    toggle.setAttribute("aria-expanded", "false");
    document.documentElement.classList.remove("menu-open");
  });
  document
    .querySelector("#menu-close")
    .addEventListener("click", () => menu.close());
  navigation.addEventListener("click", (event) => {
    if (menu.open && event.target.closest("a")) menu.close();
  });
  menu.addEventListener("click", (event) => {
    const bounds = menu.getBoundingClientRect();
    if (
      event.target === menu &&
      (event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom)
    )
      menu.close();
  });
  window
    .matchMedia("(min-width: 960px)")
    .addEventListener("change", (event) => {
      if (event.matches && menu.open) menu.close();
    });
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
  setupNavigation();
  const language = document.querySelector("#interface-language");
  language.value = locale;
  language.addEventListener("change", () => {
    try {
      localStorage.setItem(languagePreference, language.value);
    } catch {
      // Fall back to the current URL without changing project or study parameters.
    }
    const url = new URL(location.href);
    url.searchParams.set("lang", language.value);
    location.assign(url);
  });
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
