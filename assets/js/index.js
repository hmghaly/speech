import { store } from "./storage.js";
import { setup, content, textNode, showError, status, locale } from "./ui.js";

const list = document.querySelector("#projects-list");
const error = document.querySelector("#page-error");
const reload = document.querySelector("#reload");
const pageStatus = document.querySelector("#page-status");

// This decorative document symbol contains fixed SVG only, never project content.
function projectSymbol(className) {
  const symbol = document.createElement("span");
  symbol.className = className;
  symbol.setAttribute("aria-hidden", "true");
  symbol.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2h9l5 5v15H5V2Zm9 0v5h5M8 11h8m-8 4h8m-8 4h5"/></svg>';
  return symbol;
}

// Renders saved project summaries without interpreting any user-supplied HTML.
async function loadProjects() {
  reload.hidden = true;
  error.hidden = true;
  status(pageStatus, "loading");
  try {
    const projects = await store.listProjects();
    list.replaceChildren();
    if (!projects.length) {
      const empty = document.createElement("div");
      empty.className = "empty-library";
      const copy = document.createElement("div");
      const create = textNode(
        "a",
        "nav.create",
        {},
        "ui-projects-empty-create",
      );
      create.className = "button";
      create.href = "./project.html";
      copy.append(
        textNode("h2", "projects.empty", {}, "ui-projects-empty-title"),
        textNode("p", "projects.emptyHelp", {}, "ui-projects-empty-help"),
        create,
      );
      empty.append(projectSymbol("empty-symbol"), copy);
      list.append(empty);
    }
    for (const project of projects) {
      const row = document.createElement("article");
      row.className = "project-row";
      const prefix = "ui-projects-" + project.id;
      const details = document.createElement("div");
      const title = content("h3", project.title, prefix + "-title");
      title.className = "project-title";
      const date = new Intl.DateTimeFormat(locale, {
        dateStyle: "medium",
      }).format(new Date(project.createdAt));
      const count = textNode(
        "p",
        "projects.count",
        { count: project.sentenceCount ?? project.sentences.length },
        prefix + "-count",
        "ui-projects-{id}-count",
      );
      count.className = "helper";
      const metadata = document.createElement("div");
      metadata.className = "project-meta";
      metadata.append(
        count,
        textNode(
          "p",
          "projects.saved",
          { date },
          prefix + "-date",
          "ui-projects-{id}-date",
        ),
      );
      details.append(title, metadata);
      const actions = document.createElement("div");
      actions.className = "controls shrink-0";
      const open = textNode(
        "a",
        "projects.open",
        {},
        prefix + "-open",
        "ui-projects-{id}-open",
      );
      open.className = "button";
      open.href = "./record.html?project=" + encodeURIComponent(project.id);
      const copy = textNode(
        "button",
        "projects.copy",
        {},
        prefix + "-copy",
        "ui-projects-{id}-copy",
      );
      copy.className = "button subtle";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(new URL(open.href).href);
          status(pageStatus, "projects.copied");
        } catch {
          status(pageStatus, "projects.copyFailed");
        }
      });
      actions.append(open, copy);
      row.append(projectSymbol("project-symbol"), details, actions);
      list.append(row);
    }
    status(pageStatus, "projects.loaded", { count: projects.length });
    if (!projects.length) pageStatus.textContent = "";
  } catch (failure) {
    showError(error, failure);
    reload.hidden = false;
    pageStatus.textContent = "";
  }
}

reload.addEventListener("click", loadProjects);
try {
  await setup(store);
  document.querySelector("#local-link-note").hidden =
    store.capabilities.linkScope !== "browser";
  await loadProjects();
} catch (failure) {
  showError(error, failure);
  reload.hidden = false;
}
