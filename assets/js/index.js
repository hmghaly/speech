import { store } from "./storage.js";
import { setup, content, textNode, showError, status, locale } from "./ui.js";

const list = document.querySelector("#projects-list");
const error = document.querySelector("#page-error");
const reload = document.querySelector("#reload");
const pageStatus = document.querySelector("#page-status");

// Renders saved project summaries without interpreting any user-supplied HTML.
async function loadProjects() {
  reload.hidden = true;
  error.hidden = true;
  status(pageStatus, "loading");
  try {
    const projects = await store.listProjects();
    list.replaceChildren();
    document.querySelector("#local-link-note").hidden =
      !projects.length || store.capabilities.linkScope !== "browser";
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
      const illustration = document.createElement("img");
      illustration.className = "empty-projects-art";
      illustration.src = "./assets/images/empty-projects.svg";
      illustration.alt = "";
      illustration.width = 180;
      illustration.height = 128;
      illustration.setAttribute("aria-hidden", "true");
      empty.append(illustration, copy);
      list.append(empty);
    }
    for (const project of projects) {
      const row = document.createElement("article");
      row.className = "project-row";
      const prefix = "ui-projects-" + project.id;
      const details = document.createElement("div");
      const heading = document.createElement("h3");
      heading.className = "project-title";
      const title = content("a", project.title, prefix + "-title");
      title.href = "./record.html?project=" + encodeURIComponent(project.id);
      heading.append(title);
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
      details.append(heading, metadata);
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
      row.append(details, actions);
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
