// Optional integration example. Match these methods to your server, then select this provider.
export function createHttpStore(baseUrl = "/api/v1") {
  async function request(path, options = {}) {
    const { operationId, body, ...fetchOptions } = options;
    const headers = {};
    if (operationId) headers["Idempotency-Key"] = operationId;
    const multipart = body instanceof FormData;
    if (body && !multipart) headers["Content-Type"] = "application/json";
    let response;
    try {
      response = await fetch(baseUrl + path, {
        credentials: "same-origin",
        ...fetchOptions,
        headers,
        body: body ? (multipart ? body : JSON.stringify(body)) : undefined,
      });
    } catch {
      throw Object.assign(new Error("network"), { code: "network" });
    }
    let result;
    try {
      result = await response.json();
    } catch {
      throw Object.assign(new Error("saveFailed"), { code: "saveFailed" });
    }
    if (!response.ok)
      throw Object.assign(
        new Error("saveFailed"),
        { code: "saveFailed" },
        result.error,
      );
    return result;
  }
  return {
    capabilities: { persistence: "remote", linkScope: "shared" },
    ready: async () => {},
    createProject: (project, options) =>
      request("/projects", { method: "POST", body: project, ...options }),
    listProjects: async () => (await request("/projects")).items,
    getProject: async (id) =>
      (await request("/projects/" + encodeURIComponent(id))).project,
    findSession: (targetId, role) =>
      request(
        "/sessions/latest?target=" +
          encodeURIComponent(targetId) +
          "&role=" +
          encodeURIComponent(role),
      ),
    startSession: (input, options) =>
      request("/sessions", { method: "POST", body: input, ...options }),
    getSession: (id) => request("/sessions/" + encodeURIComponent(id)),
    saveRecording: (payload, options) => {
      const body = new FormData();
      body.append(
        "metadata",
        new Blob([JSON.stringify(payload.recording)], {
          type: "application/json",
        }),
      );
      body.append("audio", payload.blob);
      return request("/recordings", { method: "POST", body, ...options });
    },
    getStudy: async (id) =>
      (await request("/studies/" + encodeURIComponent(id))).study,
    resolveAudio: async (id) => ({
      ...(await request("/audio-assets/" + encodeURIComponent(id))),
      release() {},
    }),
    saveAnswer: (answer, options) =>
      request("/answers", { method: "POST", body: answer, ...options }),
    completeSession: (id) =>
      request("/sessions/" + encodeURIComponent(id) + "/complete", {
        method: "POST",
        body: {},
      }),
  };
}
