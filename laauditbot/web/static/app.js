/* global EventSource */

const projectListEl = document.getElementById("projectList");
const refreshBtn = document.getElementById("refreshBtn");
const newProjectBtn = document.getElementById("newProjectBtn");
const projectNameEl = document.getElementById("projectName");
const projectSummaryEl = document.getElementById("projectSummary");
const statusPillEl = document.getElementById("statusPill");
const settingsBtn = document.getElementById("settingsBtn");
const promptsBtn = document.getElementById("promptsBtn");
const auditPromptsBtn = document.getElementById("auditPromptsBtn");
const saveBtn = document.getElementById("saveBtn");
const runBtn = document.getElementById("runBtn");
const promptEl = document.getElementById("promptText");
const resultEl = document.getElementById("resultText");
const copyPromptBtn = document.getElementById("copyPromptBtn");
const copyResultBtn = document.getElementById("copyResultBtn");
const logOutputEl = document.getElementById("logOutput");
const clearLogViewBtn = document.getElementById("clearLogViewBtn");
const toastEl = document.getElementById("toast");
const unlockOverlayEl = document.getElementById("unlockOverlay");
const unlockPasswordEl = document.getElementById("unlockPassword");
const unlockBtn = document.getElementById("unlockBtn");
const unlockErrorEl = document.getElementById("unlockError");
const setupOverlayEl = document.getElementById("setupOverlay");
const setupApiKeyEl = document.getElementById("setupApiKey");
const setupPasswordEl = document.getElementById("setupPassword");
const setupPasswordConfirmEl = document.getElementById("setupPasswordConfirm");
const setupBtn = document.getElementById("setupBtn");
const setupErrorEl = document.getElementById("setupError");
const settingsOverlayEl = document.getElementById("settingsOverlay");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const restoreConfigDefaultsBtn = document.getElementById("restoreConfigDefaultsBtn");
const settingsErrorEl = document.getElementById("settingsError");

const cfgLogBatchesEl = document.getElementById("cfgLogBatches");
const cfgParallelRunsHintEl = document.getElementById("cfgParallelRunsHint");
const cfgFormatParallelEl = document.getElementById("cfgFormatParallel");
const cfgModelEl = document.getElementById("cfgModel");
const cfgReasoningEl = document.getElementById("cfgReasoning");
const cfgVerbosityEl = document.getElementById("cfgVerbosity");
const cfgMergeModelEl = document.getElementById("cfgMergeModel");
const cfgMergeReasoningEl = document.getElementById("cfgMergeReasoning");
const cfgMergeVerbosityEl = document.getElementById("cfgMergeVerbosity");
const cfgFormatModelEl = document.getElementById("cfgFormatModel");
const cfgFormatReasoningEl = document.getElementById("cfgFormatReasoning");
const cfgFormatVerbosityEl = document.getElementById("cfgFormatVerbosity");

const promptsOverlayEl = document.getElementById("promptsOverlay");
const closePromptsBtn = document.getElementById("closePromptsBtn");
const savePromptsBtn = document.getElementById("savePromptsBtn");
const restorePromptDefaultsBtn = document.getElementById("restorePromptDefaultsBtn");
const promptsErrorEl = document.getElementById("promptsError");
const sysPromptGenerationEl = document.getElementById("sysPromptGeneration");
const sysPromptMergeEl = document.getElementById("sysPromptMerge");
const sysPromptFormatEl = document.getElementById("sysPromptFormat");

const auditPromptsOverlayEl = document.getElementById("auditPromptsOverlay");
const closeAuditPromptsBtn = document.getElementById("closeAuditPromptsBtn");
const newAuditPromptBtn = document.getElementById("newAuditPromptBtn");
const auditPromptListEl = document.getElementById("auditPromptList");
const auditPromptNameEl = document.getElementById("auditPromptName");
const auditPromptTextEl = document.getElementById("auditPromptText");
const useAuditPromptBtn = document.getElementById("useAuditPromptBtn");
const deleteAuditPromptBtn = document.getElementById("deleteAuditPromptBtn");
const saveAuditPromptBtn = document.getElementById("saveAuditPromptBtn");
const auditPromptsErrorEl = document.getElementById("auditPromptsError");

const state = {
  projects: [],
  selectedId: null,
  eventSource: null,
  pollTimer: null,
  dirty: false,
  token: null,
  busy: false,
  currentProject: null,
  auditPrompts: [],
  activeAuditPromptId: null,
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (
    !headers["X-Audit-Token"] &&
    state.token &&
    typeof path === "string" &&
    path.startsWith("/api") &&
    !path.startsWith("/api/auth")
  ) {
    headers["X-Audit-Token"] = state.token;
  }
  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data && typeof data.detail === "string") {
        detail = data.detail;
      }
    } catch {
      // ignore
    }

    if (res.status === 401 && typeof path === "string" && !path.startsWith("/api/auth")) {
      clearToken();
      showUnlockOverlay(detail);
    }
    throw new Error(detail);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

function showToast(text) {
  toastEl.textContent = text;
  toastEl.classList.add("show");
  window.clearTimeout(showToast._t);
  showToast._t = window.setTimeout(() => toastEl.classList.remove("show"), 1600);
}

function setToken(token) {
  state.token = token;
  if (token) {
    window.sessionStorage.setItem("audit_token", token);
  }
}

function clearToken() {
  state.token = null;
  window.sessionStorage.removeItem("audit_token");
}

function showUnlockOverlay(message) {
  hideSetupOverlay();
  unlockErrorEl.textContent = message || "";
  unlockOverlayEl.classList.add("show");
  unlockOverlayEl.setAttribute("aria-hidden", "false");
  disableEditor(true);
  refreshBtn.disabled = true;
  newProjectBtn.disabled = true;
  window.setTimeout(() => unlockPasswordEl.focus(), 50);
}

function hideUnlockOverlay() {
  unlockErrorEl.textContent = "";
  unlockPasswordEl.value = "";
  unlockOverlayEl.classList.remove("show");
  unlockOverlayEl.setAttribute("aria-hidden", "true");
  refreshBtn.disabled = false;
  newProjectBtn.disabled = false;
}

function showSetupOverlay(message) {
  hideUnlockOverlay();
  setupErrorEl.textContent = message || "";
  setupOverlayEl.classList.add("show");
  setupOverlayEl.setAttribute("aria-hidden", "false");
  disableEditor(true);
  refreshBtn.disabled = true;
  newProjectBtn.disabled = true;
  window.setTimeout(() => setupApiKeyEl.focus(), 50);
}

function hideSetupOverlay() {
  setupErrorEl.textContent = "";
  setupApiKeyEl.value = "";
  setupPasswordEl.value = "";
  setupPasswordConfirmEl.value = "";
  setupOverlayEl.classList.remove("show");
  setupOverlayEl.setAttribute("aria-hidden", "true");
  refreshBtn.disabled = false;
  newProjectBtn.disabled = false;
}

function statusLabel(status) {
  if (!status) return "—";
  if (status === "idle") return "Idle";
  if (status === "running") return "Running";
  if (status === "success") return "Success";
  if (status === "error") return "Error";
  return String(status);
}

function phaseLabel(phase) {
  if (!phase) return "";
  if (phase === "starting") return "Starting";
  if (phase === "generating") return "Generating";
  if (phase === "merging") return "Merging";
  if (phase === "formatting") return "Formatting";
  return String(phase);
}

function isProjectLocked(project) {
  return Boolean(project && (project.locked || project.last_run_started_at));
}

function setBusy(isBusy) {
  const next = Boolean(isBusy);
  if (state.busy === next) return;
  state.busy = next;
  renderProjectList();
  refreshBtn.disabled = state.busy;
  newProjectBtn.disabled = state.busy;
}

function setStatusPill(status, phase, locked, errorText) {
  statusPillEl.classList.remove("running", "success", "error");
  const phaseText = phaseLabel(phase);
  if (status === "running") {
    statusPillEl.textContent = phaseText || statusLabel(status);
  } else {
    statusPillEl.textContent = statusLabel(status);
  }

  if (status === "running") statusPillEl.classList.add("running");
  if (status === "success") statusPillEl.classList.add("success");
  if (status === "error") statusPillEl.classList.add("error");

  if (status === "error" && errorText) {
    statusPillEl.title = errorText;
  } else {
    statusPillEl.title = "Status";
  }

  const running = status === "running";
  setBusy(running);
  if (!state.selectedId) {
    runBtn.disabled = true;
    runBtn.textContent = "Run";
    return;
  }
  if (running) {
    runBtn.disabled = true;
    runBtn.textContent = "Running…";
    return;
  }
  if (locked) {
    runBtn.disabled = true;
    runBtn.textContent = "Completed";
    return;
  }
  runBtn.disabled = false;
  runBtn.textContent = "Run";
}

function dotClass(status) {
  if (status === "running") return "dot running";
  if (status === "success") return "done-check";
  if (status === "error") return "dot error";
  return "dot";
}

function buildProjectSummary(project) {
  if (!project) return "—";
  const models = project.models || {};
  const parts = [];
  if (models.generation) parts.push(`Gen: ${models.generation}`);
  if (project.parallel_runs) parts.push(`Runs: ${project.parallel_runs}`);
  if (project.format_parallel_issues) parts.push(`Fmt: ${project.format_parallel_issues}`);
  const line1 = parts.join(" · ");

  const parts2 = [];
  if (models.merge) parts2.push(`Merge: ${models.merge}`);
  if (models.format) parts2.push(`Format: ${models.format}`);
  const line2 = parts2.join(" · ");

  if (line1 && line2) return `${line1} — ${line2}`;
  return line1 || line2 || "—";
}

function clearProjectList() {
  projectListEl.innerHTML = "";
}

function renderProjectList() {
  clearProjectList();

  if (!state.projects.length) {
    const empty = document.createElement("div");
    empty.className = "project-meta";
    empty.style.padding = "10px";
    empty.textContent = "No projects yet.";
    projectListEl.appendChild(empty);
    return;
  }

  for (const p of state.projects) {
    const row = document.createElement("div");
    row.className = "project-row" + (p.id === state.selectedId ? " selected" : "");
    row.dataset.projectId = p.id;

    const selectBtn = document.createElement("button");
    selectBtn.type = "button";
    selectBtn.className = "project-select";
    selectBtn.disabled = state.busy && p.id !== state.selectedId;

    const left = document.createElement("div");
    const name = document.createElement("div");
    name.className = "project-name";
    name.textContent = p.name || p.id;
    const meta1 = document.createElement("div");
    meta1.className = "project-meta";
    meta1.textContent = buildProjectSummary(p);
    const meta2 = document.createElement("div");
    meta2.className = "project-meta";
    const auditPromptName = (p.audit_prompt_used_name || "").trim();
    const meta2Text =
      p.locked && p.status !== "running" && auditPromptName ? `Audit: ${auditPromptName}` : "";
    if (meta2Text) {
      meta2.textContent = meta2Text;
    }
    left.appendChild(name);
    left.appendChild(meta1);
    if (meta2Text) left.appendChild(meta2);

    const side = document.createElement("div");
    side.className = "project-side";

    const dot = document.createElement("div");
    dot.className = dotClass(p.status);
    dot.title = statusLabel(p.status);
    dot.textContent = p.status === "success" ? "✓" : "";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "icon-btn";
    renameBtn.title = "Rename project";
    renameBtn.textContent = "✎";
    renameBtn.disabled = state.busy || p.status === "running";

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "icon-btn danger";
    delBtn.title = "Delete project";
    delBtn.textContent = "×";
    delBtn.disabled = state.busy || p.status === "running";

    selectBtn.appendChild(left);
    side.appendChild(dot);
    side.appendChild(renameBtn);
    side.appendChild(delBtn);
    row.appendChild(selectBtn);
    row.appendChild(side);

    selectBtn.addEventListener("click", async () => {
      if (p.id === state.selectedId) return;
      await selectProject(p.id);
    });

    delBtn.addEventListener("click", async () => {
      await deleteProject(p);
    });

    renameBtn.addEventListener("click", async () => {
      await renameProject(p);
    });

    projectListEl.appendChild(row);
  }
}

function disableEditor(disabled) {
  projectNameEl.disabled = disabled;
  promptEl.disabled = disabled;
  settingsBtn.disabled = disabled;
  promptsBtn.disabled = disabled;
  auditPromptsBtn.disabled = disabled;
  saveBtn.disabled = disabled;
  runBtn.disabled = disabled;
  copyPromptBtn.disabled = disabled;
  copyResultBtn.disabled = disabled;
}

function setEditor(project) {
  state.currentProject = project || null;
  projectNameEl.value = project?.name || "";
  promptEl.value = project?.prompt || "";
  resultEl.value = project?.result || "";
  const running = project?.status === "running";
  const locked = isProjectLocked(project);
  setStatusPill(project?.status, project?.phase, locked, project?.error);
  projectSummaryEl.textContent = buildProjectSummary(project);
  projectNameEl.disabled = running;
  promptEl.readOnly = locked;
  promptEl.disabled = false;
  settingsBtn.disabled = running;
  promptsBtn.disabled = running;
  auditPromptsBtn.disabled = running;
  saveBtn.disabled = running;
  state.dirty = false;
}

function appendLogLine(line) {
  const stick =
    logOutputEl.scrollTop + logOutputEl.clientHeight >= logOutputEl.scrollHeight - 24;

  logOutputEl.textContent += `${line}\n`;

  const maxChars = 220_000;
  if (logOutputEl.textContent.length > maxChars) {
    const tail = logOutputEl.textContent.slice(-180_000);
    const cut = tail.indexOf("\n");
    logOutputEl.textContent = cut >= 0 ? tail.slice(cut + 1) : tail;
  }

  if (stick) {
    logOutputEl.scrollTop = logOutputEl.scrollHeight;
  }
}

function connectLogs(projectId) {
  if (!state.token) return;
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  logOutputEl.textContent = "";
  const url = `/api/projects/${encodeURIComponent(projectId)}/logs/stream?tail=250&token=${encodeURIComponent(
    state.token,
  )}`;
  const es = new EventSource(url);
  state.eventSource = es;

  es.onmessage = (ev) => {
    if (typeof ev.data === "string") appendLogLine(ev.data);
  };

  es.onerror = () => {
    // Browser will auto-reconnect, but we keep a hint in the log view.
    appendLogLine("[ui] log stream disconnected; reconnecting…");
  };
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function startPolling(projectId) {
  stopPolling();
  state.pollTimer = window.setInterval(async () => {
    if (state.selectedId !== projectId) return;
    try {
      const project = await api(`/api/projects/${encodeURIComponent(projectId)}`);
      state.currentProject = project || null;
      // Only refresh non-editable fields while user may be typing.
      resultEl.value = project?.result || "";
      const locked = isProjectLocked(project);
      setStatusPill(project?.status, project?.phase, locked, project?.error);
      projectSummaryEl.textContent = buildProjectSummary(project);
      const running = project?.status === "running";
      projectNameEl.disabled = running;
      promptEl.readOnly = locked;
      promptEl.disabled = false;
      settingsBtn.disabled = running;
      promptsBtn.disabled = running;
      auditPromptsBtn.disabled = running;
      saveBtn.disabled = running;

      // Refresh sidebar status dots.
      await refreshProjects({ keepSelection: true, silent: true });

      if (project?.status !== "running") stopPolling();
    } catch (e) {
      stopPolling();
    }
  }, 2000);
}

async function loadProject(projectId, { resetDirty = true } = {}) {
  const project = await api(`/api/projects/${encodeURIComponent(projectId)}`);
  if (resetDirty) state.dirty = false;

  if (resetDirty || !state.dirty) {
    setEditor(project);
  } else {
    state.currentProject = project || null;
    // User has unsaved edits; update only the result + status.
    resultEl.value = project?.result || "";
    const locked = isProjectLocked(project);
    setStatusPill(project?.status, project?.phase, locked, project?.error);
    projectSummaryEl.textContent = buildProjectSummary(project);
    const running = project?.status === "running";
    projectNameEl.disabled = running;
    promptEl.readOnly = locked;
    promptEl.disabled = false;
    settingsBtn.disabled = running;
    promptsBtn.disabled = running;
    auditPromptsBtn.disabled = running;
    saveBtn.disabled = running;
  }

  if (project?.status === "running") startPolling(projectId);
  return project;
}

async function refreshProjects({ keepSelection = true, silent = false } = {}) {
  try {
    const projects = await api("/api/projects");
    state.projects = Array.isArray(projects) ? projects : [];
    if (!silent) renderProjectList();

    if (!keepSelection) return;
    if (!state.selectedId && state.projects.length) {
      await selectProject(state.projects[0].id);
    } else {
      renderProjectList();
    }
  } catch (e) {
    if (!silent) showToast(String(e.message || e));
  }
}

async function selectProject(projectId) {
  if (state.busy && projectId !== state.selectedId) {
    showToast("Busy: wait for current run to finish");
    return;
  }
  if (state.dirty) {
    const ok = window.confirm("Discard unsaved changes?");
    if (!ok) return;
  }
  state.selectedId = projectId;
  renderProjectList();
  disableEditor(false);
  connectLogs(projectId);
  await loadProject(projectId, { resetDirty: true });
}

async function createProject() {
  if (state.busy) {
    showToast("Busy: wait for current run to finish");
    return;
  }
  const name = window.prompt("New project name:", "");
  if (name === null) return;
  const meta = await api("/api/projects", {
    method: "POST",
    body: JSON.stringify({ name: name.trim() || undefined, prompt: "" }),
  });
  await refreshProjects({ keepSelection: false });
  if (meta?.id) {
    await selectProject(meta.id);
    showToast("Project created");
  }
}

async function deleteProject(project) {
  if (state.busy) {
    showToast("Busy: wait for current run to finish");
    return;
  }
  if (!project?.id) return;
  const label = project.name || project.id;
  const ok = window.confirm(`Delete project "${label}"?\n\nThis cannot be undone.`);
  if (!ok) return;

  try {
    await api(`/api/projects/${encodeURIComponent(project.id)}`, { method: "DELETE" });
    if (project.id === state.selectedId) {
      state.selectedId = null;
      state.dirty = false;
      stopPolling();
      if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
      }
      logOutputEl.textContent = "";
      disableEditor(true);
      projectSummaryEl.textContent = "—";
      statusPillEl.textContent = "—";
      runBtn.textContent = "Run";
      resultEl.value = "";
      promptEl.value = "";
      projectNameEl.value = "";
    }
    await refreshProjects({ keepSelection: true });
    showToast("Deleted");
  } catch (e) {
    showToast(String(e.message || e));
  }
}

async function renameProject(project) {
  if (state.busy) {
    showToast("Busy: wait for current run to finish");
    return;
  }
  if (!project?.id) return;
  const currentName = String(project.name || "").trim();
  const label = currentName || project.id;
  const next = window.prompt(`Rename project "${label}" to:`, currentName);
  if (next === null) return;
  const newName = next.trim();
  if (!newName) {
    showToast("Project name cannot be empty");
    return;
  }

  try {
    const updated = await api(`/api/projects/${encodeURIComponent(project.id)}`, {
      method: "PUT",
      body: JSON.stringify({ name: newName }),
    });
    if (project.id === state.selectedId) {
      if (state.currentProject) state.currentProject.name = updated?.name || newName;
      projectNameEl.value = updated?.name || newName;
    }
    await refreshProjects({ keepSelection: true, silent: true });
    showToast("Renamed");
  } catch (e) {
    showToast(String(e.message || e));
  }
}

async function cloneCurrentProjectForEdits() {
  const src = state.currentProject;
  if (!src?.id) return;

  if (state.busy || src.status === "running") {
    showToast("Busy: wait for current run to finish");
    return;
  }

  try {
    if (state.dirty) {
      await saveProject({ silent: true });
    }

    const baseName = (projectNameEl.value || src.name || src.id).trim() || "Project";
    const newName = `${baseName} (clone)`;
    const promptText = promptEl.value || src.prompt || "";

    const meta = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name: newName, prompt: promptText }),
    });

    const newId = meta?.id;
    if (!newId) throw new Error("Failed to create clone");

    try {
      const cfg = await api(`/api/projects/${encodeURIComponent(src.id)}/config`);
      if (cfg && typeof cfg === "object") {
        const cfgBody = { ...cfg };
        delete cfgBody.parallel_runs;
        await api(`/api/projects/${encodeURIComponent(newId)}/config`, {
          method: "PUT",
          body: JSON.stringify(cfgBody),
        });
      }
    } catch {
      // best-effort
    }

    try {
      const sp = await api(`/api/projects/${encodeURIComponent(src.id)}/system-prompts`);
      if (sp?.current) {
        await api(`/api/projects/${encodeURIComponent(newId)}/system-prompts`, {
          method: "PUT",
          body: JSON.stringify(sp.current),
        });
      }
    } catch {
      // best-effort
    }

    try {
      const auditPromptId = src.audit_prompt_id;
      if (auditPromptId) {
        await api(`/api/projects/${encodeURIComponent(newId)}/audit-prompt`, {
          method: "PUT",
          body: JSON.stringify({ audit_prompt_id: auditPromptId }),
        });
      }
    } catch {
      // best-effort
    }

    await refreshProjects({ keepSelection: false });
    await selectProject(newId);
    showToast("Cloned");
    promptEl.focus();
  } catch (e) {
    showToast(String(e.message || e));
  }
}

async function offerCloneBecauseLocked() {
  const src = state.currentProject;
  if (!isProjectLocked(src)) return;
  if (state.busy || src?.status === "running") {
    showToast("Audit is running; please wait");
    return;
  }

  const ok = window.confirm(
    "This audit is locked to preserve the original prompt and results.\n\n" +
      "Create a new project cloned from this prompt so you can make changes?\n\n" +
      "OK = Create clone\nCancel = Keep this audit unchanged",
  );
  if (!ok) return;
  await cloneCurrentProjectForEdits();
}

async function saveProject({ silent = false } = {}) {
  if (!state.selectedId) return;
  const locked = isProjectLocked(state.currentProject);
  const body = { name: projectNameEl.value };
  if (!locked) {
    body.prompt = promptEl.value;
  }
  await api(`/api/projects/${encodeURIComponent(state.selectedId)}`, {
    method: "PUT",
    body: JSON.stringify(body),
  });
  state.dirty = false;
  await refreshProjects({ keepSelection: true, silent: true });
  if (!silent) showToast("Saved");
}

async function runProject() {
  if (!state.selectedId) return;
  await saveProject({ silent: true });
  await api(`/api/projects/${encodeURIComponent(state.selectedId)}/run`, { method: "POST" });
  setBusy(true);
  showToast("Run started");
  await sleep(250);
  await loadProject(state.selectedId, { resetDirty: false });
  startPolling(state.selectedId);
  await refreshProjects({ keepSelection: true, silent: true });
}

function showOverlay(el) {
  el.classList.add("show");
  el.setAttribute("aria-hidden", "false");
}

function hideOverlay(el) {
  el.classList.remove("show");
  el.setAttribute("aria-hidden", "true");
}

function updateParallelRunsHint() {
  const logn = Number(cfgLogBatchesEl.value || 0);
  const n = Math.pow(2, Math.max(0, Math.min(12, logn)));
  cfgParallelRunsHintEl.textContent = `Parallel runs: ${n}`;
}

function fillConfigForm(cfg) {
  cfgLogBatchesEl.value = String(cfg?.log_number_of_batches ?? 3);
  cfgFormatParallelEl.value = String(cfg?.format_max_parallel_issues ?? 8);
  cfgModelEl.value = cfg?.model ?? "";
  cfgReasoningEl.value = cfg?.reasoning_effort ?? "high";
  cfgVerbosityEl.value = cfg?.verbosity ?? "high";
  cfgMergeModelEl.value = cfg?.merge_model ?? "";
  cfgMergeReasoningEl.value = cfg?.merge_reasoning_effort ?? "medium";
  cfgMergeVerbosityEl.value = cfg?.merge_verbosity ?? "high";
  cfgFormatModelEl.value = cfg?.format_model ?? "";
  cfgFormatReasoningEl.value = cfg?.format_reasoning_effort ?? "high";
  cfgFormatVerbosityEl.value = cfg?.format_verbosity ?? "high";
  updateParallelRunsHint();
}

function readConfigForm() {
  return {
    log_number_of_batches: Number(cfgLogBatchesEl.value || 0),
    model: cfgModelEl.value || "",
    reasoning_effort: cfgReasoningEl.value || "high",
    verbosity: cfgVerbosityEl.value || "high",
    merge_model: cfgMergeModelEl.value || "",
    merge_reasoning_effort: cfgMergeReasoningEl.value || "medium",
    merge_verbosity: cfgMergeVerbosityEl.value || "high",
    format_model: cfgFormatModelEl.value || "",
    format_reasoning_effort: cfgFormatReasoningEl.value || "high",
    format_verbosity: cfgFormatVerbosityEl.value || "high",
    format_max_parallel_issues: Number(cfgFormatParallelEl.value || 8),
  };
}

async function openSettings() {
  if (!state.selectedId || state.busy) return;
  settingsErrorEl.textContent = "";
  showOverlay(settingsOverlayEl);
  try {
    const cfg = await api(`/api/projects/${encodeURIComponent(state.selectedId)}/config`);
    fillConfigForm(cfg);
  } catch (e) {
    settingsErrorEl.textContent = String(e.message || e);
  }
}

async function saveSettings() {
  if (!state.selectedId || state.busy) return;
  settingsErrorEl.textContent = "";
  saveSettingsBtn.disabled = true;
  try {
    const body = readConfigForm();
    await api(`/api/projects/${encodeURIComponent(state.selectedId)}/config`, {
      method: "PUT",
      body: JSON.stringify(body),
    });
    await refreshProjects({ keepSelection: true, silent: true });
    await loadProject(state.selectedId, { resetDirty: false });
    showToast("Settings saved");
    hideOverlay(settingsOverlayEl);
  } catch (e) {
    settingsErrorEl.textContent = String(e.message || e);
  } finally {
    saveSettingsBtn.disabled = false;
  }
}

async function restoreConfigDefaults() {
  if (!state.selectedId || state.busy) return;
  const ok = window.confirm("Restore default settings?");
  if (!ok) return;
  settingsErrorEl.textContent = "";
  restoreConfigDefaultsBtn.disabled = true;
  try {
    const cfg = await api(
      `/api/projects/${encodeURIComponent(state.selectedId)}/config/restore-defaults`,
      { method: "POST" },
    );
    fillConfigForm(cfg);
    await refreshProjects({ keepSelection: true, silent: true });
    await loadProject(state.selectedId, { resetDirty: false });
    showToast("Defaults restored");
  } catch (e) {
    settingsErrorEl.textContent = String(e.message || e);
  } finally {
    restoreConfigDefaultsBtn.disabled = false;
  }
}

function fillPromptsForm(data) {
  const cur = data?.current || {};
  sysPromptGenerationEl.value = cur.generation || "";
  sysPromptMergeEl.value = cur.merge || "";
  sysPromptFormatEl.value = cur.format || "";
}

async function openPrompts() {
  if (!state.selectedId || state.busy) return;
  promptsErrorEl.textContent = "";
  showOverlay(promptsOverlayEl);
  try {
    const data = await api(`/api/projects/${encodeURIComponent(state.selectedId)}/system-prompts`);
    fillPromptsForm(data);
  } catch (e) {
    promptsErrorEl.textContent = String(e.message || e);
  }
}

async function savePrompts() {
  if (!state.selectedId || state.busy) return;
  promptsErrorEl.textContent = "";
  savePromptsBtn.disabled = true;
  try {
    const body = {
      generation: sysPromptGenerationEl.value,
      merge: sysPromptMergeEl.value,
      format: sysPromptFormatEl.value,
    };
    await api(`/api/projects/${encodeURIComponent(state.selectedId)}/system-prompts`, {
      method: "PUT",
      body: JSON.stringify(body),
    });
    showToast("Prompts saved");
    hideOverlay(promptsOverlayEl);
  } catch (e) {
    promptsErrorEl.textContent = String(e.message || e);
  } finally {
    savePromptsBtn.disabled = false;
  }
}

async function restorePromptDefaults() {
  if (!state.selectedId || state.busy) return;
  const ok = window.confirm("Restore default system prompts?");
  if (!ok) return;
  promptsErrorEl.textContent = "";
  restorePromptDefaultsBtn.disabled = true;
  try {
    const data = await api(
      `/api/projects/${encodeURIComponent(state.selectedId)}/system-prompts/restore-defaults`,
      { method: "POST" },
    );
    fillPromptsForm(data);
    showToast("Defaults restored");
  } catch (e) {
    promptsErrorEl.textContent = String(e.message || e);
  } finally {
    restorePromptDefaultsBtn.disabled = false;
  }
}

function auditPromptPreview(text) {
  const flat = String(text || "")
    .trim()
    .replace(/\s+/g, " ");
  if (!flat) return "—";
  const max = 84;
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

function renderAuditPromptList() {
  auditPromptListEl.innerHTML = "";

  const prompts = Array.isArray(state.auditPrompts) ? state.auditPrompts : [];
  if (!prompts.length) {
    const empty = document.createElement("div");
    empty.className = "project-meta";
    empty.style.padding = "10px";
    empty.textContent = "No audit prompts.";
    auditPromptListEl.appendChild(empty);
    return;
  }

  const selectedId = state.currentProject?.audit_prompt_id;

  for (const p of prompts) {
    const row = document.createElement("div");
    row.className = "project-row" + (p.id === state.activeAuditPromptId ? " selected" : "");

    const selectBtn = document.createElement("button");
    selectBtn.type = "button";
    selectBtn.className = "project-select";
    selectBtn.disabled = state.busy;

    const left = document.createElement("div");
    const name = document.createElement("div");
    name.className = "project-name";
    name.textContent = p.name || p.id;
    const meta1 = document.createElement("div");
    meta1.className = "project-meta";
    meta1.textContent = auditPromptPreview(p.prompt);
    left.appendChild(name);
    left.appendChild(meta1);
    selectBtn.appendChild(left);

    const side = document.createElement("div");
    side.className = "project-side";

    const dot = document.createElement("div");
    dot.className = p.id === selectedId ? "dot success" : "dot";
    dot.title = p.id === selectedId ? "Selected for this audit" : "Not selected";

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "icon-btn danger";
    delBtn.title = "Delete audit prompt";
    delBtn.textContent = "×";
    delBtn.disabled = state.busy;

    side.appendChild(dot);
    side.appendChild(delBtn);
    row.appendChild(selectBtn);
    row.appendChild(side);

    selectBtn.addEventListener("click", () => selectAuditPrompt(p.id));
    delBtn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      await deleteAuditPrompt(p);
    });

    auditPromptListEl.appendChild(row);
  }
}

function fillAuditPromptEditor(prompt) {
  auditPromptNameEl.value = prompt?.name || "";
  auditPromptTextEl.value = prompt?.prompt || "";

  const activeId = state.activeAuditPromptId;
  const selectedId = state.currentProject?.audit_prompt_id;
  const isSelected = Boolean(activeId && selectedId && activeId === selectedId);

  useAuditPromptBtn.textContent = isSelected ? "Selected" : "Use for this audit";
  useAuditPromptBtn.disabled = state.busy || !activeId || isSelected;
  saveAuditPromptBtn.disabled = state.busy || !activeId;
  deleteAuditPromptBtn.disabled = state.busy || !activeId;
}

function selectAuditPrompt(promptId) {
  state.activeAuditPromptId = promptId || null;
  const prompt = (state.auditPrompts || []).find((p) => p.id === state.activeAuditPromptId);
  fillAuditPromptEditor(prompt);
  renderAuditPromptList();
  auditPromptNameEl.focus();
}

async function openAuditPrompts() {
  if (!state.selectedId || state.busy) return;
  auditPromptsErrorEl.textContent = "";
  showOverlay(auditPromptsOverlayEl);

  try {
    const data = await api("/api/audit-prompts");
    state.auditPrompts = Array.isArray(data?.prompts) ? data.prompts : [];

    const selectedId = state.currentProject?.audit_prompt_id;
    const hasSelected = selectedId && state.auditPrompts.some((p) => p.id === selectedId);
    const firstId = state.auditPrompts[0]?.id;
    state.activeAuditPromptId = (hasSelected ? selectedId : firstId) || null;

    renderAuditPromptList();
    const prompt = (state.auditPrompts || []).find((p) => p.id === state.activeAuditPromptId);
    fillAuditPromptEditor(prompt);
  } catch (e) {
    auditPromptsErrorEl.textContent = String(e.message || e);
  }
}

async function createAuditPrompt() {
  if (state.busy) return;
  const name = window.prompt("New audit prompt name:", "");
  if (name === null) return;
  const trimmed = name.trim();
  if (!trimmed) {
    showToast("Name required");
    return;
  }

  auditPromptsErrorEl.textContent = "";
  newAuditPromptBtn.disabled = true;
  try {
    const created = await api("/api/audit-prompts", {
      method: "POST",
      body: JSON.stringify({ name: trimmed, prompt: "" }),
    });
    const data = await api("/api/audit-prompts");
    state.auditPrompts = Array.isArray(data?.prompts) ? data.prompts : [];
    state.activeAuditPromptId = created?.id || state.auditPrompts[0]?.id || null;
    renderAuditPromptList();
    const prompt = (state.auditPrompts || []).find((p) => p.id === state.activeAuditPromptId);
    fillAuditPromptEditor(prompt);
    showToast("Audit prompt created");
  } catch (e) {
    auditPromptsErrorEl.textContent = String(e.message || e);
  } finally {
    newAuditPromptBtn.disabled = false;
  }
}

async function saveAuditPrompt() {
  if (state.busy) return;
  const promptId = state.activeAuditPromptId;
  if (!promptId) return;

  const name = (auditPromptNameEl.value || "").trim();
  if (!name) {
    auditPromptsErrorEl.textContent = "Name required";
    return;
  }

  auditPromptsErrorEl.textContent = "";
  saveAuditPromptBtn.disabled = true;
  try {
    await api(`/api/audit-prompts/${encodeURIComponent(promptId)}`, {
      method: "PUT",
      body: JSON.stringify({ name, prompt: auditPromptTextEl.value || "" }),
    });
    const data = await api("/api/audit-prompts");
    state.auditPrompts = Array.isArray(data?.prompts) ? data.prompts : [];
    renderAuditPromptList();
    const prompt = (state.auditPrompts || []).find((p) => p.id === state.activeAuditPromptId);
    fillAuditPromptEditor(prompt);
    showToast("Saved");
  } catch (e) {
    auditPromptsErrorEl.textContent = String(e.message || e);
  } finally {
    saveAuditPromptBtn.disabled = false;
  }
}

async function deleteAuditPrompt(prompt) {
  if (state.busy) return;
  if (!prompt?.id) return;
  const label = prompt.name || prompt.id;
  const ok = window.confirm(`Delete audit prompt "${label}"?\n\nThis cannot be undone.`);
  if (!ok) return;

  auditPromptsErrorEl.textContent = "";
  try {
    await api(`/api/audit-prompts/${encodeURIComponent(prompt.id)}`, { method: "DELETE" });
    const data = await api("/api/audit-prompts");
    state.auditPrompts = Array.isArray(data?.prompts) ? data.prompts : [];

    if (state.activeAuditPromptId === prompt.id) {
      const selectedId = state.currentProject?.audit_prompt_id;
      const hasSelected = selectedId && state.auditPrompts.some((p) => p.id === selectedId);
      state.activeAuditPromptId = (hasSelected ? selectedId : state.auditPrompts[0]?.id) || null;
    }

    renderAuditPromptList();
    const next = (state.auditPrompts || []).find((p) => p.id === state.activeAuditPromptId);
    fillAuditPromptEditor(next);
    showToast("Deleted");
  } catch (e) {
    auditPromptsErrorEl.textContent = String(e.message || e);
  }
}

async function useActiveAuditPromptForProject() {
  if (!state.selectedId || state.busy) return;
  const promptId = state.activeAuditPromptId;
  if (!promptId) return;

  auditPromptsErrorEl.textContent = "";
  useAuditPromptBtn.disabled = true;
  try {
    const updated = await api(`/api/projects/${encodeURIComponent(state.selectedId)}/audit-prompt`, {
      method: "PUT",
      body: JSON.stringify({ audit_prompt_id: promptId }),
    });
    if (state.currentProject) {
      state.currentProject.audit_prompt_id = updated?.audit_prompt_id || promptId;
    }
    renderAuditPromptList();
    const prompt = (state.auditPrompts || []).find((p) => p.id === state.activeAuditPromptId);
    fillAuditPromptEditor(prompt);
    showToast("Selected");
    await refreshProjects({ keepSelection: true, silent: true });
  } catch (e) {
    auditPromptsErrorEl.textContent = String(e.message || e);
  } finally {
    const prompt = (state.auditPrompts || []).find((p) => p.id === state.activeAuditPromptId);
    fillAuditPromptEditor(prompt);
  }
}

async function copyFrom(el) {
  const text = el.value || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied");
  } catch {
    showToast("Copy failed");
  }
}

function bindEvents() {
  refreshBtn.addEventListener("click", () => refreshProjects({ keepSelection: true }));
  newProjectBtn.addEventListener("click", () => createProject());
  saveBtn.addEventListener("click", () => saveProject());
  runBtn.addEventListener("click", () => runProject());
  settingsBtn.addEventListener("click", () => openSettings());
  promptsBtn.addEventListener("click", () => openPrompts());
  auditPromptsBtn.addEventListener("click", () => openAuditPrompts());

  projectNameEl.addEventListener("input", () => (state.dirty = true));
  promptEl.addEventListener("input", () => (state.dirty = true));

  const maybeOfferClone = (ev) => {
    const src = state.currentProject;
    if (!isProjectLocked(src)) return;
    if (ev) ev.preventDefault();
    offerCloneBecauseLocked();
  };

  promptEl.addEventListener("keydown", (ev) => {
    const src = state.currentProject;
    if (!isProjectLocked(src)) return;
    if (state.busy || src?.status === "running") return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

    const nonEditKeys = new Set([
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "ArrowDown",
      "Shift",
      "Control",
      "Alt",
      "Meta",
      "Escape",
      "Tab",
      "Home",
      "End",
      "PageUp",
      "PageDown",
    ]);
    if (nonEditKeys.has(ev.key)) return;

    // Any other key would be an edit attempt (text input, delete, backspace, enter, etc).
    maybeOfferClone(ev);
  });

  promptEl.addEventListener("paste", (ev) => maybeOfferClone(ev));
  promptEl.addEventListener("drop", (ev) => maybeOfferClone(ev));

  copyPromptBtn.addEventListener("click", () => copyFrom(promptEl));
  copyResultBtn.addEventListener("click", () => copyFrom(resultEl));

  clearLogViewBtn.addEventListener("click", () => {
    logOutputEl.textContent = "";
    showToast("Cleared");
  });

  closeSettingsBtn.addEventListener("click", () => hideOverlay(settingsOverlayEl));
  saveSettingsBtn.addEventListener("click", () => saveSettings());
  restoreConfigDefaultsBtn.addEventListener("click", () => restoreConfigDefaults());
  cfgLogBatchesEl.addEventListener("input", () => updateParallelRunsHint());

  closePromptsBtn.addEventListener("click", () => hideOverlay(promptsOverlayEl));
  savePromptsBtn.addEventListener("click", () => savePrompts());
  restorePromptDefaultsBtn.addEventListener("click", () => restorePromptDefaults());

  closeAuditPromptsBtn.addEventListener("click", () => hideOverlay(auditPromptsOverlayEl));
  newAuditPromptBtn.addEventListener("click", () => createAuditPrompt());
  saveAuditPromptBtn.addEventListener("click", () => saveAuditPrompt());
  deleteAuditPromptBtn.addEventListener("click", async () => {
    const p = (state.auditPrompts || []).find((x) => x.id === state.activeAuditPromptId);
    await deleteAuditPrompt(p);
  });
  useAuditPromptBtn.addEventListener("click", () => useActiveAuditPromptForProject());

  const doUnlock = async () => {
    unlockErrorEl.textContent = "";
    unlockBtn.disabled = true;
    try {
      const password = unlockPasswordEl.value || "";
      const data = await api("/api/auth/unlock", {
        method: "POST",
        body: JSON.stringify({ password }),
      });
      if (data && data.token) {
        setToken(data.token);
        hideUnlockOverlay();
        await refreshProjects({ keepSelection: true });
        showToast("Unlocked");
      } else {
        throw new Error("Unlock failed");
      }
    } catch (e) {
      const msg = String(e.message || e);
      if (msg.includes("Missing OPENAI.API_KEY file")) {
        showSetupOverlay("No `OPENAI.API_KEY` found. Create one below.");
      } else {
        unlockErrorEl.textContent = msg;
      }
    } finally {
      unlockBtn.disabled = false;
      unlockPasswordEl.focus();
    }
  };

  const doSetup = async () => {
    setupErrorEl.textContent = "";
    setupBtn.disabled = true;
    try {
      const apiKey = (setupApiKeyEl.value || "").trim();
      const password = setupPasswordEl.value || "";
      const confirm = setupPasswordConfirmEl.value || "";

      if (!apiKey) throw new Error("API key required");
      if (!password.trim()) throw new Error("Password required");
      if (password !== confirm) throw new Error("Passwords do not match");

      const data = await api("/api/auth/setup", {
        method: "POST",
        body: JSON.stringify({ api_key: apiKey, password, password_confirm: confirm }),
      });
      if (data && data.token) {
        setToken(data.token);
        hideSetupOverlay();
        await refreshProjects({ keepSelection: true });
        showToast("Unlocked");
      } else {
        throw new Error("Setup failed");
      }
    } catch (e) {
      setupErrorEl.textContent = String(e.message || e);
    } finally {
      setupBtn.disabled = false;
      setupApiKeyEl.focus();
    }
  };

  unlockBtn.addEventListener("click", () => doUnlock());
  unlockPasswordEl.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") doUnlock();
  });

  setupBtn.addEventListener("click", () => doSetup());
  setupApiKeyEl.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") doSetup();
  });
  setupPasswordEl.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") doSetup();
  });
  setupPasswordConfirmEl.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") doSetup();
  });
}

async function init() {
  disableEditor(true);
  bindEvents();

  const stored = window.sessionStorage.getItem("audit_token");
  if (stored) {
    state.token = stored;
    try {
      await refreshProjects({ keepSelection: true });
      hideUnlockOverlay();
      return;
    } catch {
      clearToken();
    }
  }

  try {
    const status = await api("/api/auth/status");
    if (status && status.unlocked) {
      const tokenData = await api("/api/auth/token", { method: "POST" });
      if (tokenData && tokenData.token) {
        setToken(tokenData.token);
        hideUnlockOverlay();
        hideSetupOverlay();
        await refreshProjects({ keepSelection: true });
        return;
      }
    }
  } catch {
    // fall through to unlock modal
  }

  try {
    const status = await api("/api/auth/status");
    if (status && status.has_key_file === false) {
      showSetupOverlay("");
      return;
    }
  } catch {
    // ignore
  }

  showUnlockOverlay("");
}

init();
