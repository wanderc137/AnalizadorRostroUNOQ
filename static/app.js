/* AnalizadorRostroUNOQ — frontend sin dependencias ni build step. */

(() => {
  "use strict";

  const els = {
    stream: document.getElementById("stream"),
    overlay: document.getElementById("stream-overlay"),
    faces: document.getElementById("faces-count"),
    hudFps: document.getElementById("hud-fps"),
    badgeCamera: document.getElementById("badge-camera"),
    badgeModel: document.getElementById("badge-model"),
    badgeFps: document.getElementById("badge-fps"),
    statusCamera: document.getElementById("status-camera"),
    statusBackend: document.getElementById("status-backend"),
    statusModel: document.getElementById("status-model"),
    statusResolution: document.getElementById("status-resolution"),
    statusInfer: document.getElementById("status-infer"),
    statusFps: document.getElementById("status-fps"),
    message: document.getElementById("status-message"),
    btnStart: document.getElementById("btn-start"),
    btnStop: document.getElementById("btn-stop"),
    btnSnapshot: document.getElementById("btn-snapshot"),
    legendEvery: document.getElementById("legend-every"),
    toast: document.getElementById("toast"),
    authRow: document.getElementById("auth-row"),
    apiToken: document.getElementById("api-token"),
  };

  const BACKEND_LABELS = {
    utkface_onnx: "UTKFace ONNX",
    caffe: "Caffe · Levi & Hassner",
    none: "ninguno (degradado)",
  };

  let config = null;
  let toastTimer = null;
  let retryTimer = null;
  let lastCameraOk = null;
  let apiToken = "";

  try {
    apiToken = window.sessionStorage.getItem("analizadorApiToken") || "";
  } catch (_) {
    apiToken = "";
  }

  function toast(text, kind = "info") {
    if (!els.toast) return;
    els.toast.textContent = text;
    els.toast.dataset.kind = kind;
    els.toast.dataset.visible = "true";
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      els.toast.dataset.visible = "false";
    }, 4200);
  }

  function setBadge(el, text, state) {
    if (!el) return;
    el.textContent = text;
    el.classList.remove("badge--idle", "badge--ok", "badge--warn", "badge--bad");
    el.classList.add(state ? `badge--${state}` : "badge--idle");
  }

  function showOverlay(text) {
    if (!els.overlay) return;
    if (text) {
      els.overlay.textContent = text;
      els.overlay.hidden = false;
    } else {
      els.overlay.hidden = true;
    }
  }

  function friendlyError(error, fallback) {
    if (error && error.status === 401) {
      return "Token de API requerido o inválido. Revisa el campo Token API.";
    }
    if (error && error.status === 429) {
      return error.message || "Espera unos segundos antes de otro snapshot.";
    }
    if (error && error.status === 503) {
      return `Servicio no disponible: ${error.message}`;
    }
    return `${fallback}: ${error && error.message ? error.message : "error desconocido"}`;
  }

  function persistToken(value) {
    apiToken = value.trim();
    try {
      if (apiToken) {
        window.sessionStorage.setItem("analizadorApiToken", apiToken);
      } else {
        window.sessionStorage.removeItem("analizadorApiToken");
      }
    } catch (_) {
      /* sessionStorage puede estar bloqueado; el token vive solo en memoria */
    }
  }

  async function requestJSON(url, options = {}) {
    const headers = {
      Accept: "application/json",
      ...(options.headers || {}),
    };
    if (apiToken) {
      headers["X-Api-Token"] = apiToken;
    }
    const response = await fetch(url, {
      cache: "no-store",
      ...options,
      headers,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
      const error = new Error(detail);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function attachStream() {
    if (!els.stream) return;
    window.clearTimeout(retryTimer);
    els.stream.src = `/video?t=${Date.now()}`;
  }

  function scheduleRetry() {
    window.clearTimeout(retryTimer);
    retryTimer = window.setTimeout(attachStream, 3500);
  }

  function applyStatus(status) {
    const cameraOk = Boolean(status.camera_ok);
    const modelLoaded = Boolean(status.model_loaded);
    const fps = Number(status.fps || 0);
    const faces = Number(status.faces_count || 0);
    const backend = status.model_backend || "none";
    const requested = status.model_backend_requested || "auto";

    els.faces.textContent = String(faces);
    els.hudFps.textContent = fps.toFixed(1);
    els.statusFps.textContent = `${fps.toFixed(1)} fps`;
    els.statusCamera.textContent = cameraOk ? "Activa" : "Sin señal";
    els.statusBackend.textContent =
      BACKEND_LABELS[backend] || `${backend} (pedido: ${requested})`;
    els.statusModel.textContent = modelLoaded ? "Cargado" : "No cargado";
    const resolution = Array.isArray(status.resolution) ? status.resolution : [];
    els.statusResolution.textContent = resolution.length === 2
      ? `${resolution[0]} × ${resolution[1]}`
      : "—";
    const every = config && config.infer_every_n ? config.infer_every_n : 5;
    els.statusInfer.textContent = `1 de cada ${every} frames`;

    setBadge(
      els.badgeCamera,
      cameraOk ? "Cámara activa" : "Cámara offline",
      cameraOk ? "ok" : "warn"
    );
    setBadge(
      els.badgeModel,
      modelLoaded ? BACKEND_LABELS[backend] || backend : "Sin modelo",
      modelLoaded ? "ok" : "bad"
    );
    setBadge(els.badgeFps, `FPS ${fps.toFixed(1)}`, fps > 1 ? "ok" : "warn");

    const pipelineError = status.pipeline_error ? String(status.pipeline_error) : "";
    const statusMessage = pipelineError && status.message === "OK"
      ? `Aviso del pipeline: ${pipelineError}`
      : status.message || "—";
    els.message.textContent = statusMessage;
    els.message.classList.toggle(
      "message--warn",
      Boolean(status.degraded) || !cameraOk || !modelLoaded || Boolean(pipelineError)
    );

    if (lastCameraOk !== null && lastCameraOk !== cameraOk) {
      attachStream();
    }
    lastCameraOk = cameraOk;

    if (!cameraOk) {
      showOverlay("Cámara no disponible. Conecta la Logitech Brio o pulsa Iniciar cámara.");
    } else {
      showOverlay("");
    }

    els.btnStart.disabled = cameraOk;
    els.btnStop.disabled = !cameraOk;
  }

  async function refreshStatus() {
    try {
      const status = await requestJSON("/api/status");
      applyStatus(status);
    } catch (error) {
      setBadge(els.badgeCamera, "Sin conexión", "bad");
      setBadge(els.badgeModel, "Sin conexión", "bad");
      setBadge(els.badgeFps, "FPS —", "warn");
      els.message.textContent = `No se pudo consultar /api/status: ${error.message}`;
      els.message.classList.add("message--warn");
    }
  }

  async function loadConfig() {
    try {
      config = await requestJSON("/api/config");
      if (els.legendEvery && config.infer_every_n) {
        els.legendEvery.textContent = String(config.infer_every_n);
      }
      if (els.btnSnapshot) {
        els.btnSnapshot.hidden = !config.snap_enabled;
        els.btnSnapshot.title = config.snap_enabled
          ? "Guardar un JPG en la carpeta configurada"
          : "Configura SNAP_FOLDER para habilitar snapshots";
      }
      if (els.authRow) {
        els.authRow.hidden = !config.auth_enabled;
      }
      if (els.apiToken && config.auth_enabled) {
        els.apiToken.value = apiToken;
      }
    } catch (error) {
      toast(`No se pudo leer /api/config: ${error.message}`, "error");
    }
  }

  function handleAuthError(error) {
    if (error && error.status === 401 && els.apiToken) {
      els.apiToken.focus();
    }
  }

  function bindControls() {
    if (els.apiToken) {
      els.apiToken.value = apiToken;
      els.apiToken.addEventListener("input", () => persistToken(els.apiToken.value));
      els.apiToken.addEventListener("change", () => persistToken(els.apiToken.value));
    }

    els.btnStart.addEventListener("click", async () => {
      els.btnStart.disabled = true;
      try {
        await requestJSON("/api/camera/start", { method: "POST" });
        toast("Cámara iniciada");
        attachStream();
        await refreshStatus();
      } catch (error) {
        handleAuthError(error);
        toast(friendlyError(error, "No se pudo iniciar la cámara"), "error");
        els.btnStart.disabled = false;
      }
    });

    els.btnStop.addEventListener("click", async () => {
      els.btnStop.disabled = true;
      try {
        await requestJSON("/api/camera/stop", { method: "POST" });
        toast("Cámara detenida");
        await refreshStatus();
      } catch (error) {
        handleAuthError(error);
        toast(friendlyError(error, "No se pudo detener la cámara"), "error");
      } finally {
        els.btnStop.disabled = false;
      }
    });

    els.btnSnapshot.addEventListener("click", async () => {
      els.btnSnapshot.disabled = true;
      try {
        const result = await requestJSON("/api/snapshot", { method: "POST" });
        toast(`Snapshot guardado en ${result.path}`);
      } catch (error) {
        handleAuthError(error);
        toast(friendlyError(error, "Snapshot no disponible"), "error");
      } finally {
        els.btnSnapshot.disabled = false;
      }
    });

    if (els.stream) {
      els.stream.addEventListener("error", () => {
        showOverlay("Sin señal de vídeo. Reintentando…");
        scheduleRetry();
      });
      els.stream.addEventListener("load", () => {
        if (lastCameraOk) showOverlay("");
      });
    }
  }

  async function init() {
    bindControls();
    await loadConfig();
    attachStream();
    await refreshStatus();
    window.setInterval(refreshStatus, 1200);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
