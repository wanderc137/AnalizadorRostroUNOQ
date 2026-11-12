/* Comunidad del proyecto: FAQ y monedas anónimas. */
(() => {
  "use strict";

  const CFG = window.PORTFOLIO_COMMUNITY || {};
  const account = CFG.account || "wanderc137";
  const repo = CFG.repo || location.pathname.split("/").filter(Boolean)[0] || "proyecto";
  const coinNamespace = CFG.coinNamespace || `${account}-${repo}`;
  const coinApi = (CFG.coinApi || "https://abacus.jasoncameron.dev").replace(/\/$/, "");
  const faqs = Array.isArray(CFG.faqs) ? CFG.faqs : [];

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  const faqHtml = faqs.map((item, index) => `
    <details class="psc-faq-item"${index === 0 ? " open" : ""}>
      <summary>
        <span>${esc(item.q)}</span>
        <span class="psc-plus" aria-hidden="true">+</span>
      </summary>
      <div class="psc-faq-body">${item.a}</div>
    </details>`).join("");

  const section = document.createElement("section");
  section.className = "psc-section";
  section.id = "comunidad";
  section.innerHTML = `
    <div class="wrap">
      <div class="psc-head">
        <div class="psc-kicker">Comunidad · ${esc(repo)}</div>
        <h2>Preguntas reales.<br><span class="psc-grad">Reconocimiento anónimo.</span></h2>
        <p>Lo que más se pregunta sobre este proyecto y una moneda simbólica que no queda atada a tu cuenta.</p>
      </div>

      <div class="psc-faq" aria-label="Preguntas frecuentes">
        ${faqHtml}
      </div>

      <div class="psc-coin-wrap">
        <aside class="psc-card psc-coin" aria-labelledby="psc-coin-title">
          <div class="psc-coin-top">
            <div class="psc-coin-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="#5a3c00" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="9"></circle>
                <path d="M12 6.5v11M9 9.2h4.2a2.3 2.3 0 0 1 0 4.6H9.8a2.3 2.3 0 0 0 0 4.6H14"></path>
              </svg>
            </div>
            <div class="psc-coin-count" id="psc-coin-count" aria-live="polite">—</div>
            <div class="psc-coin-label" id="psc-coin-title">monedas anónimas</div>
          </div>
          <p class="psc-coin-copy">No es una estrella: no queda asociada a tu usuario. Es un contador global y anónimo para reconocer el proyecto. Podés regalar varias.</p>
          <button class="psc-btn primary psc-coin-btn" type="button" id="psc-coin-btn">Regalar 1 moneda</button>
          <p class="psc-coin-msg" id="psc-coin-msg" role="status" aria-live="polite">Cargando el contador…</p>
          <p class="psc-coin-small">Cada clic suma una moneda. <strong>Sin cuenta, sin correo, sin tracking.</strong></p>
        </aside>
      </div>
    </div>`;

  const main = document.querySelector("main");
  const cta = main && main.querySelector(".cta");
  if (main && cta) main.insertBefore(section, cta);
  else if (main) main.appendChild(section);
  else document.body.appendChild(section);

  const coinCount = section.querySelector("#psc-coin-count");
  const coinBtn = section.querySelector("#psc-coin-btn");
  const coinMsg = section.querySelector("#psc-coin-msg");

  const setCoinMessage = (text, kind = "") => {
    coinMsg.textContent = text;
    coinMsg.className = `psc-coin-msg${kind ? ` ${kind}` : ""}`;
  };

  const formatCoins = (n) => {
    try { return new Intl.NumberFormat("es-CL").format(n); }
    catch { return String(n); }
  };

  const loadCoins = async () => {
    try {
      const res = await fetch(`${coinApi}/get/${encodeURIComponent(coinNamespace)}/coins`, { cache: "no-store" });
      if (res.status === 404) {
        coinCount.textContent = "0";
        setCoinMessage("Todavía no hay monedas. Podés ser la primera persona en regalar una.");
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (typeof data.value !== "number") throw new Error("respuesta inválida");
      coinCount.textContent = formatCoins(data.value);
      setCoinMessage(`${formatCoins(data.value)} ${data.value === 1 ? "reconocimiento anónimo" : "reconocimientos anónimos"}.`);
    } catch {
      coinCount.textContent = "—";
      setCoinMessage("El contador no está disponible ahora. Podés intentar de nuevo en un momento.", "bad");
    }
  };

  let coinBusy = false;
  coinBtn.addEventListener("click", async () => {
    if (coinBusy) return;
    coinBusy = true;
    coinBtn.disabled = true;
    setCoinMessage("Registrando tu moneda…");
    try {
      const res = await fetch(`${coinApi}/hit/${encodeURIComponent(coinNamespace)}/coins`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (typeof data.value !== "number") throw new Error("respuesta inválida");
      coinCount.textContent = formatCoins(data.value);
      setCoinMessage("Moneda registrada. Gracias por reconocer el proyecto.", "ok");
      coinBtn.textContent = "Regalar otra moneda";
      coinCount.animate(
        [{ transform: "scale(1)" }, { transform: "scale(1.12)" }, { transform: "scale(1)" }],
        { duration: 420, easing: "ease-out" }
      );
    } catch {
      setCoinMessage("No pudimos registrar la moneda. Revisá la conexión e intentá otra vez.", "bad");
    } finally {
      setTimeout(() => {
        coinBusy = false;
        coinBtn.disabled = false;
        if (coinBtn.textContent !== "Regalar otra moneda") coinBtn.textContent = "Regalar 1 moneda";
      }, 1100);
    }
  });

  loadCoins();
})();
