(function () {
  "use strict";

  const root = document.documentElement;
  const languageButtons = Array.from(document.querySelectorAll("[data-language-button]"));
  const descriptions = {
    zh: "BatteryGuard：用早期循环证据进行寿命预测、不确定性与 OOD 判断、安全充电策略仿真和可审计 synthetic 揭示演练的离线研究原型。",
    en: "BatteryGuard is an offline research prototype for early-life prediction, uncertainty and OOD, safe policy simulation, and an auditable synthetic reveal rehearsal.",
  };
  const titles = {
    zh: "BatteryGuard — 早期寿命智能与安全充电研究原型",
    en: "BatteryGuard — Early-life intelligence and safe charging research",
  };

  function setLanguage(language, persist) {
    const resolved = language === "en" ? "en" : "zh";
    root.dataset.language = resolved;
    root.lang = resolved === "en" ? "en" : "zh-CN";
    document.title = titles[resolved];
    const description = document.querySelector('meta[name="description"]');
    if (description) description.setAttribute("content", descriptions[resolved]);
    languageButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.languageButton === resolved));
    });
    if (persist) {
      try { window.localStorage.setItem("batteryguard-language", resolved); } catch (_) { /* local preference is optional */ }
    }
  }

  let savedLanguage = "zh";
  try { savedLanguage = window.localStorage.getItem("batteryguard-language") || "zh"; } catch (_) { /* storage may be disabled */ }
  setLanguage(savedLanguage, false);
  languageButtons.forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.languageButton, true));
  });

  const menuButton = document.querySelector(".menu-toggle");
  const navigation = document.querySelector(".site-nav");
  function closeMenu() {
    if (!menuButton || !navigation) return;
    menuButton.setAttribute("aria-expanded", "false");
    navigation.dataset.open = "false";
  }
  if (menuButton && navigation) {
    menuButton.addEventListener("click", () => {
      const open = menuButton.getAttribute("aria-expanded") !== "true";
      menuButton.setAttribute("aria-expanded", String(open));
      navigation.dataset.open = String(open);
    });
    navigation.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
    window.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenu(); });
  }

  const temperatureButtons = Array.from(document.querySelectorAll("[data-temperature]"));
  const strategyBoard = document.querySelector(".strategy-board");
  function setTemperature(temperature) {
    if (!strategyBoard) return;
    strategyBoard.dataset.activeTemperature = temperature;
    temperatureButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.temperature === temperature));
    });
    strategyBoard.querySelectorAll(".strategy-card").forEach((card) => {
      const state = card.getAttribute(`data-state-${temperature}`) || "allow";
      const status = card.querySelector(".strategy-status");
      const value = card.querySelector(".temperature-value");
      if (status) status.textContent = state.toUpperCase();
      if (value) value.innerHTML = `${value.getAttribute(`data-value-${temperature}`)} <small>°C</small>`;
    });
  }
  temperatureButtons.forEach((button) => {
    button.addEventListener("click", () => setTemperature(button.dataset.temperature || "25"));
  });

  const revealButton = document.querySelector(".reveal-button");
  const revealStage = document.querySelector(".reveal-stage");
  if (revealButton && revealStage) {
    revealButton.addEventListener("click", () => {
      const revealed = revealStage.dataset.revealed !== "true";
      revealStage.dataset.revealed = String(revealed);
      revealButton.setAttribute("aria-expanded", String(revealed));
    });
  }

  const year = document.getElementById("current-year");
  if (year) year.textContent = String(new Date().getFullYear());
})();
