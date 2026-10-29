// Unlocks password-protected posts encrypted with tools/protect.mjs.
// Markup: <div class="protected-post" data-protected-base="/assets/protected/<post>/"> with a form inside.
// The decrypted HTML may contain <video data-protected-src="video.mp4.enc" data-type="video/mp4">.
(function () {
  const subtle = window.crypto && window.crypto.subtle;
  const unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const b64 = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)));

  async function fetchBytes(url) {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error("HTTP " + res.status + " for " + url);
    return new Uint8Array(await res.arrayBuffer());
  }

  function open(key, data) {
    return subtle.decrypt({ name: "AES-GCM", iv: data.slice(0, 12) }, key, data.slice(12));
  }

  async function unwrapKey(base, password) {
    const meta = await (await fetch(base + "key.json", { cache: "no-cache" })).json();
    const pw = await subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveKey"]);
    const kek = await subtle.deriveKey(
      { name: "PBKDF2", hash: "SHA-256", salt: unb64(meta.salt), iterations: meta.iterations },
      pw,
      { name: "AES-GCM", length: 256 },
      false,
      ["decrypt"]
    );
    return open(kek, unb64(meta.wrapped)); // throws on a wrong password
  }

  async function reveal(root, base, rawKey) {
    const key = await subtle.importKey("raw", rawKey, "AES-GCM", false, ["decrypt"]);
    const html = new TextDecoder().decode(await open(key, await fetchBytes(base + "content.html.enc")));
    const body = root.querySelector(".protected-body");
    body.innerHTML = html;
    root.classList.add("is-unlocked");

    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([body]).catch(() => {});

    for (const video of body.querySelectorAll("video[data-protected-src]")) {
      const status = document.createElement("p");
      status.className = "protected-status";
      status.textContent = "视频解密中…";
      video.after(status);
      const data = await open(key, await fetchBytes(base + video.dataset.protectedSrc));
      video.src = URL.createObjectURL(new Blob([data], { type: video.dataset.type || "video/mp4" }));
      status.remove();
    }

    // Chapter links: <a data-seek="12.5">
    body.addEventListener("click", (e) => {
      const a = e.target.closest("[data-seek]");
      const video = body.querySelector("video");
      if (!a || !video) return;
      e.preventDefault();
      video.currentTime = parseFloat(a.dataset.seek);
      video.play();
      video.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  function init(root) {
    const base = root.dataset.protectedBase;
    const form = root.querySelector("form");
    const input = form.querySelector("input[type=password]");
    const error = root.querySelector(".protected-error");
    const button = form.querySelector("button");
    const storeKey = "protected:" + base;

    if (!subtle) {
      error.textContent = "当前浏览器不支持解密（需要 HTTPS 和较新的浏览器）。";
      return;
    }

    // Stay unlocked for the rest of this browser tab session.
    let saved = null;
    try {
      saved = sessionStorage.getItem(storeKey);
    } catch (e) {}
    if (saved) {
      reveal(root, base, unb64(saved)).catch(() => {
        try {
          sessionStorage.removeItem(storeKey);
        } catch (e) {}
        root.classList.remove("is-unlocked");
      });
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      error.textContent = "";
      button.disabled = true;
      button.textContent = "验证中…";
      try {
        const rawKey = await unwrapKey(base, input.value);
        try {
          sessionStorage.setItem(storeKey, b64(rawKey));
        } catch (err) {}
        await reveal(root, base, rawKey);
      } catch (err) {
        error.textContent = err && err.name === "OperationError" ? "密码错误，请重试。" : "加载失败：" + err.message;
      } finally {
        button.disabled = false;
        button.textContent = "解锁";
      }
    });
  }

  document.querySelectorAll(".protected-post").forEach(init);
})();
