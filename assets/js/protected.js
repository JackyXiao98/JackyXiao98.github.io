// Unlocks password-protected content encrypted with tools/protect.mjs.
//
// Markup: <div class="protected-post" data-protected-base="/assets/protected/<id>/" data-protected-app="post|videos">
// containing a password form and an empty .protected-body.
//   post   – decrypts content.html.enc into the body (videos inside use data-protected-src).
//   videos – the Knowledge Video library: decrypts catalog.json.enc, lists the videos and plays
//            <video-id>/video.mp4.enc with <video-id>/content.html.enc as notes (route: #<video-id>).
(function () {
  const subtle = window.crypto && window.crypto.subtle;
  const unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const b64 = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)));
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

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

  const decryptBytes = async (key, url) => open(key, await fetchBytes(url));
  const decryptText = async (key, url) => new TextDecoder().decode(await decryptBytes(key, url));
  const typeset = (el) => {
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([el]).catch(() => {});
  };
  const fmtTime = (t) => Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");

  function seekHandler(body) {
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

  // ---------------------------------------------------------------------------
  // Single protected post
  // ---------------------------------------------------------------------------
  async function renderPost(body, base, key) {
    body.innerHTML = await decryptText(key, base + "content.html.enc");
    typeset(body);
    seekHandler(body);
    for (const video of body.querySelectorAll("video[data-protected-src]")) {
      const data = await decryptBytes(key, base + video.dataset.protectedSrc);
      video.src = URL.createObjectURL(new Blob([data], { type: video.dataset.type || "video/mp4" }));
    }
  }

  // ---------------------------------------------------------------------------
  // Knowledge Video library
  // ---------------------------------------------------------------------------
  async function renderVideos(body, base, key) {
    const catalog = JSON.parse(await decryptText(key, base + "catalog.json.enc"));
    const blobs = {};
    const blobUrl = async (path, type) => {
      if (!blobs[path]) blobs[path] = URL.createObjectURL(new Blob([await decryptBytes(key, base + path)], { type }));
      return blobs[path];
    };
    seekHandler(body);

    function listView() {
      const cards = catalog.videos
        .map(
          (v) => `
        <a class="kv-card tech-card" href="#${esc(v.id)}">
          <div class="kv-thumb" data-thumb="${esc(v.id)}/thumb.jpg.enc"><span class="kv-play"><i class="ti ti-player-play"></i></span>
            <span class="kv-duration">${esc(v.duration)}</span></div>
          <div class="kv-card-body">
            <h3>${esc(v.title)}</h3>
            <p>${esc(v.summary)}</p>
            <div class="kv-tags">${(v.tags || []).map((t) => `<span>${esc(t)}</span>`).join("")}</div>
          </div>
        </a>`
        )
        .join("");
      const upcoming = (catalog.upcoming || [])
        .map(
          (u, i) => `
        <li><span class="kv-num">${String(catalog.videos.length + i + 1).padStart(2, "0")}</span>
          <div><strong>${esc(u.title)}</strong><p>${esc(u.summary)}</p></div></li>`
        )
        .join("");
      body.innerHTML = `
        <p class="kv-intro">${esc(catalog.intro || "")}</p>
        <div class="kv-grid">${cards}</div>
        ${upcoming ? `<h2 class="kv-section">制作计划</h2><ol class="kv-upcoming">${upcoming}</ol>` : ""}`;
      body.querySelectorAll("[data-thumb]").forEach(async (el) => {
        try {
          el.style.backgroundImage = `url(${await blobUrl(el.dataset.thumb, "image/jpeg")})`;
        } catch (e) {}
      });
    }

    async function detailView(v) {
      const chapters = (v.chapters || [])
        .map((c) => `<li><a href="#" data-seek="${c.t}">${fmtTime(c.t)}</a> ${esc(c.title)}</li>`)
        .join("");
      body.innerHTML = `
        <a class="kv-back" href="#"><i class="ti ti-arrow-left"></i> 全部视频</a>
        <h2 class="kv-title">${esc(v.title)}</h2>
        <video controls playsinline class="protected-video"></video>
        <p class="protected-status">视频解密中…</p>
        ${chapters ? `<h3>章节</h3><ol class="protected-chapters">${chapters}</ol>` : ""}
        <div class="kv-notes"></div>`;
      window.scrollTo({ top: 0 });
      const notes = body.querySelector(".kv-notes");
      decryptText(key, base + v.id + "/content.html.enc")
        .then((html) => {
          notes.innerHTML = html;
          typeset(notes);
        })
        .catch(() => {});
      const video = body.querySelector("video");
      video.src = await blobUrl(v.id + "/video.mp4.enc", "video/mp4");
      const status = body.querySelector(".protected-status");
      if (status) status.remove();
    }

    function route() {
      const id = decodeURIComponent(location.hash.slice(1));
      const v = catalog.videos.find((x) => x.id === id);
      (v ? detailView(v) : Promise.resolve(listView())).catch((e) => {
        body.innerHTML = `<p class="protected-error">加载失败：${esc(e.message)}</p>`;
      });
    }
    window.addEventListener("hashchange", route);
    route();
  }

  const APPS = { post: renderPost, videos: renderVideos };

  function init(root) {
    const base = root.dataset.protectedBase;
    const app = APPS[root.dataset.protectedApp || "post"];
    const form = root.querySelector("form");
    const input = form.querySelector("input[type=password]");
    const error = root.querySelector(".protected-error");
    const button = form.querySelector("button");
    const body = root.querySelector(".protected-body");
    const storeKey = "protected:" + base;

    if (!subtle) {
      error.textContent = "当前浏览器不支持解密（需要 HTTPS 和较新的浏览器）。";
      return;
    }

    async function reveal(rawKey) {
      const key = await subtle.importKey("raw", rawKey, "AES-GCM", false, ["decrypt"]);
      root.classList.add("is-unlocked");
      await app(body, base, key);
    }

    // Stay unlocked for the rest of this browser tab session.
    let saved = null;
    try {
      saved = sessionStorage.getItem(storeKey);
    } catch (e) {}
    if (saved) {
      reveal(unb64(saved)).catch(() => {
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
        await reveal(rawKey);
      } catch (err) {
        root.classList.remove("is-unlocked");
        error.textContent = err && err.name === "OperationError" ? "密码错误，请重试。" : "加载失败：" + err.message;
      } finally {
        button.disabled = false;
        button.textContent = "解锁";
      }
    });
  }

  document.querySelectorAll(".protected-post").forEach(init);
})();
