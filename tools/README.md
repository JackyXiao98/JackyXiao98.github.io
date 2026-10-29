# Password-protected posts

Protected posts (e.g. `_posts/2026-10-29-diffusion-intro.md`) are encrypted with
AES-256-GCM. Their files live in `assets/protected/<post>/`:

- `content.html.enc`, `video.mp4.enc` – the encrypted post body and video
- `key.json` – the content key, encrypted with the password (PBKDF2-SHA256)

The browser decrypts everything after the reader types the password
(`assets/js/protected.js`). Without the password the files are unreadable, even though
the repository is public. Editable sources are kept encrypted in `_protected_src/`.

Default password: `checkpass`.

## Change the password (recommended: GitHub secrets, works even if you forgot it)

One-time setup: in the repository go to **Settings → Secrets and variables → Actions**
and add the secret `BLOG_CONTENT_KEY` (the content key you were given when the post was
created).

Then, whenever you want a new password:

1. Add or update the secret `BLOG_PASSWORD` with the new password.
2. Open **Actions → Deploy site → Run workflow** (branch `gh-pages`).

The deploy re-wraps every `assets/protected/*/key.json` with `BLOG_PASSWORD`. If
`BLOG_CONTENT_KEY` is not set, the committed `key.json` files (password `checkpass`)
are used unchanged.

## Change the password locally (needs the old password)

```sh
OLD_PASSWORD=checkpass NEW_PASSWORD='new password' \
  node tools/protect.mjs change-password assets/protected/diffusion-intro/key.json
```

Commit and push `key.json`. If `BLOG_CONTENT_KEY` is set, also update the
`BLOG_PASSWORD` secret, since the deploy would otherwise overwrite this change.

## Recover the content key (needs the current password)

```sh
PROTECT_PASSWORD=checkpass node tools/protect.mjs show-key assets/protected/diffusion-intro/key.json
```

## Edit or add protected content

```sh
export PROTECT_KEY='<content key>'
node tools/protect.mjs decrypt _protected_src/diffusion-intro.tar.gz.enc src.tgz   # editable sources
node tools/protect.mjs encrypt content.html assets/protected/diffusion-intro/content.html.enc
node tools/protect.mjs encrypt video.mp4    assets/protected/diffusion-intro/video.mp4.enc
```

For a new post, copy `key.json` into `assets/protected/<new-post>/` (same content key),
encrypt its files there, and put `{% include protected.liquid id="<new-post>" %}` in
the post.
