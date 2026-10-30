# Password-protected posts

Protected content is encrypted with AES-256-GCM. A protected post puts
`{% include protected.liquid id="<name>" %}` in its body; its files live in
`assets/protected/<name>/`:

- `key.json` – the content key, encrypted with the password (PBKDF2-SHA256)
- `content.html.enc` – the encrypted post body (videos inside use `data-protected-src`)

The browser decrypts them after the reader types the password (`assets/js/protected.js`).
Without the password the files are unreadable, even though the repository is public.

The Knowledge Video library (`/blog/knowledge-video/`) is public: its files are plain in
`assets/videos/` (`catalog.json`, and `<video-id>/video.mp4`, `content.html`, `thumb.jpg`).
The video generator sources stay encrypted in `_protected_src/`.

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
  node tools/protect.mjs change-password assets/protected/<name>/key.json
```

Commit and push `key.json`. If `BLOG_CONTENT_KEY` is set, also update the
`BLOG_PASSWORD` secret, since the deploy would otherwise overwrite this change.

## Recover the content key (needs the current password)

```sh
PROTECT_PASSWORD=checkpass node tools/protect.mjs show-key assets/protected/<name>/key.json
```

## Add a video to the library

Copy the files to `assets/videos/<video-id>/` (`video.mp4`, `content.html`, `thumb.jpg`) and
add an entry (id, title, summary, duration, tags, chapters) to `assets/videos/catalog.json`.

Decrypt the generator sources with
`PROTECT_KEY='<content key>' node tools/protect.mjs decrypt _protected_src/knowledge-video-src.tar.gz.enc src.tgz`.

## Protect a post

Put `{% include protected.liquid id="<name>" %}` in the post, create `assets/protected/<name>/key.json`
with `wrap` (or copy an existing one) and encrypt the post body there as `content.html.enc`.
