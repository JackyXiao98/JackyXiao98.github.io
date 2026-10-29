# Password-protected posts

Protected content is encrypted with AES-256-GCM. The Knowledge Video library
(`/blog/knowledge-video/`, entry card on the blog page) lives in `assets/protected/videos/`:

- `key.json` – the content key, encrypted with the password (PBKDF2-SHA256)
- `catalog.json.enc` – the list of videos (title, summary, chapters) and the production plan
- `<video-id>/video.mp4.enc`, `content.html.enc` (notes), `thumb.jpg.enc` (cover)

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
  node tools/protect.mjs change-password assets/protected/videos/key.json
```

Commit and push `key.json`. If `BLOG_CONTENT_KEY` is set, also update the
`BLOG_PASSWORD` secret, since the deploy would otherwise overwrite this change.

## Recover the content key (needs the current password)

```sh
PROTECT_PASSWORD=checkpass node tools/protect.mjs show-key assets/protected/videos/key.json
```

## Add a video to the library

```sh
export PROTECT_KEY='<content key>'
D=assets/protected/videos
node tools/protect.mjs decrypt $D/catalog.json.enc catalog.json      # add an entry to "videos"
node tools/protect.mjs encrypt catalog.json $D/catalog.json.enc
mkdir -p $D/<video-id>
node tools/protect.mjs encrypt video.mp4    $D/<video-id>/video.mp4.enc
node tools/protect.mjs encrypt notes.html   $D/<video-id>/content.html.enc
node tools/protect.mjs encrypt thumb.jpg    $D/<video-id>/thumb.jpg.enc
```

The video generator and narration sources are kept encrypted in `_protected_src/`
(`node tools/protect.mjs decrypt _protected_src/videos-diffusion-intro.tar.gz.enc src.tgz`).

## Protect a single post instead

Put `{% include protected.liquid id="<name>" %}` in the post, copy `key.json` into
`assets/protected/<name>/` and encrypt the post body there as `content.html.enc`.
