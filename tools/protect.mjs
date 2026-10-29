#!/usr/bin/env node
// Password protection for blog posts on a static site.
//
// Each protected post's files (HTML body, video, ...) are encrypted with one random
// content key K (AES-256-GCM). K itself is stored only "wrapped": encrypted with a key
// derived from the password (PBKDF2-SHA256) in assets/protected/<post>/key.json.
// The browser (assets/js/protected.js) derives the same key from the typed password,
// unwraps K and decrypts the files. Changing the password only rewrites key.json.
//
// File format of *.enc: 12-byte IV || AES-GCM ciphertext || 16-byte tag (WebCrypto layout).
//
// Commands (keys/passwords are read from environment variables, never from argv):
//   genkey                              print a new random content key (base64)
//   encrypt <in> <out>                  PROTECT_KEY
//   decrypt <in> <out>                  PROTECT_KEY
//   wrap <key.json>                     PROTECT_KEY, PROTECT_PASSWORD   (write key.json)
//   change-password <key.json>          OLD_PASSWORD, NEW_PASSWORD      (no content key needed)
//   show-key <key.json>                 PROTECT_PASSWORD                (recover the content key)

import { webcrypto as crypto } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

const ITERATIONS = 600000;
const subtle = crypto.subtle;
const b64 = (buf) => Buffer.from(buf).toString("base64");
const unb64 = (s) => new Uint8Array(Buffer.from(s, "base64"));

function env(name) {
  const v = process.env[name];
  if (!v) throw new Error(`environment variable ${name} is required`);
  return v;
}

async function contentKey(raw) {
  const bytes = unb64(raw);
  if (bytes.length !== 32) throw new Error("PROTECT_KEY must be 32 bytes (base64)");
  return subtle.importKey("raw", bytes, "AES-GCM", true, ["encrypt", "decrypt"]);
}

async function passwordKey(password, salt, iterations) {
  const base = await subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveKey"]);
  return subtle.deriveKey(
    { name: "PBKDF2", hash: "SHA-256", salt, iterations },
    base,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

async function seal(key, data) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv }, key, data));
  const out = new Uint8Array(12 + ct.length);
  out.set(iv);
  out.set(ct, 12);
  return out;
}

async function open(key, data) {
  return new Uint8Array(await subtle.decrypt({ name: "AES-GCM", iv: data.slice(0, 12) }, key, data.slice(12)));
}

async function wrap(rawKey, password) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const kek = await passwordKey(password, salt, ITERATIONS);
  const sealed = await seal(kek, unb64(rawKey));
  return { v: 1, kdf: "PBKDF2-SHA256", iterations: ITERATIONS, salt: b64(salt), wrapped: b64(sealed) };
}

async function unwrap(file, password) {
  const meta = JSON.parse(readFileSync(file, "utf8"));
  const kek = await passwordKey(password, unb64(meta.salt), meta.iterations);
  try {
    return b64(await open(kek, unb64(meta.wrapped)));
  } catch {
    throw new Error("wrong password");
  }
}

const [cmd, a, b] = process.argv.slice(2);
const writeJson = (file, obj) => writeFileSync(file, JSON.stringify(obj, null, 2) + "\n");

switch (cmd) {
  case "genkey":
    console.log(b64(crypto.getRandomValues(new Uint8Array(32))));
    break;
  case "encrypt":
    writeFileSync(b, await seal(await contentKey(env("PROTECT_KEY")), readFileSync(a)));
    break;
  case "decrypt":
    writeFileSync(b, await open(await contentKey(env("PROTECT_KEY")), new Uint8Array(readFileSync(a))));
    break;
  case "wrap":
    await contentKey(env("PROTECT_KEY"));
    writeJson(a, await wrap(env("PROTECT_KEY"), env("PROTECT_PASSWORD")));
    break;
  case "change-password":
    writeJson(a, await wrap(await unwrap(a, env("OLD_PASSWORD")), env("NEW_PASSWORD")));
    console.log(`password changed for ${a}`);
    break;
  case "show-key":
    console.log(await unwrap(a, env("PROTECT_PASSWORD")));
    break;
  default:
    console.error("usage: protect.mjs genkey | encrypt <in> <out> | decrypt <in> <out> | wrap <key.json> | change-password <key.json> | show-key <key.json>");
    process.exit(1);
}
