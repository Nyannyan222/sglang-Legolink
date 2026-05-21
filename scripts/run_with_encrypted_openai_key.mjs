#!/usr/bin/env node
// Run a child process with OPENAI_API_KEY decrypted in memory.
// This avoids writing plaintext keys to env files on locked/synced folders.

import fs from "node:fs";
import { spawn } from "node:child_process";
import { webcrypto } from "node:crypto";

const { subtle } = webcrypto;

function usage() {
  return `Usage:
node scripts/run_with_encrypted_openai_key.mjs \\
  --private-key <recipient-private-key.jwk.json> \\
  --ciphertext <base64url ciphertext> \\
  -- <command> [args...]

The script never prints the plaintext API key.`;
}

function parseArgs(argv) {
  const sep = argv.indexOf("--");
  if (sep < 0) {
    throw new Error(`Missing -- command separator.\n${usage()}`);
  }
  const options = argv.slice(0, sep);
  const command = argv.slice(sep + 1);
  const parsed = {};
  for (let i = 0; i < options.length; i += 1) {
    const name = options[i];
    const value = options[i + 1];
    if (!name.startsWith("--") || value == null || value.startsWith("--")) {
      throw new Error(`Invalid option near ${name}.\n${usage()}`);
    }
    parsed[name.slice(2)] = value;
    i += 1;
  }
  if (!parsed["private-key"] || !parsed.ciphertext) {
    throw new Error(`Missing --private-key or --ciphertext.\n${usage()}`);
  }
  if (command.length === 0) {
    throw new Error(`Missing child command.\n${usage()}`);
  }
  return { options: parsed, command };
}

async function decryptApiKey(privateKeyPath, ciphertext) {
  const privateJwk = JSON.parse(fs.readFileSync(privateKeyPath, "utf8"));
  const privateKey = await subtle.importKey(
    "jwk",
    privateJwk,
    { name: "RSA-OAEP", hash: "SHA-256" },
    false,
    ["decrypt"],
  );
  const plaintext = await subtle.decrypt(
    { name: "RSA-OAEP" },
    privateKey,
    Buffer.from(ciphertext, "base64url"),
  );
  const apiKey = new TextDecoder().decode(plaintext);
  if (!/^sk-[A-Za-z0-9_-]+$/.test(apiKey)) {
    throw new Error("Decrypted value is not a valid OpenAI API key literal.");
  }
  return apiKey;
}

async function main() {
  const { options, command } = parseArgs(process.argv.slice(2));
  const apiKey = await decryptApiKey(options["private-key"], options.ciphertext);
  const child = spawn(command[0], command.slice(1), {
    stdio: "inherit",
    shell: false,
    env: {
      ...process.env,
      OPENAI_API_KEY: apiKey,
    },
  });
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
