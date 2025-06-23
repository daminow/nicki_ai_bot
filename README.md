# Nicki AI – Telegram Mental‑Health Assistant

Nicki AI is an inline‑keyboard driven Telegram bot that provides **psychological first‑aid** and coaching powered by the Novita LLM API.

## Features

- 🌐 Multilingual on‑boarding (English/Russian, easily extendable through `i18n`).
- 🗣 Voice & voice‑note understanding via Whisper (automatic conversion with FFmpeg).
- 💬 Contextual conversations stored in PostgreSQL for continuity.
- 🔒 Secure verification: e‑mail one‑time code sent from `nickiai@daminov.net`.
- 🛠 Self‑service settings – reset dialogue or update personal data through inline menus or slash‑commands.
- 🐳 Fully containerised with Docker Compose; single‑command deploy on Ubuntu VM.

## Quick start

```bash
git clone https://github.com/yourusername/nicki_ai_bot
cd nicki_ai_bot
cp .env.example .env   # add your secrets
docker compose up --build -d
```
