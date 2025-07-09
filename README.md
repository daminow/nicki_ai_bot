# Nicki AI 2.0 🦋

**Nicki AI** is a friendly, CBT-inspired Telegram bot that listens, asks helpful
questions and offers coping techniques to deal with stress, anxiety or low mood.
Built with **aiogram 3**, **PostgreSQL** and **Novita AI** LLM.

---

## ✨ Features

| Area                      | Highlights                                                                                    |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| 🗂️ **Inline UX**          | All menus are inline buttons with emoji. Slash-commands mirror key actions.                   |
| 📝 **On-boarding**        | Conversational registration (name → choose email / phone verification).                       |
| 🎙 **Voice & Video Notes** | Whisper speech-to-text; only voice is extracted from circle videos.                           |
| 💾 **Database-first**     | Users, roles, messages and invites persist in PostgreSQL.                                     |
| ⚙️ **Settings**           | Change language, email, phone, name, or wipe personal data.                                   |
| 🔄 **Reset**              | One-tap dialog wipe that forgets conversation history.                                        |
| 🛠 **Admin Panel**         | Daily/weekly/monthly stats, add/remove admins, inline dashboard.                              |
| 📑 **Logs**               | Rotated daily (`logs/bot.log`) with 14-day retention (handled by `TimedRotatingFileHandler`). |
| 🐳 **Docker-ready**       | Single command `docker-compose up -d` launches both bot and PostgreSQL.                       |

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourname/nickiai.git
cd nickiai

# copy & edit env
cp .env.example .env
nano .env   # fill tokens, SMTP etc.

# build & run
docker-compose up -d
```
