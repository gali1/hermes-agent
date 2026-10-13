<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**The self-improving AI agent built by [Nous Research](https://nousresearch.com).** It's the only agent with a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from Telegram while it works on a cloud VM.

Use any model you want — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, your own endpoint, and [many others](https://hermes-agent.nousresearch.com/docs/integrations/providers). Switch with `hermes model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Seven terminal backends — local, Docker, SSH, Singularity, Modal, Daytona, and Vercel Sandbox. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

### Supported platforms and prerequisites

| Platform | Recommended install | Python | Required system tools |
| --- | --- | --- | --- |
| **Linux** (x86_64, aarch64) | `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash` | None from you — the installer provisions its own pinned interpreter | `git`, `curl`, `tar`, and `sha256sum` (or `shasum`); glibc hosts may need `libatomic.so.1`, musl hosts need the system `libstdc++` |
| **macOS** (Apple Silicon; Intel x86_64 CLI installs supported) | Same `install.sh` one-liner | Same as Linux | Same as Linux |
| **Windows 10/11** (x86_64, aarch64) | PowerShell one-liner below (MSIX desktop package requires Windows 11 22H2+) | Same — no system Python needed | Git (a verified portable Git is staged automatically if absent); Windows PowerShell 5.1+ or pwsh 7 |
| **WSL2** | The Linux one-liner, run inside the distro | Same as Linux | Same as Linux; enable systemd for gateway services |

**Distro notes:** the POSIX installer supports glibc and musl x86_64/aarch64 hosts and is tested on the latest Ubuntu and WSL2; any distro with glibc, systemd, and Filesystem Hierarchy Standard conventions is likely to work. Android/Termux uses its own signed APT package, not this script.

> **Python version:** current first-party installations run on **Python 3.14**. `pyproject.toml` declares `requires-python = ">=3.11,<3.15"` only so a source install that predates the PM migration can run `hermes update` once on an older interpreter; every runtime dependency carries a `python_version >= '3.14'` marker, so an install on 3.11–3.13 resolves with **no dependencies** and does not import. The installer (and `hermes update`) provisions the managed 3.14 environment for you; only manual installs need to supply 3.14.

### Linux, macOS, WSL2

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (native, PowerShell)

> **Heads up:** Native Windows runs Hermes without WSL — CLI, gateway, TUI, and tools all work natively. If you'd rather use WSL2, the Linux/macOS one-liner above works there too. Found a bug? Please [file issues](https://github.com/NousResearch/hermes-agent/issues).

Run this in PowerShell:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

The source installer delegates Python 3.14, Node.js, npm, ripgrep, FFmpeg,
and Python dependencies to PM. If Git is absent, it stages the verified Git
for Windows archive in Hermes' tool store. It does not replace your system Git.
See [installation methods](https://hermes-agent.nousresearch.com/docs/getting-started/installation)
for the separate MSIX/App Installer package and its update ownership.

> **Android / Termux:** A signed APT repository is available for aarch64 devices, with a `stable` channel (tagged releases) and a prerelease `canary` channel. The package includes Python, Node.js, and the TUI. Use the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux), not the desktop/server installer script.
>
> **Windows:** Native Windows is fully supported — the PowerShell one-liner above installs everything. If you'd rather use WSL2, the Linux command works there too. Native Windows install lives under `%LOCALAPPDATA%\hermes`; WSL2 installs under `~/.hermes` as on Linux.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

### Python below the required version (uv pathway)

Check the interpreter your shell sees:

```bash
python3 --version
```

Hermes runs on **Python 3.14**. On 3.11–3.13 a manual install either fails `requires-python` or resolves with no dependencies at all, because every runtime dependency in `pyproject.toml` is gated behind `python_version >= '3.14'`. The supported fix is the one-line installer, which provisions its own pinned interpreter — no system Python upgrade needed. For a manual source checkout, [uv](https://docs.astral.sh/uv/) can install a managed 3.14 alongside your system Python without replacing it.

Install uv:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Provision 3.14 and create the environment:

```bash
uv python install 3.14          # managed interpreter; system Python untouched
uv venv --python 3.14 .venv     # create .venv
source .venv/bin/activate       # bash / zsh
source .venv/bin/activate.fish  # fish
```

```powershell
.venv\Scripts\Activate.ps1      # Windows PowerShell
```

Install from a clone. Editable installs go through `build_editable`, so the wheel/sdist guard described under [Reinstalling or upgrading](#reinstalling-or-upgrading-over-an-existing-install) does not apply:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
uv pip install -e .
```

For a non-editable install, opt in to the build the same way that section documents:

```bash
HERMES_ALLOW_WHEEL_BUILD=1 uv pip install .
```

Run it without activating anything — uv reuses or creates the project `.venv`:

```bash
uv run --python 3.14 hermes
```

Prepare config and secrets (the installer normally does this for you):

```bash
mkdir -p ~/.hermes
cp .env.example ~/.hermes/.env          # secrets
cp cli-config.yaml.example ~/.hermes/config.yaml
chmod 600 ~/.hermes/.env
hermes setup                            # optional guided configuration
```

Verify the install:

```bash
hermes --version                                # version + update status
command -v hermes                               # Windows: where.exe hermes
python -c "import sys; print(sys.executable)"   # should point inside .venv
```

> A bare `uv pip install` does not prepare PM's managed tool store. Run `hermes pm install` from the checkout to provision Node, ripgrep, FFmpeg, and the browser tools. PyPI installs (`pip install hermes-agent`, `uv tool install hermes-agent`, Homebrew) remain unsupported — see [platform support](https://hermes-agent.nousresearch.com/docs/getting-started/platform-support).

### Reinstalling or upgrading over an existing install

`pip install .` installs into the Python environment you run it from. If
`hermes-agent` already exists in that environment, pip uninstalls the old copy
first and replaces it, including the `hermes` launcher. Your config, sessions,
skills, and memory under `~/.hermes` are not touched — it is a code-only swap.

```bash
# from a clone of this repo, using the environment that owns the current hermes
HERMES_ALLOW_WHEEL_BUILD=1 pip install .   # add --force-reinstall for a clean swap
hermes --version     # verify
```

Notes:

- Upstream blocks wheel/sdist builds by default; this fork restores source
  installs with `HERMES_ALLOW_WHEEL_BUILD=1` (the artifact is permitted but
  remains an unsupported distribution method).
- It only replaces the copy in **that** environment. The one-line installer keeps
  its managed copy in `~/.hermes/hermes-agent/venv`; installing elsewhere creates
  a second copy and whichever `hermes` comes first on `PATH` wins. For managed
  installs, use `hermes update` instead.
- Stop a running gateway (`hermes gateway stop`) before replacing files.
- System Python often rejects installs with `externally-managed-environment`
  (PEP 668). Use a virtualenv rather than `--break-system-packages`.
- pip/PyPI installs are **not an officially supported distribution method**. See
  [platform support](https://hermes-agent.nousresearch.com/docs/getting-started/platform-support)
  for the supported paths (installer, Docker, Nix).

### System-wide installs

For a machine-wide install every user can run, give Hermes its own virtualenv
and link the launcher into `PATH` — don't install into the system Python:

```bash
# as an admin user
sudo python3 -m venv /opt/hermes/venv
sudo env HERMES_ALLOW_WHEEL_BUILD=1 /opt/hermes/venv/bin/pip install /path/to/hermes-agent
sudo ln -sf /opt/hermes/venv/bin/hermes /usr/local/bin/hermes
```

Each user then gets their own state directory (`~/.hermes`), so configs,
sessions, and credentials stay per-user. Set `HERMES_HOME=/some/path` to relocate
a user's state (for example, a service account).

Other system-wide options:

- **Docker** (supported): `docker pull nousresearch/hermes-agent:latest`, with
  `$HERMES_HOME` bind-mounted from the host so state survives image upgrades.
- **pipx / uv tool** (per-user global CLI): `pipx install .` or
  `uv tool install .` — isolated and on `PATH`, but still an unsupported pip/PyPI
  distribution method. Both build a wheel; add `HERMES_ALLOW_WHEEL_BUILD=1` if you
  accept the unsupported method.
- **NixOS**: declarative module, see the
  [Nix setup guide](https://hermes-agent.nousresearch.com/docs/getting-started/nix-setup).

### Troubleshooting

#### Windows Defender or antivirus flags `uv.exe` as malware

If your antivirus (Bitdefender, Windows Defender, etc.) quarantines `uv.exe` from the Hermes `bin` folder (`%LOCALAPPDATA%\hermes\bin\uv.exe`), this is a **false positive**. The file is Astral's `uv` — the Rust Python package manager Hermes bundles to manage its Python environment. ML-based antivirus engines commonly flag unsigned Rust binaries that download and install packages.

**To verify your copy is authentic:**

```powershell
# Install GitHub CLI if needed
winget install --id GitHub.cli

# Login to GitHub
gh auth login

# Run verification
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

If attestation says "Verification succeeded" and the last line prints `True`, you're good.

**To whitelist Hermes:**
- **Windows Defender:** Run PowerShell as Admin → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Add an exception in the Bitdefender console (Protection > Antivirus > Settings > Manage Exceptions)
- Whitelist the **folder**, not the file hash — Hermes updates `uv` and the hash changes every version

For more context, see the upstream Astral reports: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

#### `hermes` is not on PATH

Reload your shell (or open a new terminal). POSIX installs link the launcher at `~/.local/bin/hermes`; the installer adds that directory to your shell rc files and prints the exact reload command when a running shell cannot see it. Native Windows installs add `%LOCALAPPDATA%\hermes\bin` to the user PATH. Check with `command -v hermes` (Windows: `where.exe hermes`) and run `hermes doctor`.

#### Python is older than 3.14, or pip says `externally-managed-environment`

The installer provisions its own pinned Python 3.14, so the system interpreter does not need upgrading to use Hermes. For manual source installs, provision 3.14 without replacing the system Python — see [Python below the required version](#python-below-the-required-version-uv-pathway). Never use `--break-system-packages` against the system Python; `externally-managed-environment` (PEP 668) means that interpreter belongs to your OS package manager.

#### The wheel/sdist build is blocked

`pip install .` / `uv pip install .` fail with "Building wheels or sdists for hermes-agent is not supported" unless `HERMES_ALLOW_WHEEL_BUILD=1` is set. Upstream blocks the artifact because it would ship without bundled assets; this fork keeps the opt-in: `HERMES_ALLOW_WHEEL_BUILD=1 pip install .`. Editable installs (`pip install -e .`, `uv pip install -e .`, `uv sync`) use `build_editable` and are unaffected.

#### `git` or `curl` is missing

The POSIX installer checks for `git` and `curl` up front and fails with the missing tool named — install them with your system package manager (for example `sudo apt install git curl`) and re-run. It also needs `tar` and `sha256sum` (or `shasum`) to stage its verified uv artifact. On native Windows, PowerShell handles the downloads and the installer stages a verified portable Git automatically when Git is absent.

#### WSL notes

Use **WSL2** — Hermes is not reliable on WSL1. Run the Linux installer inside the distro, not on the Windows side. Enable systemd in `/etc/wsl.conf` if you want the gateway to run as a service. Native Windows installs live under `%LOCALAPPDATA%\hermes`; WSL2 installs live under the Linux user's `~/.hermes`.

#### Gateway service notes

Install the service with `hermes gateway install` (or `hermes gateway setup`), then control it with `hermes gateway start|stop|restart|status`. The backend is a systemd unit on Linux, a LaunchAgent on macOS, and a Scheduled Task on Windows. `hermes doctor` reports installation provenance and common problems. Stop a running gateway (`hermes gateway stop`) before replacing files.

---

## Getting Started

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes config set   # Set individual config values
hermes config get   # Print individual config values
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard (configures everything at once)
hermes claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
hermes update       # Update to the latest version
hermes doctor       # Diagnose any issues
```

📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**

---

## Skip the API-key collection — Nous Portal

Hermes works with whatever provider you want — that's not changing. But if you'd rather not collect five separate API keys for the model, web search, image generation, TTS, and a cloud browser, **[Nous Portal](https://portal.nousresearch.com)** covers all of them under one subscription:

- **300+ models** — pick any of them with `/model <name>`
- **Tool Gateway** — web search, image generation (FAL), text-to-speech (OpenAI), cloud browser (Browser Use), all routed through your sub. No extra accounts.

One command from a fresh install:

```bash
hermes setup --portal
```

That logs you in via OAuth, sets Nous as your provider, and turns on the Tool Gateway. Check what's wired up any time with `hermes portal info`. Full details on the [Tool Gateway docs page](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

You can still bring your own keys per-tool whenever you want — the gateway is per-backend, not all-or-nothing.

---

## CLI vs Messaging Quick Reference

Hermes has two entry points: start the terminal UI with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action                         | CLI                                           | Messaging platforms                                                              |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| Start chatting                 | `hermes`                                      | Run `hermes gateway setup` + `hermes gateway start`, then send the bot a message |
| Start fresh conversation       | `/new` or `/reset`                            | `/new` or `/reset`                                                               |
| Change model                   | `/model [provider:model]`                     | `/model [provider:model]`                                                        |
| Set a personality              | `/personality [name]`                         | `/personality [name]`                                                            |
| Retry or undo the last turn    | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                        |
| Browse skills                  | `/skills` or `/<skill-name>`                  | `/<skill-name>`                                                                  |
| Interrupt current work         | `Ctrl+C` or send a new message                | `/stop` or send a new message                                                    |
| Platform-specific status       | `/platforms`                                  | `/status`, `/sethome`                                                            |

For the full command lists, see the [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Session Continuation After Reboot

The interactive CLI can resume one session automatically across a reboot:

| Action | Command |
| --- | --- |
| Arm the current session | `/continue-after-reboot` (alias `/reboot-continue`) |
| Revoke a pending continuation | `/cancel-continuation` (alias `/cancel-reboot-continue`) |

Arming persists intent in `$HERMES_HOME/session_continuation.json`. The marker fires only on the first bare interactive `hermes` login after the machine's boot token changes (Linux's `boot_id`, or the OS boot time elsewhere), within 72 hours of arming, and only if the recorded session still exists. It resumes through the normal `--resume` path and is consumed once; an explicit `--resume`, a `-q` query, or a `--tools` listing skips the automatic path.

- An explicit exit/quit, `/new`, or `/clear` revokes the marker.
- A crash, disconnect, or closed terminal neither arms nor revokes it.
- Arming again replaces a previous marker — one marker at a time.
- Corrupt or stale state files are quarantined and ignored.

System shutdown/reboot commands (`shutdown`, `reboot`, `halt`, `poweroff`, `init 0/6`, `systemctl poweroff/reboot`, `telinit 0/6`) are no longer hard-blocked: they run through the normal ask-approval path, subject to your configured approval mode. The unconditional hardline block remains for unrecoverable destructive commands.

---

## Memory and Context

Hermes has several distinct layers; only the first is on by default.

### Built-in curated memory (default on)

The agent maintains two bounded, human-readable files under `$HERMES_HOME/memories/`: `MEMORY.md` (facts about the environment) and `USER.md` (who you are). The `memory` tool reads and edits both, and their contents are injected into the system prompt at session start. Limits and toggles live under `memory:` in `config.yaml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `memory_enabled` | `true` | Enable `MEMORY.md` |
| `user_profile_enabled` | `true` | Enable `USER.md` |
| `memory_char_limit` | `2200` | `MEMORY.md` budget (~800 tokens) |
| `user_char_limit` | `1375` | `USER.md` budget (~500 tokens) |
| `nudge_interval` | `10` | Turns between built-in memory review nudges |
| `write_approval` | `false` | Require approval for memory writes (foreground inline; background staged) |

### Enhanced local memory (opt-in)

`memory.enhanced.enabled: true` adds a structured SQLite store on top of curated memory: typed memories with lifecycle (store/update/supersede/delete/link), FTS5 BM25 search, rank-fusion hybrid retrieval with graph expansion and temporal windows, evidence accumulation, and automatic contradiction detection. Data lives under `$HERMES_HOME/memory_engine/`. Off by default — when disabled, behavior is exactly the built-in layer plus any configured provider. Config keys: `enabled` (`false`), `data_dir` (`"memory_engine"`), `auto_recall`, `auto_capture`, `max_recall`, `recall_budget_chars`.

### Memory provider plugins (opt-in)

Set `memory.provider` in `config.yaml` to add an external memory engine — only one at a time; empty means built-in only:

- In-tree plugins under `plugins/memory/`: `rekal`, `holographic`, `retaindb`, `byterover`.
- Catalog-installed providers: Honcho, Hindsight, Supermemory, Mem0, OpenViking.

`memory.provider: rekal` uses the bundled Rekal engine — SQLite with FTS5 hybrid search, rank fusion, graph expansion, temporal recall, evidence accumulation, contradiction detection, and session ingest, with automatic recall and capture. Its native drawer/knowledge-graph/diary operations additionally need the optional `mempalace` package (`pip install mempalace`); without it the zero-dependency engine keeps working and those operations return an install hint.

### Content-aware tool-output compression (default on — not memory)

`compression.content_aware.*` shrinks tool output that is re-sent every turn, cutting tokens; it is context management, not memory. It is enabled by default under `compression.enabled: true`: lossless folds and lossless per-type renders (JSON, CSV, config) are round-trip-verified, and lossy elision (dense lines, JSON/log/search/diff drops) fires only when the original can be persisted and retrieved through the marker. `read_lifecycle` replaces stale or superseded `read_file` results. Relevant keys: `enabled`, `lossless_folds`, `type_compression`, `read_lifecycle` (all `true`), `dense_line_elision` (`false`), `net_cost_gate` (`false`, cache-aware opt-in).

Session continuation (above) is unrelated to memory: it reopens a session, it does not store facts.

---

## Documentation

All documentation lives at **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Section                                                                                             | What's Covered                                             |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)                 | Install → setup → first conversation in 2 minutes          |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                              | Commands, keybindings, personalities, sessions             |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                | Config file, providers, models, all options                |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)                | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security)                          | Command approval, DM pairing, container isolation          |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)            | 40+ tools, toolset system, terminal backends               |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)              | Procedural memory, Skills Hub, creating skills             |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                     | Persistent memory, user profiles, best practices           |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)               | Connect any MCP server for extended capabilities           |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)              | Scheduled tasks with platform delivery                     |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)       | Project context that shapes every conversation             |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)             | Project structure, agent loop, key classes                 |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)             | Development setup, PR process, code style                  |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                  | All commands and flags                                     |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Complete env var reference                                 |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, Hermes can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`hermes setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:

- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `hermes claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) for development setup, code style, and PR process.

Start with the [PM developer workflow](website/docs/reference/package-management.md#developer-workflow)
for activation, daily use, dependency changes, and leaving the environment.
[Development Setup](CONTRIBUTING.md#development-setup) covers the separate test environment and verification commands.

---

## Community

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Linux desktop-control MCP server for Hermes and other MCP hosts, with AT-SPI accessibility trees, Wayland/X11 input, screenshots, and compositor window targeting.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Community WeChat bridge: Run Hermes Agent and OpenClaw on the same WeChat account.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com).
