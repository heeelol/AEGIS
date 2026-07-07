# AEGIS on the MIC-733-AO — Operator & Developer Guide

The **MIC-733-AO** is the edge device that runs the AEGIS kitting pipeline
(camera + hand tracking + load cells + web dashboard). Internally it's an
**NVIDIA Jetson AGX Orin** (ARM64, JetPack 6). This guide is everything you need
to connect from your laptop and drive it.

---

## 1. Connect to the MIC

The MIC talks to your laptop over the **ethernet cable** (there's no Wi-Fi on it).

1. **Plug** the ethernet cable between your laptop and the MIC's LAN port.
2. **Share your laptop's connection** so the MIC gets an address:
   - **Ubuntu / GNOME:** Settings → Network → Wired → the gear icon → **IPv4 →
     "Shared to other computers"** → Apply, then toggle the wired connection off/on.
   - (This makes your laptop `10.42.0.1` and hands the MIC an address.)
3. **Find the MIC** (it's normally `10.42.0.16`):
   ```bash
   ping -c1 10.42.0.16            # usually this
   # if that fails, list what's on the link:
   ip neigh | grep 10.42.0        # the MIC is the non-.1 address
   ```
4. **Log in** (password is the same as the username: `mic-733ao`):
   ```bash
   ssh mic-733ao@10.42.0.16
   ```

> If the IP isn't `.16`, use whatever `ip neigh` shows. After the one-time setup
> below you can just type `mic`.

---

## 2. One-time laptop setup (recommended)

This installs an `ssh mic` shortcut, a login key (no more passwords), and the
`mic-*` helper commands. Run it **once** on your laptop:

```bash
scp mic-733ao@10.42.0.16:~/mic-setup.sh /tmp/mic-setup.sh && bash /tmp/mic-setup.sh
```

Then open a new terminal (or `source ~/.bashrc`). Done — you now have all the
commands below.

---

## 3. Commands (run from your laptop)

| Command | What it does |
|---|---|
| `mic` | Open a shell on the MIC. `mic <cmd>` runs one command remotely. |
| `mic-run` | **Start the pipeline.** The camera overlay + dashboard appear on the MIC's screen (HMI); the dashboard opens **fullscreen**. Logs stream to your terminal. **Ctrl-C** to stop. |
| `mic-pull` | Pull the latest AEGIS code onto the MIC (safely keeps the device's local settings). |
| `mic-config` | Edit a config file (menu: **1** settings, **2** inventory). Checks the YAML when you save. `mic-config settings` / `mic-config inventory` skip the menu. |
| `mic-fullscreen` | Toggle the dashboard in/out of **fullscreen**. |
| `mic-refresh` | **Hard-refresh** the dashboard (Ctrl+Shift+R, bypasses cache). |

**Typical session:**
```bash
mic-config inventory     # set item weights / bin assignments
mic-run                  # start it; watch the HMI + logs; Ctrl-C when done
```

---

## 4. Driving a running pipeline

While `mic-run` is going, type these **in the same terminal**:

| Key | Action |
|---|---|
| `1` | Calibrate the workstation grid (from the current camera view) |
| `2` | Initialise the kit |
| `m` | Toggle the "model's-eye" foreground tuning view |
| `q` | Quit (same as Ctrl-C) |

The same keys also work at the **MIC's own keyboard** in the overlay window.
At the HMI, Firefox's native **F11** (fullscreen) and **Ctrl+Shift+R** (hard
reload) work too — or use `mic-fullscreen` / `mic-refresh` from your laptop.

---

## 5. The config files

Both live on the MIC at `~/AEGIS/aegis-v2/integration/config/` — edit them with
`mic-config`:

- **`settings.yaml`** — camera, bin detector, hand tracker, load cells, dashboard,
  and the **work order** (target pick counts per bin).
- **`inventory.yaml`** — the item in each bin and its **grams per unit** (used to
  turn load-cell weight into a pick count).

Changes apply on the **next `mic-run`** (restart the pipeline).

> The MIC keeps two device-specific overrides in `settings.yaml`
> (`device: cuda:0`, load-cell `port: /dev/ttyUSB0`). `mic-pull` preserves them
> automatically — don't be surprised to see them differ from the repo.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ssh: connect … timed out` / no route | Ethernet unplugged, or laptop sharing is off. Redo step 1–2. Confirm your laptop has `10.42.0.1` (`ip -br addr`). |
| MIC not at `10.42.0.16` | `ip neigh \| grep 10.42.0` to find it; `ssh mic-733ao@<ip>`. |
| `Cannot open camera` | A previous run is still holding it. `mic 'pkill -f integration.src.pipeline'`, then `mic-run`. |
| Dashboard didn't open / blank | `mic-refresh`. If still bad, restart with `mic-run`. |
| `mic-fullscreen`/`mic-refresh` say "window not found" | The pipeline/dashboard isn't running — start it with `mic-run` first. |
| Dashboard shows no weights | Check the ESP32 is plugged in (`mic 'ls /dev/ttyUSB*'`) and `loadcells.enabled: true` in `settings.yaml`. |

---

## 7. Under the hood (for developers)

- **Repo:** `~/AEGIS` on the MIC (branch `main`, GitHub `heeelol/AEGIS`).
- **Run script:** `~/AEGIS/aegis-v2/run_pipeline.sh` (what `mic-run` calls) — sets
  the HMI display, opens + fullscreens the dashboard, launches the pipeline.
- **Python env:** `~/AEGIS/aegis-v2/.venv` (CUDA-enabled PyTorch 2.8 for JetPack 6).
- **Dashboard:** FastAPI on `http://localhost:8080` (on the MIC).
- **Helper scripts on the MIC:** `~/mic-update.sh` (pull), `~/mic-browser.sh`
  (fullscreen/refresh), `~/mic-config.sh` (edit config), `~/mic-setup.sh` (laptop
  bootstrap).
- **Hardware:** Logitech C270 camera → `/dev/video0`; ESP32 load cells → `/dev/ttyUSB0`.
