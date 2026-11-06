# container-pi

Container per [ai-console](https://github.com/dorigoa/ai-console), la console NiceGUI dei servizi di inferenza LLM sui nodi di casa. L'immagine ha lo stesso utente dell'host (nome, UID e GID) e un clone di ai-console; le chiavi SSH dell'host non entrano mai nell'immagine.

- **raspy** (Linux, podman rootless): servizio systemd che parte a ogni boot.
- **macOS** (CLI [`container`](https://github.com/apple/container) di Apple): avvio manuale, per sviluppo e prove.

## raspy

Serve: podman; in `~/.ssh` la chiave per GitHub (quella indicata da `ssh -G github.com`) e `id_ecdsa` per i nodi; il linger attivo (`loginctl enable-linger`); la porta 8501 libera.

```bash
git clone git@github.com:dorigoa/container-pi.git ~/container-pi
cd ~/container-pi && ./install.sh
```

`install.sh` costruisce l'immagine, installa l'unità `pi-dev.container` in `~/.config/containers/systemd/` e avvia il servizio `pi-dev`. ai-console risponde su http://127.0.0.1:8501 (per Tailscale serve) e su http://192.168.1.40:8501 (LAN); gli indirizzi sono in `pi-dev.container`.

Per aggiornare, dopo un commit in ai-console o in questo repo:

```bash
cd ~/container-pi && git pull && ./install.sh
```

| Cosa | Comando |
|---|---|
| Stato | `systemctl --user status pi-dev` |
| Log | `journalctl --user-unit pi-dev -f` (oppure `podman logs -f pi-dev`) |
| Shell nel container | `podman exec -it pi-dev bash` |
| Fermare / riavviare | `systemctl --user stop pi-dev` / `systemctl --user restart pi-dev` |

## macOS

```bash
./build.sh
./run.sh            # ai-console su http://127.0.0.1:8501
./run.sh bash       # oppure una shell al posto di ai-console
```

Per i nodi `run.sh` inoltra l'ssh-agent del Mac (`container run --ssh`).

## File

| File | Ruolo |
|---|---|
| `ContainerFile` | immagine `pi-agent`: Python 3.13 su Alpine, utente dell'host, clone di ai-console con le sue dipendenze |
| `build.sh` | build con `container` o podman; la chiave GitHub è un build secret usato solo per il clone |
| `entrypoint.sh` | all'avvio copia in `~/.ssh` le chiavi montate in sola lettura e scrive la config ssh |
| `run.sh` | avvio manuale; legge `run.env` se esiste (`LAN_IP`, `HOST_PORT`, `NODE_KEYS`, `RUNTIME`) |
| `pi-dev.container` | unità Quadlet per raspy |
| `install.sh` | build e installazione o aggiornamento del servizio (Linux) |
