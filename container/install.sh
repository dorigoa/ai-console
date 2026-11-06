#!/usr/bin/env bash
# Linux con podman: costruisce l'immagine, installa l'unità Quadlet e (ri)avvia
# il servizio, che da lì in poi parte da solo a ogni boot. Va rilanciato anche
# per aggiornare, dopo un `git pull` o un nuovo commit di ai-console.
#
# Uso: ./install.sh [unità]     (default: pi-dev.container -> servizio pi-dev)
set -euo pipefail
cd "$(dirname "$0")"

unit=${1:-pi-dev.container}
service=$(basename "$unit" .container).service
dir="${XDG_CONFIG_HOME:-$HOME/.config}/containers/systemd"

# Indirizzi pubblicati dall'unità (ip:porta), es. 127.0.0.1:8501.
mapfile -t addrs < <(awk -F'[=:]' '$1 == "PublishPort" { print $2 ":" $3 }' "$unit")
port=${addrs[0]##*:}

# La porta deve essere libera, a meno che non sia già del servizio.
if ! systemctl --user is-active --quiet "$service" && [[ -n $(ss -ltnH "sport = :$port") ]]; then
    echo "install.sh: la porta $port è già occupata (ai-console-gui.service?): liberala prima" >&2
    exit 1
fi

./build.sh

mkdir -p "$dir"
install -m 644 "$unit" "$dir/${unit##*/}"
systemctl --user daemon-reload
systemctl --user restart "$service"   # avvia, o riavvia con l'immagine appena costruita

# Senza linger il servizio parte al primo login invece che al boot.
if [[ $(loginctl show-user "$(id -un)" -p Linger --value) != yes ]]; then
    echo "install.sh: per l'avvio al boot senza login esegui: loginctl enable-linger" >&2
fi

for _ in {1..20}; do
    if curl -fs -o /dev/null "http://${addrs[0]}/"; then
        echo "ai-console attiva:"
        printf '  http://%s\n' "${addrs[@]}"
        exit 0
    fi
    sleep 1
done
echo "install.sh: ai-console non risponde su ${addrs[0]}, vedi: journalctl --user-unit ${service%.service}" >&2
exit 1
