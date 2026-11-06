#!/bin/sh
# Mette in ~/.ssh le chiavi montate da run.sh, poi esegue il comando del
# container (di default ai-console).
#
# run.sh monta in sola lettura in /run/host-ssh/github le chiavi che l'host usa
# per GitHub e in /run/host-ssh/nodes quelle per i nodi di ai-console (nessuna
# se inoltra l'ssh-agent). Qui vengono copiate, così in ~/.ssh sono dell'utente
# e hanno i permessi che ssh richiede.
set -eu

src=/run/host-ssh
dst="$HOME/.ssh"

mkdir -p "$dst"
chmod 700 "$dst"

# Copia in ~/.ssh le chiavi di $src/$1; in $keys i nomi di quelle private.
copy_keys() {
    keys=""
    for f in "$src/$1"/*; do
        [ -f "$f" ] || continue
        name=${f##*/}
        case "$name" in
            *.pub) install -m 644 "$f" "$dst/$name" ;;
            *)     install -m 600 "$f" "$dst/$name"; keys="$keys $name" ;;
        esac
    done
}
copy_keys github; github_keys=$keys
copy_keys nodes;  node_keys=$keys

# Config scritta qui: quella dell'host ha opzioni solo macOS (UseKeychain).
{
    if [ -n "$github_keys" ]; then
        echo "Host github.com"
        echo "    User git"
        for k in $github_keys; do echo "    IdentityFile ~/.ssh/$k"; done
        echo "    IdentitiesOnly yes"
    fi
    if [ -n "$node_keys" ]; then
        echo "Host * !github.com"
        for k in $node_keys; do echo "    IdentityFile ~/.ssh/$k"; done
    fi
} > "$dst/config"
chmod 600 "$dst/config"

if [ -z "$github_keys" ]; then
    echo "entrypoint: nessuna chiave in $src/github, git non potrà autenticarsi su GitHub (avvia con run.sh)" >&2
fi

exec "$@"
