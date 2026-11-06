#!/bin/sh
# Puts the keys for ai-console's nodes into ~/.ssh, then runs the container's
# command (ai-console by default).
#
# run.sh and ai-console.container mount them read-only in /run/host-ssh/nodes
# (none if run.sh forwards the host's ssh-agent instead). They are copied here
# so that in ~/.ssh they belong to the user and have the permissions ssh requires.
set -eu

src=/run/host-ssh/nodes
dst="$HOME/.ssh"

mkdir -p "$dst"
chmod 700 "$dst"

keys=""
for f in "$src"/*; do
    [ -f "$f" ] || continue
    name=${f##*/}
    case "$name" in
        *.pub) install -m 644 "$f" "$dst/$name" ;;
        *)     install -m 600 "$f" "$dst/$name"; keys="$keys $name" ;;
    esac
done

# The config is written here: the host's has macOS-only options (UseKeychain).
{
    if [ -n "$keys" ]; then
        echo "Host *"
        for k in $keys; do echo "    IdentityFile ~/.ssh/$k"; done
    fi
} > "$dst/config"
chmod 600 "$dst/config"

if [ -z "$keys" ] && [ -z "${SSH_AUTH_SOCK:-}" ]; then
    echo "entrypoint: no keys in $src and no ssh-agent: ai-console cannot reach the nodes (start with run.sh)" >&2
fi

exec "$@"
