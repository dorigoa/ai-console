#!/bin/sh
# Puts the keys mounted by run.sh into ~/.ssh, then runs the container's
# command (ai-console by default).
#
# run.sh mounts read-only in /run/host-ssh/github the keys the host uses for
# GitHub, and in /run/host-ssh/nodes the ones for ai-console's nodes (none if
# it forwards the ssh-agent). They are copied here so that in ~/.ssh they
# belong to the user and have the permissions ssh requires.
set -eu

src=/run/host-ssh
dst="$HOME/.ssh"

mkdir -p "$dst"
chmod 700 "$dst"

# Copies the keys in $src/$1 into ~/.ssh; $keys gets the names of the private ones.
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

# The config is written here: the host's has macOS-only options (UseKeychain).
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
    echo "entrypoint: no keys in $src/github, git cannot authenticate to GitHub (start with run.sh)" >&2
fi

exec "$@"
