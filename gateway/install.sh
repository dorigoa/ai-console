sudo cp *fwd.service /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable --now freetoken-fwd.service llama-server-fwd.service mlx-fwd.service splash-fwd.service strata-fwd.service

source tailscale-serve-setup.sh
