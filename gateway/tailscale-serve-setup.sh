# Web Services
tailscale serve --bg --https=5443 http://127.0.0.1:9000 # Wake/poweroff PC Linux
tailscale serve --bg --https=6443 http://127.0.0.1:3000 # Grafana
tailscale serve --bg --https=443  http://127.0.0.1:8501 # ai-console-gui.py

# AI Http API
tailscale serve --bg --https=1443 http://127.0.0.1:18088 # Forward to MacStudio2 llama-server (192.168.1.191:8088)
tailscale serve --bg --https=2443 http://127.0.0.1:8000 # Forward to PC Linux strata server (192.168.1.200:8000)
tailscale serve --bg --https=3443 http://127.0.0.1:8080 # Forward to MacStudio1 mlx_lm/jaccl (192.168.1.101:8080)
tailscale serve --bg --https=4443 http://127.0.0.1:10000 # Forward to MacStudio2 Splash (192.168.1.191:10000)
tailscale serve --bg --https=7443 http://127.0.0.1:1919 # Forward to PC Linux Freetoken API (192.168.1.200:1919)

