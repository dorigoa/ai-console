tailscale serve --bg --https=10443 http://127.0.0.1:9000 # Wake/poweroff PC Linux
tailscale serve --bg --https=11443 http://127.0.0.1:3000 # Grafana
tailscale serve --bg --https=443 http://127.0.0.1:18088 # Forward to MacStudio2 llama-server (192.168.1.191:8088)
#tailscale serve --bg --https=8443 http://127.0.0.1:8000 
tailscale serve --bg --https=9443 http://127.0.0.1:8501 # llama-console-gui.py
tailscale serve --bg --https=1443 http://127.0.0.1:8000 # Forward to PC Linux strata server (192.168.1.200:8000)
tailscale serve --bg --https=2443 http://127.0.0.1:8080 # Forward to MacStudio2 mlx_lm/jaccl (192.168.1.101:8080)
