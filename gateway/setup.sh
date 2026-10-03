tailscale serve --bg --https=10443 http://127.0.0.1:9000
tailscale serve --bg --https=11443 http://127.0.0.1:3000 # local grafana
tailscale serve --bg --https=443 http://127.0.0.1:18088 # llama-server on 192.168.1.191
tailscale serve --bg --https=8443 http://127.0.0.1:8000
tailscale serve --bg --https=9443 http://127.0.0.1:8501
tailscale serve --bg --https=12443 http://127.0.0.1:8001
tailscale serve --bg --https=13443 http://127.0.0.1:8080
