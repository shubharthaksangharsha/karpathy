# 🚀 GPT-2 Web App Deployment Guide

Complete guide to deploy your GPT-2 Q&A app on Oracle Server with systemd and Caddy.

---

## 📋 Prerequisites

- Oracle Server running Ubuntu/Debian
- Domain DNS configured (gpt2.devshubh.me → 80.225.224.160)
- Python 3.12 with dependencies installed
- Root/sudo access

---

## 🔧 Step 1: Install Caddy (if not already installed)

```bash
# Install Caddy
sudo apt update
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

Verify installation:
```bash
caddy version
```

---

## 🔧 Step 2: Copy Files to Server

On your **Oracle Server**, navigate to your project directory:

```bash
cd /home/shubharthak/Desktop/karpathy/chapter-9-sft-rhlf-dpo-gpt2-124m
```

Make sure these files exist:
- `webapp_standalone.py` (updated with footer)
- `qa-sft_best.pt` (your trained model)
- `checkpoints/model_09535.pt` (base model)

---

## 🔧 Step 3: Setup Systemd Service

### 1. Copy the service file:

```bash
sudo cp gpt2-webapp.service /etc/systemd/system/
```

### 2. Set correct permissions:

```bash
sudo chmod 644 /etc/systemd/system/gpt2-webapp.service
```

### 3. Reload systemd:

```bash
sudo systemctl daemon-reload
```

### 4. Enable the service (start on boot):

```bash
sudo systemctl enable gpt2-webapp.service
```

### 5. Start the service:

```bash
sudo systemctl start gpt2-webapp.service
```

### 6. Check status:

```bash
sudo systemctl status gpt2-webapp.service
```

You should see "active (running)" in green!

---

## 🔧 Step 4: Configure Caddy

### 1. Create Caddy config directory:

```bash
sudo mkdir -p /etc/caddy
```

### 2. Copy Caddyfile:

```bash
sudo cp Caddyfile /etc/caddy/Caddyfile
```

### 3. Create log directory:

```bash
sudo mkdir -p /var/log/caddy
sudo chown caddy:caddy /var/log/caddy
```

### 4. Test Caddy configuration:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
```

### 5. Reload Caddy:

```bash
sudo systemctl reload caddy
```

If Caddy isn't running yet:

```bash
sudo systemctl enable caddy
sudo systemctl start caddy
```

### 6. Check Caddy status:

```bash
sudo systemctl status caddy
```

---

## 🔧 Step 5: Firewall Configuration

Make sure ports 80 and 443 are open:

```bash
# For UFW (Ubuntu Firewall)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload

# Or for Oracle Cloud (iptables)
sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 1 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

**Important for Oracle Cloud**: Also configure the security list in Oracle Cloud Console:
1. Go to: Virtual Cloud Networks → Your VCN → Security Lists
2. Add Ingress Rules:
   - Source CIDR: `0.0.0.0/0`
   - Destination Port: `80`
   - Source CIDR: `0.0.0.0/0`
   - Destination Port: `443`

---

## 🔧 Step 6: Verify DNS

Check that DNS is resolving correctly:

```bash
dig gpt2.devshubh.me +short
# Should return: 80.225.224.160

nslookup gpt2.devshubh.me
```

---

## 🎉 Step 7: Access Your App!

Open in your browser:

```
https://gpt2.devshubh.me
```

Caddy will automatically obtain an SSL certificate from Let's Encrypt on first access!

---

## 🛠️ Useful Commands

### Service Management

```bash
# Check service status
sudo systemctl status gpt2-webapp.service

# View logs
sudo journalctl -u gpt2-webapp.service -f

# Restart service
sudo systemctl restart gpt2-webapp.service

# Stop service
sudo systemctl stop gpt2-webapp.service

# Disable service (prevent auto-start)
sudo systemctl disable gpt2-webapp.service
```

### Caddy Management

```bash
# Check Caddy status
sudo systemctl status caddy

# Reload Caddy config (no downtime)
sudo systemctl reload caddy

# Restart Caddy
sudo systemctl restart caddy

# View Caddy logs
sudo journalctl -u caddy -f

# Or view access logs
sudo tail -f /var/log/caddy/gpt2-webapp-access.log
```

### Debugging

```bash
# Check if app is running on port 7860
curl http://localhost:7860

# Check if Caddy is proxying correctly
curl -I https://gpt2.devshubh.me

# View real-time logs
sudo journalctl -u gpt2-webapp.service -u caddy -f

# Check port usage
sudo netstat -tulpn | grep 7860
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443
```

---

## 🔄 Updating the App

When you update the code:

```bash
# 1. Stop the service
sudo systemctl stop gpt2-webapp.service

# 2. Pull/update your code
cd /home/shubharthak/Desktop/karpathy/chapter-9-sft-rhlf-dpo-gpt2-124m
# (make your changes)

# 3. Restart the service
sudo systemctl start gpt2-webapp.service

# 4. Check logs
sudo journalctl -u gpt2-webapp.service -f
```

Or simply restart:

```bash
sudo systemctl restart gpt2-webapp.service
```

---

## 🐛 Troubleshooting

### App won't start?

1. Check logs:
   ```bash
   sudo journalctl -u gpt2-webapp.service -n 50 --no-pager
   ```

2. Check file permissions:
   ```bash
   ls -la /home/shubharthak/Desktop/karpathy/chapter-9-sft-rhlf-dpo-gpt2-124m/
   ```

3. Test manually:
   ```bash
   cd /home/shubharthak/Desktop/karpathy/chapter-9-sft-rhlf-dpo-gpt2-124m
   python webapp_standalone.py
   ```

### Can't access via HTTPS?

1. Check DNS:
   ```bash
   dig gpt2.devshubh.me +short
   ```

2. Check Caddy:
   ```bash
   sudo systemctl status caddy
   sudo journalctl -u caddy -n 50
   ```

3. Check firewall:
   ```bash
   sudo ufw status
   sudo iptables -L -n | grep -E '80|443'
   ```

4. Wait for SSL (first access takes ~30 seconds for Let's Encrypt)

### Port 7860 already in use?

```bash
# Find and kill the process
sudo lsof -ti:7860 | xargs kill -9

# Then restart service
sudo systemctl restart gpt2-webapp.service
```

---

## 📊 Monitoring

### Check resource usage:

```bash
# CPU and Memory
top -p $(pgrep -f webapp_standalone)

# Or use htop
htop -p $(pgrep -f webapp_standalone)
```

### Set up log rotation (optional):

```bash
sudo nano /etc/logrotate.d/gpt2-webapp
```

Add:
```
/var/log/caddy/gpt2-webapp-access.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    missingok
    create 0640 caddy caddy
}
```

---

## 🔒 Security Best Practices

1. **Keep system updated:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Enable automatic security updates:**
   ```bash
   sudo apt install unattended-upgrades
   sudo dpkg-reconfigure --priority=low unattended-upgrades
   ```

3. **Setup fail2ban (optional):**
   ```bash
   sudo apt install fail2ban
   sudo systemctl enable fail2ban
   sudo systemctl start fail2ban
   ```

4. **Monitor logs regularly:**
   ```bash
   sudo journalctl -u gpt2-webapp.service --since today
   ```

---

## 📈 Performance Tuning

If you expect high traffic, consider:

1. **Increase worker processes** in webapp_standalone.py
2. **Add rate limiting** in Caddy
3. **Setup caching** for static assets
4. **Use a CDN** for global distribution

---

## ✅ Success Checklist

- [ ] Caddy installed and running
- [ ] gpt2-webapp.service created and enabled
- [ ] Service is active (running)
- [ ] Port 7860 is listening
- [ ] Firewall allows ports 80 and 443
- [ ] DNS resolves to correct IP
- [ ] HTTPS works with valid certificate
- [ ] App accessible at https://gpt2.devshubh.me
- [ ] Footer shows your links correctly

---

## 🎯 Final URLs

- **Production**: https://gpt2.devshubh.me
- **Your Portfolio**: https://devshubh.me
- **LinkedIn**: https://linkedin.com/in/shubharthaksangharsha/
- **GitHub Repo**: https://github.com/shubharthaksangharsha/karpathy

---

## 💡 Additional Resources

- [Caddy Documentation](https://caddyserver.com/docs/)
- [Systemd Service Documentation](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [Gradio Documentation](https://gradio.app/docs/)
- [Oracle Cloud Firewall Guide](https://docs.oracle.com/en-us/iaas/Content/Network/Concepts/securitylists.htm)

---

**Good luck with your deployment! 🚀**

If you encounter any issues, check the logs first:
```bash
sudo journalctl -u gpt2-webapp.service -u caddy -f
```



