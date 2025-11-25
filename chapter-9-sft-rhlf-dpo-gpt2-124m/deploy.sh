#!/bin/bash
# GPT-2 Web App Deployment Script for Oracle Server
# Run with: sudo bash deploy.sh

set -e  # Exit on error

echo "=================================================="
echo "🚀 GPT-2 Web App Deployment Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root or with sudo${NC}"
    exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
echo -e "${GREEN}✓${NC} Running as: $ACTUAL_USER"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo -e "${GREEN}✓${NC} Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Install Caddy (if not installed)
echo "📦 Step 1: Checking Caddy installation..."
if ! command -v caddy &> /dev/null; then
    echo "Installing Caddy..."
    apt update
    apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
    apt update
    apt install -y caddy
    echo -e "${GREEN}✓${NC} Caddy installed successfully"
else
    echo -e "${GREEN}✓${NC} Caddy already installed: $(caddy version | head -1)"
fi
echo ""

# Step 2: Setup systemd service
echo "🔧 Step 2: Setting up systemd service..."
if [ -f "$SCRIPT_DIR/gpt2-webapp.service" ]; then
    cp "$SCRIPT_DIR/gpt2-webapp.service" /etc/systemd/system/
    chmod 644 /etc/systemd/system/gpt2-webapp.service
    echo -e "${GREEN}✓${NC} Service file copied"
else
    echo -e "${RED}❌ gpt2-webapp.service not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Reload systemd
systemctl daemon-reload
echo -e "${GREEN}✓${NC} Systemd reloaded"
echo ""

# Step 3: Configure Caddy
echo "🌐 Step 3: Configuring Caddy..."
mkdir -p /etc/caddy
if [ -f "$SCRIPT_DIR/Caddyfile" ]; then
    cp "$SCRIPT_DIR/Caddyfile" /etc/caddy/Caddyfile
    chmod 644 /etc/caddy/Caddyfile
    echo -e "${GREEN}✓${NC} Caddyfile copied"
else
    echo -e "${RED}❌ Caddyfile not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Create log directory
mkdir -p /var/log/caddy
chown -R caddy:caddy /var/log/caddy
echo -e "${GREEN}✓${NC} Log directory created"

# Validate Caddy config
if caddy validate --config /etc/caddy/Caddyfile; then
    echo -e "${GREEN}✓${NC} Caddy configuration valid"
else
    echo -e "${RED}❌ Caddy configuration invalid${NC}"
    exit 1
fi
echo ""

# Step 4: Configure firewall
echo "🔥 Step 4: Configuring firewall..."
if command -v ufw &> /dev/null; then
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw --force enable
    echo -e "${GREEN}✓${NC} UFW rules added"
fi

# Also add iptables rules (for Oracle Cloud)
iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || iptables -I INPUT 1 -p tcp --dport 443 -j ACCEPT

# Save iptables rules
if command -v netfilter-persistent &> /dev/null; then
    netfilter-persistent save
    echo -e "${GREEN}✓${NC} iptables rules saved"
else
    echo -e "${YELLOW}⚠${NC} netfilter-persistent not found, iptables rules may not persist"
fi
echo ""

# Step 5: Start services
echo "🚀 Step 5: Starting services..."

# Stop services if already running
systemctl stop gpt2-webapp.service 2>/dev/null || true

# Enable and start webapp service
systemctl enable gpt2-webapp.service
systemctl start gpt2-webapp.service

# Wait for service to start
echo "Waiting for webapp to start..."
sleep 5

# Check service status
if systemctl is-active --quiet gpt2-webapp.service; then
    echo -e "${GREEN}✓${NC} GPT-2 webapp service is running"
else
    echo -e "${RED}❌ GPT-2 webapp service failed to start${NC}"
    echo "Check logs with: sudo journalctl -u gpt2-webapp.service -n 50"
    exit 1
fi

# Enable and reload Caddy
systemctl enable caddy
systemctl reload caddy

# Check Caddy status
if systemctl is-active --quiet caddy; then
    echo -e "${GREEN}✓${NC} Caddy service is running"
else
    echo -e "${RED}❌ Caddy service failed to start${NC}"
    echo "Check logs with: sudo journalctl -u caddy -n 50"
    exit 1
fi
echo ""

# Step 6: Verify deployment
echo "✅ Step 6: Verifying deployment..."

# Check if app is listening on port 7860
if netstat -tuln | grep -q ":7860 "; then
    echo -e "${GREEN}✓${NC} App is listening on port 7860"
else
    echo -e "${YELLOW}⚠${NC} App may not be listening on port 7860 yet"
fi

# Check if ports 80 and 443 are open
if netstat -tuln | grep -q ":80 "; then
    echo -e "${GREEN}✓${NC} Port 80 is open"
fi

if netstat -tuln | grep -q ":443 "; then
    echo -e "${GREEN}✓${NC} Port 443 is open"
fi

echo ""
echo "=================================================="
echo "🎉 Deployment Complete!"
echo "=================================================="
echo ""
echo "Your app should be accessible at:"
echo "  🌐 https://gpt2.devshubh.me"
echo ""
echo "📝 Important notes:"
echo "  • First HTTPS access may take ~30 seconds (SSL cert provisioning)"
echo "  • Make sure Oracle Cloud security list allows ports 80 and 443"
echo "  • DNS should point gpt2.devshubh.me to 80.225.224.160"
echo ""
echo "📊 Useful commands:"
echo "  • View logs: sudo journalctl -u gpt2-webapp.service -f"
echo "  • Restart app: sudo systemctl restart gpt2-webapp.service"
echo "  • Reload Caddy: sudo systemctl reload caddy"
echo ""
echo "📖 Full guide: See DEPLOYMENT_GUIDE.md"
echo ""
echo "=================================================="




