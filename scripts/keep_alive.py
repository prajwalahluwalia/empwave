#!/usr/bin/env python3
"""Keep Render free instance alive by pinging health endpoint every 10 minutes."""

import os
import time
import logging
import requests
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
APP_URL = os.getenv("APP_URL", "http://localhost:5007")
PING_INTERVAL = 600  # 10 minutes in seconds
HEALTH_ENDPOINT = f"{APP_URL}/health"

def ping_server():
    """Ping the health endpoint to keep server alive."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ Keep-alive ping successful ({response.status_code})")
            return True
        else:
            logger.warning(f"⚠️  Ping returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Keep-alive ping failed: {e}")
        return False

def main():
    """Run keep-alive loop."""
    logger.info(f"Starting keep-alive service for {APP_URL}")
    logger.info(f"Pinging every {PING_INTERVAL} seconds ({PING_INTERVAL // 60} minutes)")
    
    ping_count = 0
    while True:
        try:
            time.sleep(PING_INTERVAL)
            ping_count += 1
            logger.info(f"[Ping #{ping_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ping_server()
        except KeyboardInterrupt:
            logger.info("Keep-alive service stopped")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    main()
