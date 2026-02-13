#!/usr/bin/env python
"""
Trading Bot Runner Script
This script is a convenient way to start the trading bot.
"""
import sys
import logging
from bot.main import initialize_bot, trading_loop
from bot.notifications import get_notifier

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("trading_bot_runner")
    
    logger.info("=== Starting trading bot runner ===")
    
    try:
        if initialize_bot():
            trading_loop()
        else:
            logger.critical("Failed to initialize bot. Exiting.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        try:
            _notifier = get_notifier()
            _notifier.send_lifecycle("stopped")
            _notifier.shutdown()
        except:
            pass
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        try:
            _notifier = get_notifier()
            _notifier.send_lifecycle("crashed")
            _notifier.shutdown()
        except:
            pass
        sys.exit(1)