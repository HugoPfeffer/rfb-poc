import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import json
from functools import wraps
import sys
import time
from datetime import datetime

class ContextualLogger:
    """Logger that adds contextual information to log messages."""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
        
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / 'logs'
        
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(context)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

    def add_context(self, **kwargs) -> None:
        """Add context to the logger."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context."""
        self.context.clear()

    def _format_context(self) -> str:
        """Format context for logging."""
        return json.dumps(self.context) if self.context else '{}'

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Internal logging method."""
        extra = kwargs.pop('extra', {})
        extra['context'] = self._format_context()
        self.logger.log(level, msg, *args, extra=extra, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message with context."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message with context."""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message with context."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

def log_execution_time(logger: ContextualLogger):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.add_context(
                function=func.__name__,
                start_time=datetime.now().isoformat()
            )
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
            finally:
                logger.clear_context()
        
        return wrapper
    return decorator

# Create default logger instance
app_logger = ContextualLogger('rfb_poc') 