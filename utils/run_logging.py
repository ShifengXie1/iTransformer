"""Mirror Python console output to one UTF-8 log file per run."""

import atexit
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import threading


class _Tee:
    def __init__(self, console, log_file, lock):
        self.console = console
        self.log_file = log_file
        self.lock = lock

    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.log_file.write(text)
            self.log_file.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.console.flush()
            if not self.log_file.closed:
                self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def start_run_logging(log_dir, args):
    """Capture stdout/stderr until exit, including uncaught Python tracebacks.

    One invocation of run.py produces one log (including all --itr repeats).
    Existing console output remains visible, and each write is flushed to disk.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model = re.sub(r'[^A-Za-z0-9_.-]', '_', args.model)[:64]
    data = re.sub(r'[^A-Za-z0-9_.-]', '_', args.data)[:64]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path = log_dir / (
        f'{timestamp}_{model}_{data}_sl{args.seq_len}_pl{args.pred_len}'
        f'_{os.getpid()}.log'
    )
    log_file = log_path.open('x', encoding='utf-8', buffering=1)
    stdout, stderr = sys.stdout, sys.stderr
    lock = threading.RLock()
    tee_out = _Tee(stdout, log_file, lock)
    tee_err = _Tee(stderr, log_file, lock)
    sys.stdout, sys.stderr = tee_out, tee_err

    def close_log():
        try:
            tee_out.flush()
            tee_err.flush()
        finally:
            if sys.stdout is tee_out:
                sys.stdout = stdout
            if sys.stderr is tee_err:
                sys.stderr = stderr
            log_file.close()

    atexit.register(close_log)
    print('Log file:', log_path.resolve())
    print('Started at:', datetime.now().astimezone().isoformat(timespec='seconds'))
    print('Working directory:', Path.cwd())
    print('Command arguments:', sys.argv)
    return str(log_path.resolve())
