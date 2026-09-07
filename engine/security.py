"""
Process isolation and security utilities (single-instance mutex, file permissions).
"""

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes

# Default project mutex identifier
MUTEX_NAME = "Local\\Woeyyy_Audio_Suite_SingleInstance_Mutex"


class SingleInstanceLock:
    """
    Guarantees strictly ONE running instance of Woeyyy Audio Suite at a time.
    Uses Windows Kernel Named Mutex via kernel32.CreateMutexW.
    Unlike lock files, Windows kernel automatically releases Named Mutexes
    even on abnormal crashes, power cuts, or Task Manager termination.
    """

    def __init__(self, mutex_name: str = MUTEX_NAME):
        self.mutex_name = mutex_name
        self._handle = None
        self.is_locked = False

    def acquire(self) -> bool:
        """
        Attempt to acquire the single-instance lock.
        Returns True if this process is the first/only instance.
        Returns False if another instance is already running.
        """
        if sys.platform != "win32":
            # Fallback for non-windows environments
            return True

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            CreateMutexW = kernel32.CreateMutexW
            CreateMutexW.argtypes = [wintypes.LPCVOID, wintypes.BOOL, wintypes.LPCWSTR]
            CreateMutexW.restype = wintypes.HANDLE

            # ERROR_ALREADY_EXISTS = 183
            self._handle = CreateMutexW(None, False, self.mutex_name)
            last_err = ctypes.get_last_error()

            if self._handle and last_err == 183:
                # Mutex exists: another instance is running
                self.is_locked = False
                return False

            if self._handle:
                self.is_locked = True
                return True

            return False
        except Exception as e:
            print(f"[Security] Warning: Could not create single-instance mutex: {e}")
            return True

    def release(self):
        """Release the mutex handle upon clean exit."""
        if self._handle and sys.platform == "win32":
            try:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None
            self.is_locked = False

    @staticmethod
    def focus_existing_window(title_keyword: str = "Woeyyy") -> bool:
        """
        Find an existing window containing title_keyword and bring it to the foreground.
        """
        if sys.platform != "win32":
            return False

        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            found_hwnd = None

            def _enum_windows_cb(hwnd, _):
                nonlocal found_hwnd
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        if title_keyword.lower() in buf.value.lower():
                            found_hwnd = hwnd
                            return False
                return True

            user32.EnumWindows(WNDENUMPROC(_enum_windows_cb), 0)

            if found_hwnd:
                # SW_RESTORE = 9
                user32.ShowWindow(found_hwnd, 9)
                user32.SetForegroundWindow(found_hwnd)
                return True
        except Exception:
            pass
        return False


def secure_file_permissions(filepath: str):
    """
    Harden file permissions on Windows so only the current user profile
    can read/write the file, stripping inherited permissions from other accounts.
    """
    if not os.path.exists(filepath):
        return

    if sys.platform == "win32":
        username = os.environ.get("USERNAME")
        if username:
            try:
                cmd = ["icacls", os.path.abspath(filepath), "/inheritance:r", "/grant:r", f"{username}:(R,W)"]
                subprocess.run(cmd, capture_output=True, text=True, check=False)
            except Exception:
                pass
    else:
        try:
            os.chmod(filepath, 0o600)
        except Exception:
            pass

