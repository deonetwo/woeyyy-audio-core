"""Unit tests for engine/security.py."""

import os
import tempfile
import unittest
from engine.security import SingleInstanceLock, secure_file_permissions


class TestSecurityModule(unittest.TestCase):
    def test_single_instance_lock_mutex(self):
        """Verify Windows Kernel Named Mutex prevents duplicate instances."""
        lock1 = SingleInstanceLock("Local\\Woeyyy_UnitTest_Mutex")
        acquired1 = lock1.acquire()
        self.assertTrue(acquired1, "First instance must successfully acquire lock")

        lock2 = SingleInstanceLock("Local\\Woeyyy_UnitTest_Mutex")
        acquired2 = lock2.acquire()
        self.assertFalse(acquired2, "Second instance must be rejected as already running")

        # Release first instance
        lock1.release()
        lock2.release()

        # Re-acquire should now succeed
        lock3 = SingleInstanceLock("Local\\Woeyyy_UnitTest_Mutex")
        acquired3 = lock3.acquire()
        self.assertTrue(acquired3, "Lock can be acquired again after release")
        lock3.release()

    def test_focus_existing_window(self):
        """Verify window focusing executes safely without error."""
        result = SingleInstanceLock.focus_existing_window("NonExistentWindow_UnitTest_12345")
        self.assertFalse(result)

    def test_secure_file_permissions(self):
        """Verify file permissions hardening executes safely on a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            secure_file_permissions(temp_path)
            self.assertTrue(os.path.exists(temp_path))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

