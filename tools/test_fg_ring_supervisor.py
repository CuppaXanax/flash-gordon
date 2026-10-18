import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "fg_ring_supervisor", Path(__file__).with_name("fg_ring_supervisor.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
control_spec = importlib.util.spec_from_file_location(
    "fg_ring_control", Path(__file__).with_name("fg_ring_control.py")
)
control = importlib.util.module_from_spec(control_spec)
control_spec.loader.exec_module(control)


class SupervisorTests(unittest.TestCase):
    def make(self):
        supervisor = module.Supervisor({
            "operator": ".", "nodes": [{"rank": i, "host": str(i)} for i in range(4)]
        })
        supervisor.events = []
        supervisor.command = lambda node, action: supervisor.events.append(
            (node["rank"], action)
        )
        supervisor.ready = lambda epoch=None: supervisor.events.append(("ready", epoch))
        return supervisor

    def test_cold_boot_starts_workers_before_coordinator(self):
        supervisor = self.make()
        supervisor.restart({"status": "down"})
        events = supervisor.events
        self.assertEqual(events[0], (0, "stop"))
        self.assertLess(
            max(i for i, event in enumerate(events) if event[1] == "stop"),
            min(i for i, event in enumerate(events) if event[1] == "start"),
        )
        self.assertEqual(events[-2], (0, "start"))
        self.assertEqual(events[-1], ("ready", None))

    def test_transport_recovery_keeps_api_pid(self):
        supervisor = self.make()
        supervisor.restart({"status": "recovering", "ring_recovery": True,
                            "recovery_epoch": 7})
        self.assertNotIn((0, "stop"), supervisor.events)
        self.assertNotIn((0, "start"), supervisor.events)
        self.assertEqual(supervisor.events[-1], ("ready", 7))

    def test_worker_failure_waits_for_new_epoch(self):
        supervisor = self.make()
        supervisor.restart({"status": "ok", "ring_recovery": True, "recovery_epoch": 7})
        self.assertEqual(supervisor.events[-1], ("ready", 8))

    def test_reject_duplicate_ranks(self):
        with self.assertRaises(ValueError):
            module.Supervisor({"operator": ".", "nodes": [
                {"rank": 0, "host": "a"}, {"rank": 0, "host": "b"}]})

    def test_stop_prevents_restart(self):
        supervisor = self.make()
        supervisor.stop.set()
        with self.assertRaises(InterruptedError):
            supervisor.restart({"status": "down"})
        self.assertEqual(supervisor.events, [])

    def test_pause_mid_restart_prevents_start(self):
        supervisor = self.make()
        with patch.object(module.Path, "exists", side_effect=[False, False, True]):
            with self.assertRaises(InterruptedError):
                supervisor.restart({"status": "down"})
        self.assertFalse(any(action == "start" for _, action in supervisor.events))

    def test_control_rejects_arbitrary_ssh_command(self):
        with self.assertRaises(ValueError):
            control.control(".", "start; echo unexpected")

    def test_control_rejects_foreign_pid(self):
        with patch.object(control.Path, "read_text", side_effect=[
            "1", '{"release":"/release","manifest":"/pack"}', "123"
        ]), patch.object(control.os, "readlink", return_value="/unowned"), \
                patch.object(control.Path, "read_bytes", return_value=b"--manifest\0/pack\0"):
            with self.assertRaises(ValueError):
                control.control(".", "status")


if __name__ == "__main__":
    unittest.main()
