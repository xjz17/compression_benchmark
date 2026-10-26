#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import download_project179_thulab_browser_v1 as browser_dl  # noqa: E402


REVISION = "a" * 40


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_fixture(root: Path, files: list[tuple[str, bytes]]) -> tuple[Path, Path]:
    scope = {
        "schema": browser_dl.SCOPE_SCHEMA,
        "timestamp_dataset_count": 1,
        "timestamp_datasets": [{
            "id": "toy",
            "repo_id": "THULab/toy",
            "revision": REVISION,
        }],
    }
    scope_path = root / "scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    receipts = root / "source_receipts"
    receipts.mkdir()
    receipt = {
        "schema": browser_dl.RECEIPT_SCHEMA,
        "status": "complete",
        "dataset_id": "toy",
        "repo_id": "THULab/toy",
        "revision": REVISION,
        "files": [
            {"path": path, "bytes": len(payload), "sha256": digest(payload)}
            for path, payload in files
        ],
    }
    (receipts / "toy.json").write_text(json.dumps(receipt), encoding="utf-8")
    return scope_path, receipts


class FakeBrowser:
    def __init__(
        self,
        payload_by_path: dict[str, bytes],
        *,
        error: Exception | None = None,
        partial: bytes | None = None,
    ) -> None:
        self.payload_by_path = payload_by_path
        self.error = error
        self.partial = partial
        self.calls: list[tuple[str, str]] = []
        self.current_url = ""

    def open(self, url: str) -> None:
        self.current_url = url
        self.calls.append(("open", url))

    def snapshot(self) -> str:
        self.calls.append(("snapshot", self.current_url))
        return json.dumps({"data": {"snapshot": '- link "Download file" [ref=e17]'}})

    def full_snapshot(self) -> str:
        self.calls.append(("full_snapshot", self.current_url))
        return self.snapshot()

    def title(self) -> str:
        self.calls.append(("title", self.current_url))
        return "THULab frozen file"

    def get_href(self, ref: str) -> str:
        self.calls.append(("get_href", ref))
        marker = f"/blob/{REVISION}/"
        repo_path = self.current_url.split(marker, 1)[1]
        return f"/datasets/THULab/toy/resolve/{REVISION}/{repo_path}?download=true"

    def download(
        self,
        ref: str,
        path: Path,
        *,
        resolved_url: str,
        frozen: browser_dl.FrozenObject,
    ) -> None:
        self.calls.append(("download", str(path)))
        marker = f"/blob/{REVISION}/"
        repo_path = self.current_url.split(marker, 1)[1]
        if self.partial is not None:
            (path.parent / "browser-guid").write_bytes(self.partial)
        if self.error is not None:
            raise self.error
        path.write_bytes(self.payload_by_path[repo_path])

    def arm_download_directory(self, directory: Path) -> None:
        self.calls.append(("arm_download_directory", str(directory)))
        (directory / browser_dl.DOWNLOAD_PATH_ARM_FILE).write_bytes(
            browser_dl.DOWNLOAD_PATH_ARM_BYTES
        )

    def click_download(
        self,
        ref: str,
        directory: Path,
        *,
        resolved_url: str,
        frozen: browser_dl.FrozenObject,
    ) -> None:
        self.calls.append(("click_download", str(directory)))
        marker = f"/blob/{REVISION}/"
        repo_path = self.current_url.split(marker, 1)[1]
        (directory / "browser-payload").write_bytes(
            self.payload_by_path[repo_path]
        )

    def screenshot(self, path: Path) -> None:
        self.calls.append(("screenshot", str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")


class ForbiddenBrowser(FakeBrowser):
    def __init__(self) -> None:
        super().__init__({})

    def open(self, url: str) -> None:
        raise AssertionError("browser must not be used")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class BrowserDownloaderTests(unittest.TestCase):
    def test_source_line_429_is_not_an_http_rate_limit(self):
        snapshot = json.dumps({
            "data": {
                "snapshot": (
                    '- heading ".gitattributes"\n'
                    '- cell "429"\n'
                    '- cell "finance1k_panel_26.tsfile filter=lfs"'
                )
            }
        })
        self.assertFalse(browser_dl.page_is_rate_limited(
            snapshot, ".gitattributes · THULab/finance1k"
        ))
        self.assertTrue(browser_dl.page_is_rate_limited(
            '- heading "429 Too Many Requests"', "429 Too Many Requests"
        ))
        self.assertTrue(browser_dl.page_is_rate_limited(
            "We had to rate limit your IP. Please retry later.", "THULab frozen file"
        ))

    def test_load_inventory_binds_scope_receipt_and_fixed_blob_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("nested/a b.tsfile", b"abc")])
            inventory = browser_dl.load_inventory(scope, receipts)
            self.assertEqual(len(inventory), 1)
            self.assertEqual(inventory[0].relative_path, Path("toy/nested/a b.tsfile"))
            self.assertEqual(
                inventory[0].blob_url,
                f"https://huggingface.co/datasets/THULab/toy/blob/{REVISION}/nested/a%20b.tsfile",
            )

    def test_receipt_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("../escape.tsfile", b"abc")])
            with self.assertRaises(browser_dl.BrowserMirrorError):
                browser_dl.load_inventory(scope, receipts)

    def test_matching_existing_file_is_skipped_without_browser(self):
        payload = b"already-present"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("a.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            target = destination / "toy" / "a.tsfile"
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
            summary = browser_dl.run_download(
                inventory, destination, ForbiddenBrowser(), screenshot_every=0
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["counts"], {"already_verified": 1})

    def test_conflicting_existing_file_is_never_overwritten(self):
        payload = b"expected"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("a.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            target = destination / "toy" / "a.tsfile"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"user-data")
            with self.assertRaises(browser_dl.BrowserMirrorError):
                browser_dl.run_download(
                    inventory, destination, ForbiddenBrowser(), screenshot_every=0
                )
            self.assertEqual(target.read_bytes(), b"user-data")

    def test_browser_download_validates_href_then_publishes(self):
        payload = b"downloaded-with-browser"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("nested/a.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = FakeBrowser({"nested/a.tsfile": payload})
            summary = browser_dl.run_download(
                inventory, destination, browser, screenshot_every=1
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["counts"], {"downloaded": 1})
            self.assertEqual(
                (destination / "toy" / "nested" / "a.tsfile").read_bytes(), payload
            )
            self.assertEqual([call[0] for call in browser.calls[:5]], [
                "open", "snapshot", "title", "get_href", "download"
            ])
            self.assertTrue(any(call[0] == "screenshot" for call in browser.calls))

    def test_new_page_navigations_are_throttled_without_changing_urls(self):
        first = b"first"
        second = b"second"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(
                root, [("a.tsfile", first), ("b.tsfile", second)]
            )
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = FakeBrowser({"a.tsfile": first, "b.tsfile": second})
            clock = FakeClock()
            summary = browser_dl.run_download(
                inventory,
                destination,
                browser,
                screenshot_every=0,
                navigation_interval_seconds=5.0,
                sleep_fn=clock.sleep,
                monotonic_fn=clock.monotonic,
            )
            opens = [value for verb, value in browser.calls if verb == "open"]
            self.assertEqual(opens, [item.blob_url for item in inventory])
            self.assertEqual(clock.sleeps, [5.0])
            self.assertEqual(summary["counts"], {"downloaded": 2})

    def test_noninteractive_hf_429_cools_with_heartbeats_then_retries_same_object(self):
        payload = b"after-rate-limit"

        class OnceRateLimitedBrowser(FakeBrowser):
            def __init__(self) -> None:
                super().__init__({"a.tsfile": payload})
                self.open_count = 0

            def open(self, url: str) -> None:
                super().open(url)
                self.open_count += 1

            def snapshot(self) -> str:
                self.calls.append(("snapshot", self.current_url))
                if self.open_count == 1:
                    return json.dumps({"data": {"snapshot": '- button "Retry" [ref=e2]'}})
                return json.dumps({"data": {"snapshot": '- link "Download" [ref=e17]'}})

            def full_snapshot(self) -> str:
                self.calls.append(("full_snapshot", self.current_url))
                return "We had to rate limit your IP. Please retry later."

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("a.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = OnceRateLimitedBrowser()
            clock = FakeClock()
            output = io.StringIO()
            with redirect_stdout(output):
                summary = browser_dl.run_download(
                    inventory,
                    destination,
                    browser,
                    screenshot_every=0,
                    navigation_interval_seconds=5.0,
                    rate_limit_cooldown_seconds=360.0,
                    rate_limit_max_retries=3,
                    sleep_fn=clock.sleep,
                    monotonic_fn=clock.monotonic,
                )
            opens = [value for verb, value in browser.calls if verb == "open"]
            self.assertEqual(opens, [inventory[0].blob_url, inventory[0].blob_url])
            self.assertEqual(clock.sleeps, [30.0] * 12)
            heartbeat_lines = [line for line in output.getvalue().splitlines() if line]
            self.assertGreaterEqual(len(heartbeat_lines), 12)
            self.assertTrue(all("next_retry_at" in line for line in heartbeat_lines))
            event_receipts = list(
                (destination / browser_dl.CONTROL_DIRECTORY / "staging").rglob(
                    "rate_limit_events/001.json"
                )
            )
            self.assertEqual(len(event_receipts), 1)
            event = json.loads(event_receipts[0].read_text(encoding="utf-8"))
            self.assertEqual(event["status"], "rate_limit_cooldown")
            self.assertEqual(event["rate_limit_response"], 1)
            self.assertTrue(Path(event["screenshot"]).is_file())
            progress = json.loads(
                (
                    destination
                    / browser_dl.CONTROL_DIRECTORY
                    / "PROGRESS.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIsNone(progress["active"])
            self.assertEqual(summary["counts"], {"downloaded": 1})
            self.assertEqual(
                (destination / "toy" / "a.tsfile").read_bytes(), payload
            )

    def test_general_page_failure_is_not_retried_as_429(self):
        payload = b"never-downloaded"

        class GeneralFailureBrowser(FakeBrowser):
            def snapshot(self) -> str:
                self.calls.append(("snapshot", self.current_url))
                return '- heading "Service unavailable"'

            def full_snapshot(self) -> str:
                self.calls.append(("full_snapshot", self.current_url))
                return "The page could not be loaded."

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("a.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = GeneralFailureBrowser({"a.tsfile": payload})
            clock = FakeClock()
            with self.assertRaises(browser_dl.BrowserMirrorError):
                browser_dl.run_download(
                    inventory,
                    destination,
                    browser,
                    screenshot_every=0,
                    sleep_fn=clock.sleep,
                    monotonic_fn=clock.monotonic,
                )
            self.assertEqual(
                [verb for verb, _value in browser.calls].count("open"), 1
            )
            self.assertEqual(clock.sleeps, [])

    def test_429_retries_are_bounded_and_failed_receipt_does_not_block_restart(self):
        payload = b"restart-after-bounded-429"

        class AlwaysRateLimitedBrowser(FakeBrowser):
            def snapshot(self) -> str:
                self.calls.append(("snapshot", self.current_url))
                return '- heading "429 Too Many Requests"'

            def title(self) -> str:
                self.calls.append(("title", self.current_url))
                return "429 Too Many Requests"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("a.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            limited = AlwaysRateLimitedBrowser({"a.tsfile": payload})
            clock = FakeClock()
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(browser_dl.RateLimitRetryExhaustedError):
                    browser_dl.run_download(
                        inventory,
                        destination,
                        limited,
                        screenshot_every=0,
                        rate_limit_cooldown_seconds=360.0,
                        rate_limit_max_retries=1,
                        sleep_fn=clock.sleep,
                        monotonic_fn=clock.monotonic,
                    )
            self.assertEqual(
                [verb for verb, _value in limited.calls].count("open"), 2
            )
            self.assertEqual(clock.sleeps, [30.0] * 12)
            event_receipts = list(
                (destination / browser_dl.CONTROL_DIRECTORY / "staging").rglob(
                    "rate_limit_events/*.json"
                )
            )
            self.assertEqual(len(event_receipts), 2)
            self.assertFalse((destination / "toy" / "a.tsfile").exists())

            # A new invocation must ignore old attempt.json/event metadata and
            # create a fresh attempt for the exact same frozen object.
            recovered_browser = FakeBrowser({"a.tsfile": payload})
            restart_clock = FakeClock()
            summary = browser_dl.run_download(
                inventory,
                destination,
                recovered_browser,
                screenshot_every=0,
                sleep_fn=restart_clock.sleep,
                monotonic_fn=restart_clock.monotonic,
            )
            self.assertEqual(summary["counts"], {"downloaded": 1})
            self.assertEqual(
                (destination / "toy" / "a.tsfile").read_bytes(), payload
            )

    def test_timeout_preserves_partial_and_completed_stage_is_recovered(self):
        payload = b"large-browser-object"
        timeout = browser_dl.BrowserCommandError("30 second timeout", timed_out=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("large.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = FakeBrowser({}, error=timeout, partial=payload[:5])
            with self.assertRaises(browser_dl.BrowserCommandError):
                browser_dl.run_download(
                    inventory,
                    destination,
                    browser,
                    screenshot_every=0,
                    post_timeout_wait_seconds=0,
                )
            self.assertFalse((destination / "toy" / "large.tsfile").exists())
            attempts = list(
                (destination / browser_dl.CONTROL_DIRECTORY / "staging").rglob("attempt.json")
            )
            self.assertEqual(len(attempts), 1)
            attempt = json.loads(attempts[0].read_text(encoding="utf-8"))
            self.assertEqual(attempt["status"], "browser_partial_retained")
            partials = [
                path for path in attempts[0].parent.iterdir()
                if path.name != "attempt.json"
            ]
            self.assertEqual(len(partials), 1)
            self.assertEqual(partials[0].read_bytes(), payload[:5])

            # Simulate Chrome finishing the timed-out GUID file before restart.
            partials[0].write_bytes(payload)
            summary = browser_dl.run_download(
                inventory, destination, ForbiddenBrowser(), screenshot_every=0
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["counts"], {"recovered": 1})
            self.assertEqual(
                (destination / "toy" / "large.tsfile").read_bytes(), payload
            )

    def test_continue_mode_stops_after_timeout_to_avoid_overlapping_downloads(self):
        first = b"large"
        second = b"small"
        timeout = browser_dl.BrowserCommandError("30 second timeout", timed_out=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(
                root, [("a.tsfile", first), ("b.tsfile", second)]
            )
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = FakeBrowser({}, error=timeout, partial=b"x")
            summary = browser_dl.run_download(
                inventory,
                destination,
                browser,
                screenshot_every=0,
                fail_fast=False,
                post_timeout_wait_seconds=0,
            )
            self.assertEqual(summary["counts"], {"pending_partial": 1})
            self.assertEqual(
                [call[0] for call in browser.calls].count("download"), 1
            )

    def test_download_ref_and_href_must_be_unambiguous_and_frozen(self):
        frozen = browser_dl.FrozenObject(
            "toy", "THULab/toy", REVISION, "a.tsfile", 1, digest(b"x")
        )
        self.assertEqual(
            browser_dl.find_download_refs('- link "Download" [ref=e9]'), ["e9"]
        )
        self.assertEqual(
            browser_dl.find_download_refs(
                '- link "Download" [ref=e1]\n- link "download" [ref=e2]'
            ),
            ["e1", "e2"],
        )
        with self.assertRaises(browser_dl.BrowserMirrorError):
            browser_dl.validate_download_href(
                f"/datasets/THULab/toy/resolve/{'b' * 40}/a.tsfile", frozen
            )

    def test_duplicate_download_labels_are_resolved_by_frozen_href(self):
        frozen = browser_dl.FrozenObject(
            "toy", "THULab/toy", REVISION, "a.tsfile", 1, digest(b"x")
        )

        class CandidateBrowser(ForbiddenBrowser):
            def get_href(self, ref: str) -> str:
                if ref == "e1":
                    return f"/datasets/THULab/toy/blob/{REVISION}/a.tsfile"
                return f"/datasets/THULab/toy/resolve/{REVISION}/a.tsfile?download=true"

        ref, href = browser_dl.select_download_link(
            '- link "Download" [ref=e1]\n- link "download" [ref=e2]',
            CandidateBrowser(),
            frozen,
        )
        self.assertEqual(ref, "e2")
        self.assertIn(f"/resolve/{REVISION}/a.tsfile", href)

    def test_raw_metadata_link_is_validated_and_forced_through_browser_download(self):
        frozen = browser_dl.FrozenObject(
            "toy", "THULab/toy", REVISION, ".gitattributes", 1, digest(b"x")
        )
        raw = f"/datasets/THULab/toy/raw/{REVISION}/.gitattributes"

        class RawBrowser(ForbiddenBrowser):
            def get_href(self, ref: str) -> str:
                return raw

        ref, resolved = browser_dl.select_download_link(
            '- link "Raw" [ref=e21]\n- button "Copy download link" [ref=e39]',
            RawBrowser(),
            frozen,
        )
        self.assertEqual(ref, "e21")
        self.assertIn(f"/raw/{REVISION}/.gitattributes", resolved)
        forced = browser_dl.force_browser_download_url(resolved, frozen)
        self.assertEqual(
            forced,
            f"https://huggingface.co/datasets/THULab/toy/resolve/{REVISION}/.gitattributes?download=true",
        )

        class RecordingClient(browser_dl.AgentBrowserClient):
            def __init__(self) -> None:
                super().__init__(Path("/agent-browser"), cdp="9222", session="test")
                self.commands = []

            def _run(self, *arguments: str, **kwargs) -> str:
                self.commands.append((arguments, kwargs))
                return ""

        client = RecordingClient()
        requested = Path("/tmp/unique-attempt/payload.download")
        client.download(
            "e21", requested, resolved_url=resolved, frozen=frozen
        )
        self.assertEqual(client.commands[0][0][0], "eval")
        self.assertIn(f"/resolve/{REVISION}/.gitattributes?download=true", client.commands[0][0][1])
        self.assertEqual(client.commands[1][0][0], "download")
        self.assertEqual(client.commands[1][1]["download_path"], requested.parent)

    def test_agent_large_flow_arms_with_tiny_download_then_plain_clicks_once(self):
        frozen = browser_dl.FrozenObject(
            "toy", "THULab/toy", REVISION, "a.tsfile", 1, digest(b"x")
        )
        resolved = (
            f"https://huggingface.co/datasets/THULab/toy/resolve/"
            f"{REVISION}/a.tsfile?download=true"
        )

        class RecordingClient(browser_dl.AgentBrowserClient):
            def __init__(self) -> None:
                super().__init__(Path("/agent-browser"), cdp="9222", session="test")
                self.commands = []

            def _run(self, *arguments: str, **kwargs) -> str:
                self.commands.append((arguments, kwargs))
                if arguments[0] == "download":
                    Path(arguments[2]).write_bytes(browser_dl.DOWNLOAD_PATH_ARM_BYTES)
                return ""

        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "unique-attempt"
            attempt.mkdir()
            client = RecordingClient()
            client.arm_download_directory(attempt)
            client.click_download(
                "e17", attempt, resolved_url=resolved, frozen=frozen
            )
            verbs = [command[0][0] for command in client.commands]
            self.assertEqual(verbs, ["eval", "download", "click"])
            self.assertEqual(client.commands[1][1]["download_path"], attempt)
            self.assertEqual(client.commands[2][1]["download_path"], attempt)
            self.assertEqual(client.commands[1][1]["timeout"], 20.0)
            self.assertEqual(client.commands[2][1]["timeout"], 20.0)

    def test_timeout_monitor_recovers_after_crdownload_finishes(self):
        payload = b"eventually-complete-large-file"
        timeout = browser_dl.BrowserCommandError("30 second timeout", timed_out=True)

        class DeferredBrowser(FakeBrowser):
            def download(
                self,
                ref: str,
                path: Path,
                *,
                resolved_url: str,
                frozen: browser_dl.FrozenObject,
            ) -> None:
                partial = path.parent / "browser-file.crdownload"
                partial.write_bytes(payload[:4])

                def finish() -> None:
                    partial.unlink(missing_ok=True)
                    (path.parent / "browser-file").write_bytes(payload)

                threading.Timer(0.05, finish).start()
                raise timeout

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("large.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            summary = browser_dl.run_download(
                inventory,
                destination,
                DeferredBrowser({}),
                screenshot_every=0,
                post_timeout_wait_seconds=2,
                poll_seconds=0.01,
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(
                (destination / "toy" / "large.tsfile").read_bytes(), payload
            )

    def test_large_file_arms_directory_then_uses_one_plain_click(self):
        payload = b"large-single-click-payload"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("large.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = FakeBrowser({"large.tsfile": payload})
            summary = browser_dl.run_download(
                inventory,
                destination,
                browser,
                screenshot_every=0,
                poll_seconds=0.01,
                large_file_threshold_bytes=1,
            )
            verbs = [call[0] for call in browser.calls]
            self.assertEqual(verbs.count("arm_download_directory"), 1)
            self.assertEqual(verbs.count("click_download"), 1)
            self.assertEqual(verbs.count("download"), 0)
            self.assertEqual(verbs.count("snapshot"), 2)
            self.assertLess(
                verbs.index("arm_download_directory"),
                len(verbs) - 1 - verbs[::-1].index("snapshot"),
            )
            self.assertEqual(summary["counts"], {"downloaded": 1})
            self.assertEqual(
                (destination / "toy" / "large.tsfile").read_bytes(), payload
            )

    def test_large_single_click_fails_closed_on_two_active_payloads(self):
        payload = b"large-duplicate-test"

        class DuplicateClickBrowser(FakeBrowser):
            def click_download(
                self,
                ref: str,
                directory: Path,
                *,
                resolved_url: str,
                frozen: browser_dl.FrozenObject,
            ) -> None:
                self.calls.append(("click_download", str(directory)))
                (directory / "first.crdownload").write_bytes(payload[:3])
                (directory / "second.crdownload").write_bytes(payload[:5])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("large.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            browser = DuplicateClickBrowser({"large.tsfile": payload})
            with self.assertRaises(browser_dl.DuplicateBrowserDownloadError):
                browser_dl.run_download(
                    inventory,
                    destination,
                    browser,
                    screenshot_every=0,
                    post_timeout_wait_seconds=0,
                    poll_seconds=0.01,
                    large_file_threshold_bytes=1,
                )
            self.assertFalse((destination / "toy" / "large.tsfile").exists())
            payloads = [
                path
                for path in (
                    destination / browser_dl.CONTROL_DIRECTORY / "staging"
                ).rglob("*.crdownload")
            ]
            self.assertEqual(len(payloads), 2)
            attempts = list(
                (destination / browser_dl.CONTROL_DIRECTORY / "staging").rglob(
                    "attempt.json"
                )
            )
            receipt = json.loads(attempts[0].read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["status"], "duplicate_browser_payloads_retained"
            )

    def test_large_single_click_fails_closed_on_complete_plus_active_payload(self):
        payload = b"large-complete-plus-active-test"

        class MixedClickBrowser(FakeBrowser):
            def click_download(
                self,
                ref: str,
                directory: Path,
                *,
                resolved_url: str,
                frozen: browser_dl.FrozenObject,
            ) -> None:
                self.calls.append(("click_download", str(directory)))
                (directory / "first-complete").write_bytes(payload)
                (directory / "second.crdownload").write_bytes(payload[:5])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("large.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            with self.assertRaises(browser_dl.DuplicateBrowserDownloadError):
                browser_dl.run_download(
                    inventory,
                    destination,
                    MixedClickBrowser({"large.tsfile": payload}),
                    screenshot_every=0,
                    post_timeout_wait_seconds=0,
                    poll_seconds=0.01,
                    large_file_threshold_bytes=1,
                )
            self.assertFalse((destination / "toy" / "large.tsfile").exists())
            staged = [
                path
                for path in (
                    destination / browser_dl.CONTROL_DIRECTORY / "staging"
                ).rglob("*")
                if path.is_file()
                and path.name
                not in {"attempt.json", browser_dl.DOWNLOAD_PATH_ARM_FILE}
            ]
            self.assertEqual({path.name for path in staged}, {
                "first-complete", "second.crdownload"
            })

    def test_matching_crdownload_alone_is_never_published(self):
        payload = b"complete-bytes-but-browser-still-owns-the-file"

        class ActiveClickBrowser(FakeBrowser):
            def click_download(
                self,
                ref: str,
                directory: Path,
                *,
                resolved_url: str,
                frozen: browser_dl.FrozenObject,
            ) -> None:
                self.calls.append(("click_download", str(directory)))
                (directory / "only.crdownload").write_bytes(payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("large.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            destination = root / "mirror"
            with self.assertRaises(browser_dl.PendingBrowserDownloadError):
                browser_dl.run_download(
                    inventory,
                    destination,
                    ActiveClickBrowser({"large.tsfile": payload}),
                    screenshot_every=0,
                    post_timeout_wait_seconds=0,
                    poll_seconds=0.01,
                    large_file_threshold_bytes=1,
                )
            self.assertFalse((destination / "toy" / "large.tsfile").exists())
            active = list(
                (destination / browser_dl.CONTROL_DIRECTORY / "staging").rglob(
                    "*.crdownload"
                )
            )
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0].read_bytes(), payload)

    def test_old_identical_complete_duplicates_publish_one_and_retain_one(self):
        payload = b"old-retry-produced-two-identical-complete-files"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, receipts = write_fixture(root, [("large.tsfile", payload)])
            inventory = browser_dl.load_inventory(scope, receipts)
            frozen = inventory[0]
            destination = root / "mirror"
            key_directory = (
                destination
                / browser_dl.CONTROL_DIRECTORY
                / "staging"
                / frozen.dataset_id
                / frozen.object_key
            )
            attempt = key_directory / "attempt-old"
            attempt.mkdir(parents=True)
            (attempt / "first-complete").write_bytes(payload)
            (attempt / "second-complete").write_bytes(payload)

            summary = browser_dl.run_download(
                inventory, destination, ForbiddenBrowser(), screenshot_every=0
            )
            self.assertEqual(summary["counts"], {"recovered": 1})
            self.assertIn("retained 1 duplicate", summary["outcomes"][0]["detail"])
            self.assertEqual(
                (destination / "toy" / "large.tsfile").read_bytes(), payload
            )
            retained = list(browser_dl._candidate_files(attempt))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), payload)

    def test_publish_no_clobber_keeps_conflicting_final_bytes(self):
        payload = b"expected"
        frozen = browser_dl.FrozenObject(
            "toy", "THULab/toy", REVISION, "a.tsfile", len(payload), digest(payload)
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "mirror"
            destination.mkdir()
            control = destination / browser_dl.CONTROL_DIRECTORY
            control.mkdir()
            staged = destination / "staged"
            staged.write_bytes(payload)
            final = destination / "toy" / "a.tsfile"
            final.parent.mkdir()
            final.write_bytes(b"conflict")
            with self.assertRaises(browser_dl.BrowserMirrorError):
                browser_dl.publish_no_clobber(staged, destination, control, frozen)
            self.assertEqual(final.read_bytes(), b"conflict")
            self.assertEqual(staged.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
