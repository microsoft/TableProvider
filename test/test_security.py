import importlib.util
import json
import logging
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safe_output = load_module("safe_output_test", "table_provider/libs/safe_output.py")


@unittest.skipUnless(importlib.util.find_spec("flask"), "Web tests require Flask")
class RequestSecurityTests(unittest.TestCase):
    TOKEN = "test-token-" + "a" * 40

    def setUp(self):
        from flask import Flask
        self.app = Flask(__name__)
        self.app.config.update(TESTING=True, TABLEPROVIDER_API_TOKEN=self.TOKEN,
                               TABLEPROVIDER_ALLOWED_HOSTS=["localhost:8080", "127.0.0.1:8080"])
        self.security = load_module("request_security_test", "table_provider/route/security.py")
        self.security.protect_requests(self.app)
        self.calls = []
        @self.app.route("/table_provider/execution", methods=["POST"])
        def execute():
            self.calls.append(True)
            return {"ok": True}
        self.client = self.app.test_client()

    def post(self, headers=None, base_url="http://localhost:8080"):
        return self.client.post("/table_provider/execution", json={}, headers=headers or {}, base_url=base_url)

    def test_requires_explicit_token_not_cookie_or_query(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Cookie": "token=" + self.TOKEN}):
            with self.subTest(headers=list(headers)):
                self.assertEqual(self.post(headers).status_code, 401)
        self.assertEqual(self.client.post("/table_provider/execution?token=" + self.TOKEN, json={}, base_url="http://localhost:8080").status_code, 401)
        self.assertEqual(self.calls, [])

    def test_valid_non_browser_and_same_origin_requests(self):
        for origin in (None, "http://localhost:8080"):
            headers = {"Authorization": "Bearer " + self.TOKEN}
            if origin:
                headers["Origin"] = origin
            self.assertEqual(self.post(headers).status_code, 200)
        self.assertEqual(len(self.calls), 2)

    def test_cross_origin_and_null_origin_rejected(self):
        for origin in ("https://attacker.example", "null", "http://localhost:9090", "http://localhost:8080.attacker.example"):
            with self.subTest(origin=origin):
                self.assertEqual(self.post({"Authorization": "Bearer " + self.TOKEN, "Origin": origin}).status_code, 403)
        self.assertEqual(self.calls, [])

    def test_dns_rebinding_host_and_forwarded_headers_rejected(self):
        headers = {"Authorization": "Bearer " + self.TOKEN, "X-Forwarded-Host": "localhost:8080", "X-Forwarded-For": "127.0.0.1"}
        self.assertEqual(self.post(headers, "http://attacker.example:8080").status_code, 400)
        self.assertEqual(self.calls, [])

    def test_cross_site_metadata_rejected_without_origin(self):
        for site in ("cross-site", "same-site"):
            self.assertEqual(self.post({"Authorization": "Bearer " + self.TOKEN, "Sec-Fetch-Site": site}).status_code, 403)
        self.assertEqual(self.calls, [])

    def test_cors_preflight_does_not_grant_access(self):
        response = self.client.options("/table_provider/execution", base_url="http://localhost:8080", headers={"Origin": "https://attacker.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"})
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_missing_token_prevents_startup(self):
        from flask import Flask
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            self.security.protect_requests(Flask("missing-token"))

    def test_invalid_security_configuration_fails_closed(self):
        from flask import Flask
        for config in ({"TABLEPROVIDER_API_TOKEN": "short"}, {"TABLEPROVIDER_ALLOWED_HOSTS": ["*"]}, {"TABLEPROVIDER_ALLOWED_HOSTS": ["http://localhost:8080"]}):
            app = Flask("invalid-config")
            app.config.update(TABLEPROVIDER_API_TOKEN=self.TOKEN, TABLEPROVIDER_ALLOWED_HOSTS=["localhost:8080"])
            app.config.update(config)
            with self.subTest(config=list(config)), self.assertRaises(RuntimeError):
                self.security.protect_requests(app)

    def test_authorized_payload_size_is_bounded(self):
        response = self.client.post("/table_provider/execution", base_url="http://localhost:8080",
                                    data="x" * (1024 * 1024 + 1), content_type="application/json",
                                    headers={"Authorization": "Bearer " + self.TOKEN})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.calls, [])

    def test_ambiguous_or_non_bearer_credentials_do_not_authorize(self):
        for authorization in ("Basic " + self.TOKEN, "Bearer " + self.TOKEN + ", Bearer " + self.TOKEN,
                              "Bearer  " + self.TOKEN, "Bearer " + self.TOKEN + " ", "Bearer \u00e9"):
            with self.subTest(authorization=authorization[:15]):
                self.assertEqual(self.post({"Authorization": authorization}).status_code, 401)
        self.assertEqual(self.calls, [])

    def test_host_suffix_and_origin_tricks_are_rejected(self):
        for host in ("localhost:8080.attacker.example", "localhost.attacker.example:8080", "127.0.0.1.attacker.example:8080"):
            with self.subTest(host=host):
                self.assertEqual(self.post({"Authorization": "Bearer " + self.TOKEN, "Host": host}).status_code, 400)
        for origin in ("http://localhost:8080/", "http://localhost:8080@attacker.example", "http://localhost:8080, https://attacker.example"):
            with self.subTest(origin=origin):
                self.assertEqual(self.post({"Authorization": "Bearer " + self.TOKEN, "Origin": origin}).status_code, 403)
        self.assertEqual(self.calls, [])


@unittest.skipUnless(importlib.util.find_spec("flask_restful") and importlib.util.find_spec("pydantic"), "Endpoint integration requires Flask-RESTful and Pydantic")
class PipelineSecurityTests(unittest.TestCase):
    TOKEN = RequestSecurityTests.TOKEN

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "output"
        self.calls = []
        calls = self.calls
        class Orchestrator:
            def __init__(self):
                calls.append("created")
            def execute(self, request, context, response):
                calls.append("executed")
                response.result = {"ok": True}
        package = types.ModuleType("table_provider")
        package.__path__ = [str(ROOT / "table_provider")]
        package.output_directory = str(self.output)
        package.logger = logging.getLogger("tableprovider-security-test")
        orchestrator = types.ModuleType("table_provider.orchestrator.orchestrators.table_provider_orchestrator")
        orchestrator.TableProviderOrchestrator = Orchestrator
        progress = types.ModuleType("tqdm")
        progress.tqdm = lambda iterable, **kwargs: iterable
        formatter = types.ModuleType("table_provider.service.table_format_service")
        class Formatter:
            def __init__(self, config):
                pass
            def format_sampled_table_or_augmentation_info(self, **kwargs):
                return [{"value": "formatted"}]
            def format_tables_composition(self, **kwargs):
                return [{"value": "combined"}]
        formatter.TableFormatter = Formatter
        self.modules = patch.dict(sys.modules, {"table_provider": package,
            orchestrator.__name__: orchestrator, formatter.__name__: formatter, "tqdm": progress})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.main = load_module("tableprovider_main_test", "main.py")
        self.app = self.main.create_app({"TESTING": True, "TABLEPROVIDER_API_TOKEN": self.TOKEN,
                                         "TABLEPROVIDER_ALLOWED_HOSTS": ["localhost:8080"]})
        self.client = self.app.test_client()

    def payload(self, identifier="job_123"):
        return {"table_loader": {"table_source": "input", "table_name": "table", "table_header": ["a"], "table_content": [[1]]},
                "table_sampling": {"sampling_type": "evenly_sample"},
                "table_augmentation": {"augmentation_type": "table_size"},
                "table_format": {"format_type": "json", "unique_format_id": identifier,
                                 "enable_separate": True, "enable_composition": True}}

    def post(self, payload=None, headers=None, base_url="http://localhost:8080"):
        return self.client.post("/table_provider/execution", json=self.payload() if payload is None else payload,
                                headers=headers or {}, base_url=base_url)

    def test_actual_route_accepts_authenticated_valid_request(self):
        response = self.post(headers={"Authorization": "Bearer " + self.TOKEN})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})
        self.assertEqual(self.calls, ["created", "executed"])
        self.assertFalse(self.app.debug)

    def test_actual_route_blocks_before_orchestrator_creation(self):
        for headers, url in (({}, "http://localhost:8080"),
                             ({"Authorization": "Bearer " + self.TOKEN, "Origin": "https://evil.example"}, "http://localhost:8080"),
                             ({"Authorization": "Bearer " + self.TOKEN}, "http://evil.example:8080")):
            with self.subTest(url=url, headers=list(headers)):
                self.assertIn(self.post(headers=headers, base_url=url).status_code, (400, 401, 403))
        self.assertEqual(self.calls, [])

    def test_actual_route_rejects_traversal_in_model(self):
        for identifier in ("../../outside", "..\\outside", "/tmp/outside", "%2e%2e%2foutside", "a\x00b"):
            with self.subTest(identifier=identifier):
                response = self.post(self.payload(identifier), {"Authorization": "Bearer " + self.TOKEN})
                self.assertEqual(response.status_code, 400)
        self.assertNotIn("executed", self.calls)
        self.assertFalse(self.output.exists())

    def test_actual_route_rejects_non_object_json(self):
        self.assertEqual(self.post([], {"Authorization": "Bearer " + self.TOKEN}).status_code, 400)
        self.assertNotIn("executed", self.calls)

    def test_actual_route_rejects_malformed_and_non_json_requests(self):
        for body, content_type in (("{broken", "application/json"), ("{}", "text/plain"), ("a=b", "application/x-www-form-urlencoded")):
            response = self.client.post("/table_provider/execution", data=body, content_type=content_type,
                                        base_url="http://localhost:8080", headers={"Authorization": "Bearer " + self.TOKEN})
            self.assertIn(response.status_code, (400, 415))
        self.assertNotIn("executed", self.calls)

    def test_actual_route_bounds_stream_without_content_length(self):
        import io
        stream = io.BytesIO(b" " * (1024 * 1024 + 1))
        response = self.client.open("/table_provider/execution", method="POST", base_url="http://localhost:8080",
                                    headers={"Authorization": "Bearer " + self.TOKEN},
                                    environ_overrides={"wsgi.input": stream,
                                                       "wsgi.input_terminated": True, "CONTENT_LENGTH": None,
                                                       "HTTP_TRANSFER_ENCODING": "chunked",
                                                       "CONTENT_TYPE": "application/json"})
        self.assertIn(response.status_code, (400, 413))
        self.assertGreater(stream.tell(), 0)
        self.assertLessEqual(stream.tell(), 1024 * 1024)
        self.assertNotIn("executed", self.calls)

    @unittest.skipUnless(os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"), "Secure writes require POSIX")
    def test_real_format_extension_saves_only_inside_root(self):
        import pandas as pd
        from table_provider.model.dto.table_provider_request import TableProviderRequest
        from table_provider.model.dto.table_provider_context import TableProviderContext
        from table_provider.model.dto.table_provider_response import TableProviderResponse
        from table_provider.orchestrator.extensions.table_format_extension import TableFormatExtension
        request = TableProviderRequest(**self.payload())
        context = TableProviderContext()
        context.sampled_table_info_map = {"evenly_sample": [pd.DataFrame({"a": [1]})], "random_sample": ["table"]}
        context.table_augmentation_info_map = {"table_size": ["one row"]}
        response = TableProviderResponse()
        extension = TableFormatExtension()
        extension._execute_internal(request, context, response)
        outputs = list(self.output.rglob("*.jsonl"))
        self.assertEqual(len(outputs), 5)
        originals = {path: path.read_bytes() for path in outputs}
        extension._execute_internal(request, context, response)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
            path.resolve().relative_to(self.output.resolve())
        outside = Path(self.temporary.name) / "outside.jsonl"
        with self.assertRaises(ValueError):
            extension._save_to_jsonl(self.output / ".." / "outside.jsonl", [{}])
        self.assertFalse(outside.exists())


class OutputIdTests(unittest.TestCase):
    def test_valid_ids(self):
        for value in ("job_123-test", "a", "a" * 128):
            self.assertEqual(safe_output.validate_output_id(value), value)
        self.assertIsNone(safe_output.validate_output_id(None))
        self.assertIsNone(safe_output.validate_output_id(""))

    def test_traversal_and_non_identifiers_rejected(self):
        for value in ("../escape", "..", "/tmp/escape", "C:\\escape", "a/b", "a\\b", "%2e%2e", "a\x00", "a\n", "a" * 129, 42):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_output.validate_output_id(value)

    def test_lexical_escape_rejected_before_filesystem_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            for target in (Path(directory) / "escape.jsonl", root / ".." / "escape.jsonl"):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    safe_output.write_jsonl(root, target, [{}])
            self.assertFalse(root.exists())


@unittest.skipUnless(os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"), "Secure writes require POSIX directory descriptors")
class SecureOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "output"
        self.target = self.root / "date" / "job" / "result.jsonl"

    def test_writes_inside_root_with_private_permissions(self):
        self.assertTrue(safe_output.write_jsonl(self.root, self.target, [{"value": "data"}]))
        self.assertEqual(json.loads(self.target.read_text()), {"value": "data"})
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o600)

    def test_existing_output_is_not_overwritten(self):
        safe_output.write_jsonl(self.root, self.target, [{"value": "original"}])
        self.assertFalse(safe_output.write_jsonl(self.root, self.target, [{"value": "new"}]))
        self.assertEqual(json.loads(self.target.read_text()), {"value": "original"})

    def test_parent_symlink_cannot_escape(self):
        self.root.mkdir()
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.root / "date").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            safe_output.write_jsonl(self.root, self.target, [{}])
        self.assertEqual(list(outside.iterdir()), [])

    def test_file_symlink_is_rejected(self):
        self.target.parent.mkdir(parents=True)
        outside = Path(self.temporary.name) / "outside.txt"
        outside.write_text("unchanged")
        self.target.symlink_to(outside)
        with self.assertRaises(ValueError):
            safe_output.write_jsonl(self.root, self.target, [{}])
        self.assertEqual(outside.read_text(), "unchanged")

    def test_root_symlink_is_rejected(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        self.root.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            safe_output.write_jsonl(self.root, self.target, [{}])
        self.assertEqual(list(outside.iterdir()), [])

    def test_parent_replacement_does_not_redirect_open_descriptor(self):
        self.target.parent.mkdir(parents=True)
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        moved = self.target.parent.with_name("original_job")
        real_open = os.open

        def replace_parent(path, flags, mode=0o777, *, dir_fd=None):
            if path == self.target.name and dir_fd is not None:
                self.target.parent.rename(moved)
                self.target.parent.symlink_to(outside, target_is_directory=True)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with patch.object(safe_output.os, "open", side_effect=replace_parent) as hooked_open:
            with patch.object(safe_output.os, "supports_dir_fd", os.supports_dir_fd | {hooked_open}):
                self.assertTrue(safe_output.write_jsonl(self.root, self.target, [{"value": "data"}]))
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(json.loads((moved / self.target.name).read_text()), {"value": "data"})

    def test_partial_write_is_removed_and_can_be_retried(self):
        with self.assertRaises(TypeError):
            safe_output.write_jsonl(self.root, self.target, [{"ok": True}, {"unsupported": object()}])
        self.assertFalse(self.target.exists())
        self.assertTrue(safe_output.write_jsonl(self.root, self.target, [{"retry": True}]))
        self.assertEqual(json.loads(self.target.read_text()), {"retry": True})

    def test_fifo_output_is_rejected_without_blocking(self):
        self.target.parent.mkdir(parents=True)
        os.mkfifo(self.target)
        with self.assertRaises(ValueError):
            safe_output.write_jsonl(self.root, self.target, [{}])


if __name__ == "__main__":
    unittest.main()