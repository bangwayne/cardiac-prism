"""Administrative setup must verify identity, preserve unrelated rules, and avoid token output."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from scripts import apply_github_rulesets as setup


class FakeGitHub:
    def __init__(self):
        self.rules = {}
        self.writes = []
        self.login = 'bangwayne'

    def request(self, method, route, payload=None):
        if route == '/user':
            return {'login': self.login, 'id': setup.OWNER_ID}
        base = '/repos/' + setup.REPOSITORY
        if route == base:
            return {'full_name': setup.REPOSITORY, 'default_branch': 'main', 'permissions': {'admin': True}}
        if route.startswith(base + '/branches/'):
            return {'name': route.rsplit('/', 1)[-1]}
        if '/rules/branches/' in route:
            branch = route.rsplit('/', 1)[-1]
            return [rule for item in self.rules.values()
                    if 'refs/heads/' + branch in item['conditions']['ref_name']['include']
                    for rule in copy.deepcopy(item['rules'])]
        if '?per_page=' in route:
            return copy.deepcopy(list(self.rules.values()))
        if method == 'GET':
            return copy.deepcopy(self.rules[int(route.rsplit('/', 1)[-1])])
        identifier = len(self.rules) + 1 if method == 'POST' else int(route.rsplit('/', 1)[-1])
        self.writes.append((method, route, copy.deepcopy(payload)))
        self.rules[identifier] = {**copy.deepcopy(payload), 'id': identifier,
                                  'source_type': 'Repository', 'source': setup.REPOSITORY}
        return copy.deepcopy(self.rules[identifier])


class GitHubRulesetSetupTests(unittest.TestCase):
    def configure(self, client, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return setup.configure(client, **kwargs)

    def test_create_then_repeat_verifies_server_without_duplicate_writes(self):
        client = FakeGitHub()
        with tempfile.TemporaryDirectory() as directory:
            report = self.configure(client, output=Path(directory))
            self.assertTrue(report['verified'])
            self.assertEqual(len(client.writes), 4)
            self.assertFalse(json.loads((Path(directory) / 'result.json').read_text())['token_saved'])
            self.assertEqual(json.loads((Path(directory) / 'before.json').read_text()), [])
        repeated = self.configure(client)
        self.assertEqual(len(client.writes), 4)
        self.assertTrue(all(item['state'] == 'already active' for item in repeated['rulesets']))

    def test_wrong_identity_is_rejected_before_any_write(self):
        client = FakeGitHub()
        client.login = 'another-account'
        with self.assertRaisesRegex(RuntimeError, 'Authenticate as bangwayne'):
            self.configure(client)
        self.assertEqual(client.writes, [])

    def test_server_omitted_default_update_parameter_is_verified_without_duplicate_writes(self):
        class GitHubDefaults(FakeGitHub):
            def request(self, method, route, payload=None):
                response = super().request(method, route, payload)
                if method == 'GET' and '/rulesets/' in route:
                    for rule in response['rules']:
                        if rule['type'] == 'update':
                            rule.pop('parameters', None)
                return response

        client = GitHubDefaults()
        # Resume the first active rule left by a prior interrupted setup.
        payload = json.loads((setup.TEMPLATES / setup.FILES[0]).read_text())
        client.request('POST', '/repos/' + setup.REPOSITORY + '/rulesets', payload)
        report = self.configure(client)
        self.assertTrue(report['verified'])
        self.assertEqual(report['rulesets'][0]['state'], 'already active')
        self.assertEqual(len(client.writes), 4)
        self.configure(client)
        self.assertEqual(len(client.writes), 4)

    def test_omitted_update_default_cannot_satisfy_enabled_fetch_and_merge(self):
        enabled = {'type': 'update', 'parameters': {'update_allows_fetch_and_merge': True}}
        self.assertFalse(setup.same_configuration({'type': 'update'}, enabled))
        disabled = {'type': 'update', 'parameters': {'update_allows_fetch_and_merge': False}}
        self.assertFalse(setup.same_configuration(enabled, disabled))

    def test_missing_bypass_permissions_still_fails_verification_and_saves_details(self):
        class HiddenActors(FakeGitHub):
            def request(self, method, route, payload=None):
                response = super().request(method, route, payload)
                if method == 'GET' and '/rulesets/' in route:
                    response.pop('bypass_actors')
                return response

        client = HiddenActors()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'Server verification failed'):
                self.configure(client, output=output)
            detail = json.loads((output / 'verify-1.json').read_text())
            self.assertIn('bypass_actors', detail['expected'])
            self.assertNotIn('bypass_actors', detail['actual'])

    def test_dry_run_does_not_mutate_repository(self):
        client = FakeGitHub()
        report = self.configure(client, dry_run=True)
        self.assertFalse(report['verified'])
        self.assertEqual(client.writes, [])
        self.assertEqual(len(report['rulesets']), 4)

    def test_extra_bypass_permission_is_corrected_and_unrelated_rule_is_preserved(self):
        client = FakeGitHub()
        self.configure(client)
        client.rules[1]['bypass_actors'].append({'actor_id': 1, 'actor_type': 'User', 'bypass_mode': 'always'})
        unrelated = {'id': 5, 'name': 'Unrelated feature protection', 'source_type': 'Repository',
                     'source': setup.REPOSITORY, 'conditions': {'ref_name': {'include': ['refs/heads/feature'], 'exclude': []}},
                     'rules': [{'type': 'non_fast_forward'}]}
        client.rules[5] = copy.deepcopy(unrelated)
        self.configure(client)
        self.assertEqual(len(client.writes), 5)
        self.assertEqual(client.writes[-1][0], 'PUT')
        self.assertEqual(len(client.rules[1]['bypass_actors']), 1)
        self.assertEqual(client.rules[5], unrelated)

    def test_duplicate_names_are_rejected_before_changes(self):
        client = FakeGitHub()
        self.configure(client)
        client.rules[5] = {**copy.deepcopy(client.rules[1]), 'id': 5}
        before = len(client.writes)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate rulesets'):
            self.configure(client)
        self.assertEqual(len(client.writes), before)

    def test_http_error_does_not_expose_token(self):
        token = 'test-secret-value'
        body = io.BytesIO(json.dumps({'message': 'Rejected ' + token}).encode())
        error = urllib.error.HTTPError('https://api.github.com/user', 401, 'Unauthorized', {}, body)
        with patch.object(setup.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                setup.GitHub(token).request('GET', '/user')
        self.assertNotIn(token, str(caught.exception))
        self.assertIn('[redacted]', str(caught.exception))

    def test_noninteractive_token_input_is_refused(self):
        with patch('sys.argv', ['apply_github_rulesets.py']), patch('sys.stdin.isatty', return_value=False), \
                patch.object(setup.getpass, 'getpass') as prompt, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                setup.main()
        self.assertEqual(caught.exception.code, 2)
        prompt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
