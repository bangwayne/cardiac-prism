#!/usr/bin/env python3
"""Apply the repository's branch rules using a one-time, hidden token prompt."""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request


REPOSITORY = 'bangwayne/cardiac-prism'
OWNER_ID = 192292443
ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / '.github' / 'rulesets'
FILES = (
    '01-main-owner-only.json',
    '02-preserve-history.json',
    '03-develop-reviewed-prs.json',
    '04-develop-required-tests.json',
)


class GitHub:
    def __init__(self, token):
        self._token = token

    def request(self, method, route, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            'https://api.github.com' + route, data=data, method=method,
            headers={
                'Authorization': 'Bearer ' + self._token,
                'Accept': 'application/vnd.github+json',
                'Content-Type': 'application/json',
                'X-GitHub-Api-Version': '2026-03-10',
                'User-Agent': 'Cardiac-Prism-ruleset-setup',
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                message = json.load(error).get('message', 'Request rejected')
            except (ValueError, AttributeError):
                message = 'Request rejected'
            message = str(message).replace(self._token, '[redacted]')
            raise RuntimeError(f'GitHub HTTP {error.code}: {message}') from None
        except urllib.error.URLError:
            raise RuntimeError('Cannot connect to api.github.com.') from None


def same_configuration(actual, expected):
    """Ignore response metadata, but never ignore extra actors or rules."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and same_configuration(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            any(same_configuration(item, wanted) for item in actual)
            for wanted in expected
        )
    return actual == expected


def configure(client, dry_run=False, output=None):
    desired = [json.loads((TEMPLATES / file).read_text()) for file in FILES]
    user = client.request('GET', '/user')
    if user.get('login') != 'bangwayne' or user.get('id') != OWNER_ID:
        raise RuntimeError('Authenticate as bangwayne, the configured repository owner.')
    base = '/repos/' + REPOSITORY
    repository = client.request('GET', base)
    if repository.get('full_name') != REPOSITORY or repository.get('default_branch') != 'main':
        raise RuntimeError('Expected bangwayne/cardiac-prism with main as its default branch.')
    if not repository.get('permissions', {}).get('admin'):
        raise RuntimeError('The authenticated account must administer this repository.')
    for branch in ('main', 'develop'):
        client.request('GET', base + '/branches/' + branch)

    summaries = []
    page = 1
    while True:
        items = client.request('GET', base + f'/rulesets?per_page=100&page={page}')
        summaries.extend(items)
        if len(items) < 100:
            break
        page += 1
    existing = {}
    for wanted in desired:
        matches = [item for item in summaries if item.get('name') == wanted['name']]
        if len(matches) > 1:
            raise RuntimeError('Duplicate rulesets named ' + wanted['name'] + '; inspect them in GitHub first.')
        if matches:
            item = matches[0]
            if item.get('source_type') != 'Repository' or item.get('source') != REPOSITORY:
                raise RuntimeError('A matching ruleset is inherited from another source; inspect it first.')
            existing[wanted['name']] = client.request('GET', base + '/rulesets/' + str(item['id']))

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'before.json').write_text(json.dumps(list(existing.values()), indent=2) + '\n')
    results = []
    for wanted in desired:
        previous = existing.get(wanted['name'])
        if previous and same_configuration(previous, wanted):
            state, identifier = 'already active', previous['id']
        elif dry_run:
            state, identifier = 'would update' if previous else 'would create', None
        else:
            method = 'PUT' if previous else 'POST'
            route = base + '/rulesets' + ('/' + str(previous['id']) if previous else '')
            applied = client.request(method, route, wanted)
            identifier = applied['id']
            verified = client.request('GET', base + '/rulesets/' + str(identifier))
            if not same_configuration(verified, wanted):
                raise RuntimeError('Server verification failed for ' + wanted['name'])
            state = 'updated' if previous else 'created'
        results.append({'name': wanted['name'], 'state': state, 'id': identifier})
        print(f'{state}: {wanted["name"]}', flush=True)
    if not dry_run:
        for branch in ('main', 'develop'):
            active = client.request('GET', base + '/rules/branches/' + branch)
            expected_types = {'update', 'deletion', 'non_fast_forward'} if branch == 'main' else {
                'pull_request', 'required_status_checks', 'deletion', 'non_fast_forward'
            }
            if not expected_types.issubset({item['type'] for item in active}):
                raise RuntimeError('Active branch-rule verification failed for ' + branch)
    report = {'repository': REPOSITORY, 'dry_run': dry_run, 'rulesets': results,
              'verified': not dry_run, 'token_saved': False}
    if output is not None:
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Inspect the planned setup without writing rules.')
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error('Run this command in your own terminal; the token requires hidden interactive input.')
    print('Repository: ' + REPOSITORY)
    print('Token: only this repository; Administration: Read and write; short expiration.')
    token = getpass.getpass('Paste the fine-grained GitHub token (hidden): ').strip()
    if not token:
        parser.error('A token is required.')
    output = ROOT / 'test-results' / ('github-rules-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    try:
        configure(GitHub(token), dry_run=args.dry_run, output=output)
    except (RuntimeError, ValueError, OSError) as error:
        print('Setup stopped: ' + str(error).replace(token, '[redacted]'), file=sys.stderr)
        print('Inspect the repository rules page before retrying. Reports: ' + str(output), file=sys.stderr)
        return 1
    print('Reports: ' + str(output))
    print('The token was not saved. Revoke it in GitHub when finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
