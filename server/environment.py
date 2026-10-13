"""Translate configuration from earlier workstation installations."""
import os


def apply_legacy_environment(environment=None):
    """Use current settings first, then earlier workstation/model aliases."""
    environment = os.environ if environment is None else environment
    for name, value in list(environment.items()):
        if name.startswith('KPRISM_'):
            environment.setdefault('CARDIAC_PRISM_' + name[len('KPRISM_'):], value)
    for option in ('CONFIG', 'CHECKPOINT', 'DEVICE'):
        old_name = 'VERSE_' + option
        if old_name in environment:
            environment.setdefault('CARDIAC_PRISM_' + option, environment[old_name])
