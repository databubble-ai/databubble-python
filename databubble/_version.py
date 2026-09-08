# databubble/_version.py
"""Single source of truth for the package version.

Kept in its own module so client.py can stamp a User-Agent without importing
the package root (circular import).
"""

__version__ = "0.8.0"
