"""Thin wrappers around third-party models.  Nothing here is imported by the CPU parts of the package; each wrapper
imports its model lazily, so only the models actually used need to be installed (see README, "Installation")."""
