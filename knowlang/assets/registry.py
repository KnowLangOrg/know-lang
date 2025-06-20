

class DomainAssetRegistry:
    """Registry for all domain asset managers."""

    def __init__(self):
        self._registry = {}

    def register(self):
        pass

    def get_manager(self):
        pass

    def list_domains(self):
        """List all registered domains."""
        return list(self._registry.keys())