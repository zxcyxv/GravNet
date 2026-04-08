import importlib
import inspect

def _resolve_identifier(identifier: str):
    """Split an identifier into module and class parts.

    Accepts both ``module@class`` style identifiers (original format) and bare
    module names. For bare module names, attempts to infer the class name using
    a few reasonable conventions after importing the module.
    """

    if "@" in identifier:
        return identifier.split("@")

    module_path = identifier
    class_name = module_path.split(".")[-1]
    return module_path, class_name


def _import_module_with_fallbacks(module_path: str, prefix: str):
    """Import a module, retrying common casing fallbacks."""

    try:
        return importlib.import_module(prefix + module_path)
    except ModuleNotFoundError as first_err:
        lowered = module_path.lower()
        if lowered != module_path:
            try:
                return importlib.import_module(prefix + lowered)
            except ModuleNotFoundError:
                pass
        # Re-raise the original error so the message is informative
        raise first_err


def _infer_class_from_module(module, class_name: str):
    """Infer a class from a module when the identifier omits ``@``.

    Tries a sequence of common name variants before raising a descriptive
    error. This keeps evaluation resilient to older checkpoints or overrides
    that specify only the module name for evaluators.
    """

    candidates = [
        class_name,
        class_name.capitalize(),
        "".join(part.capitalize() for part in class_name.split("_")),
        class_name.upper(),
    ]

    for candidate in candidates:
        if hasattr(module, candidate):
            return getattr(module, candidate)

    raise ValueError(
        f"Could not infer class from module '{module.__name__}' using candidates {candidates}. "
        "Provide the identifier in 'module@class' format to disambiguate."
    )


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = _resolve_identifier(identifier)

    # Import the module
    module = _import_module_with_fallbacks(module_path, prefix)
    if "@" in identifier:
        return getattr(module, class_name)

    return _infer_class_from_module(module, class_name)


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = _resolve_identifier(identifier)

    module = _import_module_with_fallbacks(module_path, prefix)
    return inspect.getsourcefile(module)
