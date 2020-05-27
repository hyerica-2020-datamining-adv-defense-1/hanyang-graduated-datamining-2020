from .attack_method import attack, AttackMethod
import importlib
import pathlib


for module in pathlib.Path("adversarial/attacks").glob("*.py"):
    if module.name == "__init__.py" or module.name == "attack_method.py":
        continue

    importlib.import_module(f"adversarial.attacks.{module.name.replace('.py', '')}")

