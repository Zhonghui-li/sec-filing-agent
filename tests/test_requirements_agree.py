"""The container must install a version the tests could have run with.

There are two requirements files and only one of them is tested. `requirements.txt` builds the
venv that runs pytest and every eval; `service/requirements.txt` is the only one the image
installs (service/Dockerfile line 14). Nothing made them agree.

On 2026-10-08 the project moved to langchain 1.x — root file and local venv only. Production kept
building langchain-core 0.3.x for three days without a symptom, because no test builds the image.
It surfaced when `.text` shipped: a property in 1.x, a METHOD in 0.3.x, so every live answer
became "TypeError: expected string or bytes-like object, got 'method'".

The rule here is one-directional on purpose. The service file may be STRICTER than the root (it
pins langfuse where the root leaves it open — a narrower runtime is fine), but it may never allow
a version the root forbids, because that version is one nothing has ever tested.
"""
import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def _parse(path):
    """{canonical name: SpecifierSet} for one requirements file. Comments and blanks skipped."""
    out = {}
    for line in path.read_text().splitlines():
        line = re.sub(r"\s+#.*$", "", line).strip()
        if not line or line.startswith(("#", "-")):
            continue
        req = Requirement(line)
        out[canonicalize_name(req.name)] = req.specifier
    return out


def _service_may_install(service: SpecifierSet, root: SpecifierSet) -> bool:
    """Can the service spec admit a version the root spec forbids?

    Exact enough for the shapes these two files use: identical specs agree trivially, and a pin is
    checked against the root range directly. Anything else is required to match verbatim rather
    than guessing at range containment — if that ever becomes a real constraint, make the specs
    equal instead of making this smarter.
    """
    if str(service) == str(root):
        return True
    pinned = [s.version for s in service if s.operator == "=="]
    if len(pinned) == 1 and len(list(service)) == 1:
        return root.contains(pinned[0], prereleases=True)
    return False


def test_shared_dependencies_agree_between_the_two_requirements_files():
    root = _parse(ROOT / "requirements.txt")
    service = _parse(ROOT / "service" / "requirements.txt")
    shared = sorted(set(root) & set(service))
    assert shared, "no shared packages found — one of the files failed to parse"

    bad = [f"  {name}: service wants '{service[name]}', root allows '{root[name]}'"
           for name in shared if not _service_may_install(service[name], root[name])]
    assert not bad, (
        "service/requirements.txt may install a version requirements.txt forbids, so the image "
        "would run code no test has run:\n" + "\n".join(bad) +
        "\nMake the specs identical, or pin the service inside the root's range."
    )


def test_the_guard_catches_the_2026_10_08_divergence():
    """The failure this file exists for. A guard that cannot fail is not a guard."""
    assert not _service_may_install(SpecifierSet("==0.3.17"), SpecifierSet(">=1.0,<2"))
    # and the two shapes that must keep passing
    assert _service_may_install(SpecifierSet(">=1.0,<2"), SpecifierSet(">=1.0,<2"))
    assert _service_may_install(SpecifierSet("==4.7.1"), SpecifierSet(""))      # stricter is fine
