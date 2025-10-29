from pathlib import Path

import bddl
import git

import omnigibson as og


def git_info(directory):
    repo = git.Repo(directory)
    try:
        branch_name = repo.active_branch.name
    except TypeError:
        branch_name = "[DETACHED]"
    return {
        "directory": str(directory),
        "code_diff": repo.git.diff(None),
        "code_diff_staged": repo.git.diff("--staged"),
        "commit_hash": repo.head.commit.hexsha,
        "branch_name": branch_name,
    }


def project_git_info():
    # Handle case where bddl.__file__ might be None (e.g., with namespace packages or editable installs)
    if bddl.__file__ is not None:
        bddl_path = Path(bddl.__file__).parent.parent
    else:
        # Fallback: use bddl module's __path__ attribute
        bddl_path = Path(bddl.__path__[0])

    return {
        "OmniGibson": git_info(Path(og.root_path).parent),
        "bddl": git_info(bddl_path),
    }
