"""加载部署场景模板"""

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Profile:
    name: str
    description: str
    var_axes: dict  # key -> list of workload var_axes dicts or scalar lists


def load_profile(path: Path) -> Profile:
    """加载一个 profile YAML"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Profile(
        name=data["name"],
        description=data.get("description", ""),
        var_axes=data.get("var_axes", {}),
    )


def load_all_profiles(profiles_dir: Path) -> dict[str, Profile]:
    """加载目录下所有 profile，返回 name -> Profile"""
    profiles = {}
    for p in sorted(profiles_dir.glob("*.yaml")):
        try:
            profile = load_profile(p)
            profiles[profile.name] = profile
        except Exception as e:
            print(f"Warning: failed to load profile {p.name}: {e}")
    return profiles
