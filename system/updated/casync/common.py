import dataclasses
import json
import pathlib
import subprocess

from openpilot.system.version import BUILD_METADATA_FILENAME, BuildMetadata


CASYNC_ARGS = ["--with=symlinks", "--with=permissions", "--compression=xz"]
CASYNC_FILES = [BUILD_METADATA_FILENAME, ".caexclude"]


def run(cmd):
  return subprocess.check_output(cmd)


def get_exclude_set(path) -> set[str]:
  exclude_set = set(CASYNC_FILES)

  for file in path.rglob("*"):
    if file.is_file() or file.is_symlink():

      while file.resolve() != path.resolve():
        exclude_set.add(str(file.relative_to(path)))

        file = file.parent

  return exclude_set


def create_caexclude_file(path: pathlib.Path):
  with open(path / ".caexclude", "w") as f:
    # exclude everything except the paths already in the release
    f.write("*\n")
    f.write(".*\n")

    for file in sorted(get_exclude_set(path)):
      f.write(f"!{file}\n")


def create_build_metadata_file(path: pathlib.Path, build_metadata: BuildMetadata, channel: str):
  with open(path / BUILD_METADATA_FILENAME, "w") as f:
    build_metadata_dict = dataclasses.asdict(build_metadata)
    build_metadata_dict["channel"] = channel
    build_metadata_dict["openpilot"].pop("is_dirty")  # this is determined at runtime
    f.write(json.dumps(build_metadata_dict))


def create_casync_release(target_dir: pathlib.Path, output_dir: pathlib.Path, caibx_name: str):
  tar_file = output_dir / f"{caibx_name}.tar"
  run(["tar", "-cf", str(tar_file), target_dir])
  caibx_file = output_dir / f"{caibx_name}.caibx"
  run(["casync", "make", *CASYNC_ARGS, caibx_file, str(tar_file)])
  tar_file.unlink()
  digest = run(["casync", "digest", *CASYNC_ARGS, target_dir]).decode("utf-8").strip()
  return digest, caibx_file
