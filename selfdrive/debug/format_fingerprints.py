#!/usr/bin/env python3
import jinja2
import os

from cereal import car
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car.interfaces import get_interface_attr

Ecu = car.CarParams.Ecu

CARS = get_interface_attr('CAR')
FW_VERSIONS = get_interface_attr('FW_VERSIONS')
FINGERPRINTS = get_interface_attr('FINGERPRINTS')
PLATFORM_TO_PYTHON_CAR_NAME = {brand: {car.value: car.name for car in CARS[brand]} for brand in CARS}
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

FINGERPRINTS_PY_TEMPLATE = jinja2.Template("""
{%- if FINGERPRINTS[brand] %}
# ruff: noqa: E501
{% endif %}
{% if FW_VERSIONS[brand] %}
from cereal import car
{% endif %}
from openpilot.selfdrive.car.{{brand}}.values import CAR
{% if FW_VERSIONS[brand] %}

Ecu = car.CarParams.Ecu
{% endif %}

{% if FINGERPRINTS[brand] %}
FINGERPRINTS = {
{% for car, fingerprints in FINGERPRINTS[brand].items() %}
  CAR.{{PLATFORM_TO_PYTHON_CAR_NAME[brand][car]}}: [
{% for fingerprint in fingerprints %}
  {
    {% for key, value in fingerprint.items() %}{{key}}: {{value}}{% if not loop.last %}, {% endif %}{% endfor %}

  }{% if loop.last %}]{% endif %},
{% endfor %}
{% endfor %}
}
{% endif %}

{% if FW_VERSIONS[brand] %}
FW_VERSIONS = {
{% for car, _ in FW_VERSIONS[brand].items() %}
  CAR.{{PLATFORM_TO_PYTHON_CAR_NAME[brand][car]}}: {
{% for key, fw_versions in FW_VERSIONS[brand][car].items() %}
    (Ecu.{{ECU_NAME[key[0]]}}, 0x{{"%0x" | format(key[1] | int)}}, \
{% if key[2] %}0x{{"%0x" | format(key[2] | int)}}{% else %}{{key[2]}}{% endif %}): [
  {% for fw_version in fw_versions %}
    {{fw_version}},
  {% endfor %}
  ],
{% endfor %}
  },
{% endfor %}
}
{% endif %}
""", trim_blocks=True)


def format_brand_fw_versions(brand):
  with open(os.path.join(BASEDIR, f"selfdrive/car/{brand}/fingerprints.py"), "w") as f:
    f.write(FINGERPRINTS_PY_TEMPLATE.render(brand=brand, ECU_NAME=ECU_NAME, PLATFORM_TO_PYTHON_CAR_NAME=PLATFORM_TO_PYTHON_CAR_NAME,
                                            FINGERPRINTS=FINGERPRINTS, FW_VERSIONS=FW_VERSIONS))


if __name__ == "__main__":
  for brand in FW_VERSIONS.keys():
    format_brand_fw_versions(brand)
