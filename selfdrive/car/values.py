from typing import get_args
from openpilot.selfdrive.car.body.values import CAR as BODY
from openpilot.selfdrive.car.chrysler.values import CAR as CHRYSLER, ChryslerFlags
from openpilot.selfdrive.car.ford.values import CAR as FORD, FordFlags
from openpilot.selfdrive.car.gm.values import CAR as GM
from openpilot.selfdrive.car.honda.values import CAR as HONDA, HondaFlags
from openpilot.selfdrive.car.hyundai.values import CAR as HYUNDAI, HyundaiFlags
from openpilot.selfdrive.car.mazda.values import CAR as MAZDA, MazdaFlags
from openpilot.selfdrive.car.mock.values import CAR as MOCK
from openpilot.selfdrive.car.nissan.values import CAR as NISSAN
from openpilot.selfdrive.car.subaru.values import CAR as SUBARU, SubaruFlags
from openpilot.selfdrive.car.tesla.values import CAR as TESLA
from openpilot.selfdrive.car.toyota.values import CAR as TOYOTA, ToyotaFlags
from openpilot.selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN, VolkswagenFlags

Platform = BODY | CHRYSLER | FORD | GM | HONDA | HYUNDAI | MAZDA | MOCK | NISSAN | SUBARU | TESLA | TOYOTA | VOLKSWAGEN
BRANDS = get_args(Platform)

PLATFORMS: dict[str, Platform] = {str(platform): platform for brand in BRANDS for platform in brand}
FLAGS = [ChryslerFlags, FordFlags, HondaFlags, HyundaiFlags, MazdaFlags, SubaruFlags, ToyotaFlags, VolkswagenFlags]
