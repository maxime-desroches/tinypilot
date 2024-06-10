from cereal import car
from openpilot.selfdrive.car.subaru.values import CAR

Ecu = car.CarParams.Ecu

FW_VERSIONS = {
  CAR.SUBARU_ASCENT: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa5 \x19\x02\x00',
      b'\xa5 !\x02\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x05\xc0\xd0\x00',
      b'\x85\xc0\xd0\x00',
      b'\x95\xc0\xd0\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00d\xb9\x00\x00\x00\x00',
      b'\x00\x00d\xb9\x1f@ \x10',
      b'\x00\x00e@\x00\x00\x00\x00',
      b'\x00\x00e@\x1f@ $',
      b"\x00\x00e~\x1f@ '",
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xbb,\xa0t\x07',
      b'\xd1,\xa0q\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\x00>\xf0\x00\x00',
      b'\x00\xfe\xf7\x00\x00',
      b'\x01\xfe\xf7\x00\x00',
      b'\x01\xfe\xf9\x00\x00',
      b'\x01\xfe\xfa\x00\x00',
    ],
  },
  CAR.SUBARU_ASCENT_2023: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa5 #\x03\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'%\xc0\xd0\x11',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x05!\x08\x1dK\x05!\x08\x01/',
    ],
    (Ecu.engine, 0x7a2, None): [
      b'\xe5,\xa0P\x07',
    ],
    (Ecu.transmission, 0x7a3, None): [
      b'\x04\xfe\xf3\x00\x00',
    ],
  },
  CAR.SUBARU_LEGACY: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa1  \x02\x01',
      b'\xa1  \x02\x02',
      b'\xa1  \x03\x03',
      b'\xa1  \x04\x01',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x9b\xc0\x11\x00',
      b'\x9b\xc0\x11\x02',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00e\x80\x00\x1f@ \x19\x00',
      b'\x00\x00e\x9a\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xde"a0\x07',
      b'\xde,\xa0@\x07',
      b'\xe2"aq\x07',
      b'\xe2,\xa0@\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xa5\xf6\x05@\x00',
      b'\xa5\xfe\xc7@\x00',
      b'\xa7\xf6\x04@\x00',
      b'\xa7\xfe\xc4@\x00',
    ],
  },
  CAR.SUBARU_IMPREZA: {
    (Ecu.abs, 0x7b0, None): [
      b'z\x84\x19\x90\x00',
      b'z\x94\x08\x90\x00',
      b'z\x94\x08\x90\x01',
      b'z\x94\x0c\x90\x00',
      b'z\x94\x0c\x90\x01',
      b'z\x94.\x90\x00',
      b'z\x94?\x90\x00',
      b'z\x9c\x19\x80\x01',
      b'\xa2 \x185\x00',
      b'\xa2 \x193\x00',
      b'\xa2 \x194\x00',
      b'\xa2 \x19`\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'z\xc0\x00\x00',
      b'z\xc0\x04\x00',
      b'z\xc0\x08\x00',
      b'z\xc0\n\x00',
      b'z\xc0\x0c\x00',
      b'\x8a\xc0\x00\x00',
      b'\x8a\xc0\x10\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00c\xf4\x00\x00\x00\x00',
      b'\x00\x00c\xf4\x1f@ \x07',
      b'\x00\x00d)\x00\x00\x00\x00',
      b'\x00\x00d)\x1f@ \x07',
      b'\x00\x00dd\x00\x00\x00\x00',
      b'\x00\x00dd\x1f@ \x0e',
      b'\x00\x00d\xb5\x1f@ \x0e',
      b'\x00\x00d\xdc\x00\x00\x00\x00',
      b'\x00\x00d\xdc\x1f@ \x0e',
      b'\x00\x00e\x02\x1f@ \x14',
      b'\x00\x00e\x1c\x00\x00\x00\x00',
      b'\x00\x00e\x1c\x1f@ \x14',
      b'\x00\x00e+\x00\x00\x00\x00',
      b'\x00\x00e+\x1f@ \x14',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xaa\x00Bu\x07',
      b'\xaa\x01bt\x07',
      b'\xaa!`u\x07',
      b'\xaa!au\x07',
      b'\xaa!av\x07',
      b'\xaa!aw\x07',
      b'\xaa!dq\x07',
      b'\xaa!ds\x07',
      b'\xaa!dt\x07',
      b'\xaaafs\x07',
      b'\xbe!as\x07',
      b'\xbe!at\x07',
      b'\xbeacr\x07',
      b'\xc5!`r\x07',
      b'\xc5!`s\x07',
      b'\xc5!ap\x07',
      b'\xc5!ar\x07',
      b'\xc5!as\x07',
      b'\xc5!dr\x07',
      b'\xc5!ds\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xe3\xd0\x081\x00',
      b'\xe3\xd5\x161\x00',
      b'\xe3\xe5F1\x00',
      b'\xe3\xf5\x06\x00\x00',
      b'\xe3\xf5\x07\x00\x00',
      b'\xe3\xf5C\x00\x00',
      b'\xe3\xf5F\x00\x00',
      b'\xe3\xf5G\x00\x00',
      b'\xe4\xe5\x021\x00',
      b'\xe4\xe5\x061\x00',
      b'\xe4\xf5\x02\x00\x00',
      b'\xe4\xf5\x07\x00\x00',
      b'\xe5\xf5\x04\x00\x00',
      b'\xe5\xf5$\x00\x00',
      b'\xe5\xf5B\x00\x00',
    ],
  },
  CAR.SUBARU_IMPREZA_2020: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa2 \x193\x00',
      b'\xa2 \x194\x00',
      b'\xa2  `\x00',
      b'\xa2 !3\x00',
      b'\xa2 !6\x00',
      b'\xa2 !`\x00',
      b'\xa2 !i\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\n\xc0\x04\x00',
      b'\n\xc0\x04\x01',
      b'\x9a\xc0\x00\x00',
      b'\x9a\xc0\x04\x00',
      b'\x9a\xc0\n\x01',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00eb\x1f@ "',
      b'\x00\x00eq\x00\x00\x00\x00',
      b'\x00\x00eq\x1f@ "',
      b'\x00\x00e\x8f\x00\x00\x00\x00',
      b'\x00\x00e\x8f\x1f@ )',
      b'\x00\x00e\x92\x00\x00\x00\x00',
      b'\x00\x00e\xa4\x00\x00\x00\x00',
      b'\x00\x00e\xa4\x1f@ (',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xca!`0\x07',
      b'\xca!`p\x07',
      b'\xca!ap\x07',
      b'\xca!f@\x07',
      b'\xca!fp\x07',
      b'\xcaacp\x07',
      b'\xcc!`p\x07',
      b'\xcc!fp\x07',
      b'\xcc"f0\x07',
      b'\xe6!`@\x07',
      b'\xe6!fp\x07',
      b'\xe6"f0\x07',
      b'\xe6"fp\x07',
      b'\xf3"f@\x07',
      b'\xf3"fp\x07',
      b'\xf3"fr\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xe6\x15\x042\x00',
      b'\xe6\xf5\x04\x00\x00',
      b'\xe6\xf5$\x00\x00',
      b'\xe6\xf5D0\x00',
      b'\xe7\xf5\x04\x00\x00',
      b'\xe7\xf5D0\x00',
      b'\xe7\xf6B0\x00',
      b'\xe9\xf5"\x00\x00',
      b'\xe9\xf5B0\x00',
      b'\xe9\xf6B0\x00',
      b'\xe9\xf6F0\x00',
    ],
  },
  CAR.SUBARU_CROSSTREK_HYBRID: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa2 \x19e\x01',
      b'\xa2 !e\x01',
    ],
    (Ecu.eps, 0x746, None): [
      b'\n\xc2\x01\x00',
      b'\x9a\xc2\x01\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00el\x1f@ #',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xd7!`@\x07',
      b'\xd7!`p\x07',
      b'\xf4!`0\x07',
    ],
  },
  CAR.SUBARU_FORESTER: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa3 \x18\x14\x00',
      b'\xa3 \x18&\x00',
      b'\xa3 \x19\x14\x00',
      b'\xa3 \x19&\x00',
      b'\xa3 \x19h\x00',
      b'\xa3  \x14\x00',
      b'\xa3  \x14\x01',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x8d\xc0\x00\x00',
      b'\x8d\xc0\x04\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00e!\x00\x00\x00\x00',
      b'\x00\x00e!\x1f@ \x11',
      b'\x00\x00e^\x00\x00\x00\x00',
      b'\x00\x00e^\x1f@ !',
      b'\x00\x00e`\x00\x00\x00\x00',
      b'\x00\x00e`\x1f@  ',
      b'\x00\x00e\x97\x00\x00\x00\x00',
      b'\x00\x00e\x97\x1f@ 0',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xb6"`A\x07',
      b'\xb6\xa2`A\x07',
      b'\xcb"`@\x07',
      b'\xcb"`p\x07',
      b'\xcf"`0\x07',
      b'\xcf"`p\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\x1a\xe6B1\x00',
      b'\x1a\xe6F1\x00',
      b'\x1a\xf6B0\x00',
      b'\x1a\xf6B`\x00',
      b'\x1a\xf6F`\x00',
      b'\x1a\xf6b0\x00',
      b'\x1a\xf6b`\x00',
      b'\x1a\xf6f`\x00',
    ],
  },
  CAR.SUBARU_FORESTER_HYBRID: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa3 \x19T\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x8d\xc2\x00\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00eY\x1f@ !',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xd2\xa1`r\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\x1b\xa7@a\x00',
    ],
  },
  CAR.SUBARU_FORESTER_PREGLOBAL: {
    (Ecu.abs, 0x7b0, None): [
      b'm\x97\x14@',
      b'}\x97\x14@',
    ],
    (Ecu.eps, 0x746, None): [
      b'm\xc0\x10\x00',
      b'}\xc0\x10\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00c\xe9\x00\x00\x00\x00',
      b'\x00\x00c\xe9\x1f@ \x03',
      b'\x00\x00d5\x1f@ \t',
      b'\x00\x00d\xd3\x1f@ \t',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xa7"@0\x07',
      b'\xa7"@p\x07',
      b'\xa7)\xa0q\x07',
      b'\xba"@@\x07',
      b'\xba"@p\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\x1a\xf6F`\x00',
      b'\xda\xf2`\x80\x00',
      b'\xda\xf2`p\x00',
      b'\xda\xfd\xe0\x80\x00',
      b'\xdc\xf2@`\x00',
      b'\xdc\xf2``\x00',
      b'\xdc\xf2`\x80\x00',
      b'\xdc\xf2`\x81\x00',
    ],
  },
  CAR.SUBARU_LEGACY_PREGLOBAL: {
    (Ecu.abs, 0x7b0, None): [
      b'[\x97D\x00',
      b'[\xba\xc4\x03',
      b'k\x97D\x00',
      b'k\x9aD\x00',
      b'{\x97D\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'K\xb0\x00\x01',
      b'[\xb0\x00\x01',
      b'k\xb0\x00\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00c\x94\x1f@\x10\x08',
      b'\x00\x00c\xb7\x1f@\x10\x16',
      b'\x00\x00c\xec\x1f@ \x04',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xa0"@q\x07',
      b'\xa0+@p\x07',
      b'\xab*@r\x07',
      b'\xab+@p\x07',
      b'\xb4"@0\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xbd\xf2\x00`\x00',
      b'\xbe\xf2\x00p\x00',
      b'\xbe\xfb\xc0p\x00',
      b'\xbf\xf2\x00\x80\x00',
      b'\xbf\xfb\xc0\x80\x00',
    ],
  },
  CAR.SUBARU_OUTBACK_PREGLOBAL: {
    (Ecu.abs, 0x7b0, None): [
      b'[\xba\xac\x03',
      b'[\xf7\xac\x00',
      b'[\xf7\xac\x03',
      b'[\xf7\xbc\x03',
      b'k\x97\xac\x00',
      b'k\x9a\xac\x00',
      b'{\x97\xac\x00',
      b'{\x9a\xac\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'K\xb0\x00\x00',
      b'K\xb0\x00\x02',
      b'[\xb0\x00\x00',
      b'k\xb0\x00\x00',
      b'{\xb0\x00\x01',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00c\x90\x1f@\x10\x0e',
      b'\x00\x00c\x94\x00\x00\x00\x00',
      b'\x00\x00c\x94\x1f@\x10\x08',
      b'\x00\x00c\xb7\x1f@\x10\x16',
      b'\x00\x00c\xd1\x1f@\x10\x17',
      b'\x00\x00c\xec\x1f@ \x04',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xa0"@\x80\x07',
      b'\xa0*@q\x07',
      b'\xa0*@u\x07',
      b'\xa0+@@\x07',
      b'\xa0bAq\x07',
      b'\xab"@@\x07',
      b'\xab"@s\x07',
      b'\xab*@@\x07',
      b'\xab+@@\x07',
      b'\xb4"@0\x07',
      b'\xb4"@p\x07',
      b'\xb4"@r\x07',
      b'\xb4+@p\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xbd\xf2@`\x00',
      b'\xbd\xf2@\x81\x00',
      b'\xbd\xfb\xe0\x80\x00',
      b'\xbe\xf2@p\x00',
      b'\xbe\xf2@\x80\x00',
      b'\xbe\xfb\xe0p\x00',
      b'\xbf\xe2@\x80\x00',
      b'\xbf\xf2@\x80\x00',
      b'\xbf\xfb\xe0b\x00',
    ],
  },
  CAR.SUBARU_OUTBACK_PREGLOBAL_2018: {
    (Ecu.abs, 0x7b0, None): [
      b'\x8b\x97\xac\x00',
      b'\x8b\x97\xbc\x00',
      b'\x8b\x99\xac\x00',
      b'\x8b\x9a\xac\x00',
      b'\x9b\x97\xac\x00',
      b'\x9b\x97\xbe\x10',
      b'\x9b\x9a\xac\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'{\xb0\x00\x00',
      b'{\xb0\x00\x01',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00df\x1f@ \n',
      b'\x00\x00d\x95\x00\x00\x00\x00',
      b'\x00\x00d\x95\x1f@ \x0f',
      b'\x00\x00d\xfe\x00\x00\x00\x00',
      b'\x00\x00d\xfe\x1f@ \x15',
      b'\x00\x00e\x19\x1f@ \x15',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xb5"@P\x07',
      b'\xb5"@p\x07',
      b'\xb5+@@\x07',
      b'\xb5b@1\x07',
      b'\xb5q\xe0@\x07',
      b'\xc4"@0\x07',
      b'\xc4+@0\x07',
      b'\xc4b@p\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xbb\xf2@`\x00',
      b'\xbb\xfb\xe0`\x00',
      b'\xbc\xaf\xe0`\x00',
      b'\xbc\xe2@\x80\x00',
      b'\xbc\xf2@\x80\x00',
      b'\xbc\xf2@\x81\x00',
      b'\xbc\xfb\xe0`\x00',
      b'\xbc\xfb\xe0\x80\x00',
    ],
  },
  CAR.SUBARU_OUTBACK: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa1  \x06\x00',
      b'\xa1  \x06\x01',
      b'\xa1  \x06\x02',
      b'\xa1  \x07\x00',
      b'\xa1  \x07\x02',
      b'\xa1  \x07\x03',
      b'\xa1  \x08\x00',
      b'\xa1  \x08\x01',
      b'\xa1  \x08\x02',
      b'\xa1 "\t\x00',
      b'\xa1 "\t\x01',
    ],
    (Ecu.eps, 0x746, None): [
      b'\x1b\xc0\x10\x00',
      b'\x9b\xc0\x10\x00',
      b'\x9b\xc0\x10\x02',
      b'\x9b\xc0 \x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x00\x00eJ\x00\x00\x00\x00\x00\x00',
      b'\x00\x00eJ\x00\x1f@ \x19\x00',
      b'\x00\x00e\x80\x00\x1f@ \x19\x00',
      b'\x00\x00e\x9a\x00\x00\x00\x00\x00\x00',
      b'\x00\x00e\x9a\x00\x1f@ 1\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xbc"`@\x07',
      b'\xbc"`q\x07',
      b'\xbc,\xa0q\x07',
      b'\xbc,\xa0u\x07',
      b'\xde"`0\x07',
      b'\xde,\xa0@\x07',
      b'\xe2"`0\x07',
      b'\xe2"`p\x07',
      b'\xe2"`q\x07',
      b'\xe3,\xa0@\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\xa5\xf6D@\x00',
      b'\xa5\xfe\xf6@\x00',
      b'\xa5\xfe\xf7@\x00',
      b'\xa5\xfe\xf8@\x00',
      b'\xa7\x8e\xf40\x00',
      b'\xa7\xf6D@\x00',
      b'\xa7\xfe\xf4@\x00',
    ],
  },
  CAR.SUBARU_FORESTER_2022: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa3 !v\x00',
      b'\xa3 !x\x00',
      b'\xa3 "v\x00',
      b'\xa3 "x\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'-\xc0\x040',
      b'-\xc0%0',
      b'=\xc0%\x02',
      b'=\xc04\x02',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\x04!\x01\x1eD\x07!\x00\x04,',
      b'\x04!\x08\x01.\x07!\x08\x022',
      b'\r!\x08\x017\n!\x08\x003',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'\xd5"`0\x07',
      b'\xd5"a0\x07',
      b'\xf1"`q\x07',
      b'\xf1"aq\x07',
      b'\xfa"ap\x07',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'\x1d\x86B0\x00',
      b'\x1d\xf6B0\x00',
      b'\x1e\x86B0\x00',
      b'\x1e\x86F0\x00',
      b'\x1e\xf6D0\x00',
    ],
  },
  CAR.SUBARU_OUTBACK_2023: {
    (Ecu.abs, 0x7b0, None): [
      b'\xa1 #\x14\x00',
      b'\xa1 #\x17\x00',
    ],
    (Ecu.eps, 0x746, None): [
      b'+\xc0\x10\x11\x00',
      b'+\xc0\x12\x11\x00',
    ],
    (Ecu.fwdCamera, 0x787, None): [
      b'\t!\x08\x046\x05!\x08\x01/',
    ],
    (Ecu.engine, 0x7a2, None): [
      b'\xed,\xa0q\x07',
      b'\xed,\xa2q\x07',
    ],
    (Ecu.transmission, 0x7a3, None): [
      b'\xa8\x8e\xf41\x00',
      b'\xa8\xfe\xf41\x00',
    ],
  },
}
