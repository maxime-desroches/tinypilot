from cereal import car
from openpilot.selfdrive.car.chrysler.values import CAR

Ecu = car.CarParams.Ecu

FW_VERSIONS = {
  CAR.CHRYSLER_PACIFICA_2017_HYBRID: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68239262AH',
      b'68239262AI',
      b'68239262AJ',
      b'68239263AH',
      b'68239263AJ',
    ],
    (Ecu.srs, 0x744, None): [
      b'68238840AH',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'68226356AI',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68288309AC',
      b'68288309AD',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'68277480AV ',
      b'68277480AX ',
      b'68277480AZ ',
    ],
    (Ecu.hybrid, 0x7e2, None): [
      b'05190175BF',
      b'05190175BH',
      b'05190226AK',
    ],
  },
  CAR.CHRYSLER_PACIFICA_2018: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68227902AF',
      b'68227902AG',
      b'68227902AH',
      b'68227905AG',
      b'68360252AC',
    ],
    (Ecu.srs, 0x744, None): [
      b'68211617AF',
      b'68211617AG',
      b'68358974AC',
      b'68405937AA',
    ],
    (Ecu.abs, 0x747, None): [
      b'68222747AG',
      b'68330876AA',
      b'68330876AB',
      b'68352227AA',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672758AA',
      b'68226356AF',
      b'68226356AH',
      b'68226356AI',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68288891AE',
      b'68378884AA',
      b'68525338AA',
      b'68525338AB',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'68267018AO ',
      b'68267020AJ ',
      b'68303534AG ',
      b'68303534AJ ',
      b'68340762AD ',
      b'68340764AD ',
      b'68352652AE ',
      b'68352654AE ',
      b'68366851AH ',
      b'68366853AE ',
      b'68372861AF ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68277370AJ',
      b'68277370AM',
      b'68277372AD',
      b'68277372AE',
      b'68277372AN',
      b'68277374AA',
      b'68277374AB',
      b'68277374AD',
      b'68277374AN',
      b'68367471AC',
      b'68380571AB',
    ],
  },
  CAR.CHRYSLER_PACIFICA_2020: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68405327AC',
      b'68436233AB',
      b'68436233AC',
      b'68436234AB',
      b'68436250AE',
      b'68529067AA',
      b'68594993AB',
      b'68594994AB',
    ],
    (Ecu.srs, 0x744, None): [
      b'68405565AB',
      b'68405565AC',
      b'68444299AC',
      b'68480707AC',
      b'68480708AC',
      b'68526663AB',
    ],
    (Ecu.abs, 0x747, None): [
      b'68397394AA',
      b'68433480AB',
      b'68453575AF',
      b'68577676AA',
      b'68593395AA',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672758AA',
      b'04672758AB',
      b'68417813AF',
      b'68540436AA',
      b'68540436AC',
      b'68540436AD',
      b'68598670AB',
      b'68598670AC',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68416742AA',
      b'68460393AA',
      b'68460393AB',
      b'68494461AB',
      b'68494461AC',
      b'68524936AA',
      b'68524936AB',
      b'68525338AB',
      b'68594337AB',
      b'68594340AB',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'68413871AD ',
      b'68413871AE ',
      b'68413871AH ',
      b'68413871AI ',
      b'68413873AH ',
      b'68413873AI ',
      b'68443120AE ',
      b'68443123AC ',
      b'68443125AC ',
      b'68496647AI ',
      b'68496647AJ ',
      b'68496650AH ',
      b'68496650AI ',
      b'68496652AH ',
      b'68526752AD ',
      b'68526752AE ',
      b'68526754AE ',
      b'68536264AE ',
      b'68700304AB ',
      b'68700306AB ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68414271AC',
      b'68414271AD',
      b'68414275AC',
      b'68414275AD',
      b'68443154AB',
      b'68443155AC',
      b'68443158AB',
      b'68501050AD',
      b'68501051AD',
      b'68501055AD',
      b'68527221AB',
      b'68527223AB',
      b'68586231AD',
      b'68586233AD',
    ],
  },
  CAR.CHRYSLER_PACIFICA_2018_HYBRID: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68358439AE',
      b'68358439AG',
    ],
    (Ecu.srs, 0x744, None): [
      b'68358990AC',
      b'68405939AA',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672758AA',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68288309AD',
      b'68525339AA',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'68366580AI ',
      b'68366580AK ',
      b'68366580AM ',
    ],
    (Ecu.hybrid, 0x7e2, None): [
      b'05190226AI',
      b'05190226AK',
      b'05190226AM',
    ],
  },
  CAR.CHRYSLER_PACIFICA_2019_HYBRID: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68405292AC',
      b'68434956AC',
      b'68434956AD',
      b'68434960AE',
      b'68434960AF',
      b'68529064AB',
      b'68594990AB',
    ],
    (Ecu.srs, 0x744, None): [
      b'68405567AB',
      b'68405567AC',
      b'68453076AD',
      b'68480710AC',
      b'68526665AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672758AB',
      b'68417813AF',
      b'68540436AA',
      b'68540436AB',
      b'68540436AC',
      b'68540436AD',
      b'68598670AB',
      b'68598670AC',
      b'68645752AA',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68416741AA',
      b'68460392AA',
      b'68525339AA',
      b'68525339AB',
      b'68594341AB',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'68416680AE ',
      b'68416680AF ',
      b'68416680AG ',
      b'68444228AD ',
      b'68444228AE ',
      b'68444228AF ',
      b'68499122AD ',
      b'68499122AE ',
      b'68499122AF ',
      b'68526772AD ',
      b'68526772AH ',
      b'68599493AC ',
      b'68657433AA ',
    ],
    (Ecu.hybrid, 0x7e2, None): [
      b'05185116AF',
      b'05185116AJ',
      b'05185116AK',
      b'05190240AP',
      b'05190240AQ',
      b'05190240AR',
      b'05190265AG',
      b'05190265AH',
      b'05190289AE',
      b'68540977AH',
      b'68540977AK',
      b'68597647AE',
      b'68632416AB',
    ],
  },
  CAR.JEEP_GRAND_CHEROKEE: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68243549AG',
      b'68302211AC',
      b'68302212AD',
      b'68302223AC',
      b'68302246AC',
      b'68331511AC',
      b'68331574AC',
      b'68331687AC',
      b'68331690AC',
      b'68340272AD',
    ],
    (Ecu.srs, 0x744, None): [
      b'68309533AA',
      b'68316742AB',
      b'68355363AB',
    ],
    (Ecu.abs, 0x747, None): [
      b'68252642AG',
      b'68306178AD',
      b'68336275AB',
      b'68336276AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672627AB',
      b'68251506AF',
      b'68332015AB',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68276201AG',
      b'68321644AB',
      b'68321644AC',
      b'68321646AC',
      b'68321648AC',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'05035920AE ',
      b'68252272AG ',
      b'68284455AI ',
      b'68284456AI ',
      b'68284477AF ',
      b'68325564AH ',
      b'68325564AI ',
      b'68325565AH ',
      b'68325565AI ',
      b'68325618AD ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'05035517AH',
      b'68253222AF',
      b'68311218AC',
      b'68311223AF',
      b'68311223AG',
      b'68361911AE',
      b'68361911AF',
      b'68361911AH',
      b'68361916AD',
    ],
  },
  CAR.JEEP_GRAND_CHEROKEE_2019: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68402703AB',
      b'68402704AB',
      b'68402708AB',
      b'68402971AD',
      b'68454144AD',
      b'68454145AB',
      b'68454152AB',
      b'68454156AB',
      b'68516650AB',
      b'68516651AB',
      b'68516669AB',
      b'68516671AB',
      b'68516683AB',
    ],
    (Ecu.srs, 0x744, None): [
      b'68355363AB',
      b'68355364AB',
    ],
    (Ecu.abs, 0x747, None): [
      b'68408639AC',
      b'68408639AD',
      b'68499978AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672788AA',
      b'68456722AC',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68417279AA',
      b'68417280AA',
      b'68417281AA',
      b'68453431AA',
      b'68453433AA',
      b'68453435AA',
      b'68499171AA',
      b'68499171AB',
      b'68501183AA',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'05035674AB ',
      b'68412635AG ',
      b'68412660AD ',
      b'68422860AB',
      b'68449435AE ',
      b'68496223AA ',
      b'68504959AD ',
      b'68504959AE ',
      b'68504960AD ',
      b'68504993AC ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'05035707AA',
      b'68419672AC',
      b'68419678AB',
      b'68423905AB',
      b'68449258AC',
      b'68495807AA',
      b'68495807AB',
      b'68503641AC',
      b'68503664AC',
    ],
  },
  CAR.RAM_1500_5TH_GEN: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68294051AG',
      b'68294051AI',
      b'68294052AG',
      b'68294052AH',
      b'68294063AG',
      b'68294063AH',
      b'68294063AI',
      b'68434846AC',
      b'68434847AC',
      b'68434849AC',
      b'68434856AC',
      b'68434858AC',
      b'68434859AC',
      b'68434860AC',
      b'68453483AC',
      b'68453483AD',
      b'68453487AD',
      b'68453491AC',
      b'68453491AD',
      b'68453499AD',
      b'68453503AC',
      b'68453503AD',
      b'68453505AC',
      b'68453505AD',
      b'68453511AC',
      b'68453513AC',
      b'68453513AD',
      b'68453514AD',
      b'68505633AB',
      b'68510277AG',
      b'68510277AH',
      b'68510280AG',
      b'68510282AG',
      b'68510282AH',
      b'68510283AG',
      b'68527346AE',
      b'68527361AD',
      b'68527375AD',
      b'68527381AE',
      b'68527382AE',
      b'68527383AD',
      b'68527383AE',
      b'68527387AE',
      b'68527403AC',
      b'68527403AD',
      b'68546047AF',
      b'68631938AA',
      b'68631939AA',
      b'68631940AA',
      b'68631940AB',
      b'68631942AA',
      b'68631943AB',
    ],
    (Ecu.srs, 0x744, None): [
      b'68428609AB',
      b'68441329AB',
      b'68473844AB',
      b'68490898AA',
      b'68500728AA',
      b'68615033AA',
      b'68615034AA',
    ],
    (Ecu.abs, 0x747, None): [
      b'68292406AG',
      b'68292406AH',
      b'68432418AB',
      b'68432418AC',
      b'68432418AD',
      b'68436004AD',
      b'68436004AE',
      b'68438454AC',
      b'68438454AD',
      b'68438456AE',
      b'68438456AF',
      b'68535469AB',
      b'68535470AC',
      b'68548900AB',
      b'68586307AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672892AB',
      b'04672932AB',
      b'04672932AC',
      b'22DTRHD_AA',
      b'68320950AH',
      b'68320950AI',
      b'68320950AJ',
      b'68320950AL',
      b'68320950AM',
      b'68454268AB',
      b'68475160AE',
      b'68475160AF',
      b'68475160AG',
    ],
    (Ecu.eps, 0x75a, None): [
      b'21590101AA',
      b'21590101AB',
      b'68273275AF',
      b'68273275AG',
      b'68273275AH',
      b'68312176AE',
      b'68312176AG',
      b'68440789AC',
      b'68466110AA',
      b'68466110AB',
      b'68466113AA',
      b'68469901AA',
      b'68469907AA',
      b'68522583AA',
      b'68522583AB',
      b'68522584AA',
      b'68522585AB',
      b'68552788AA',
      b'68552789AA',
      b'68552790AA',
      b'68552791AB',
      b'68552794AA',
      b'68552794AD',
      b'68585106AB',
      b'68585107AB',
      b'68585108AB',
      b'68585109AB',
      b'68585112AB',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'05035699AG ',
      b'05035841AC ',
      b'05035841AD ',
      b'05036026AB ',
      b'05036065AE ',
      b'05036066AE ',
      b'05036193AA ',
      b'05149368AA ',
      b'05149591AD ',
      b'05149591AE ',
      b'05149592AE ',
      b'05149599AE ',
      b'05149600AD ',
      b'05149605AE ',
      b'05149846AA ',
      b'05149848AA ',
      b'05149848AC ',
      b'05190341AD',
      b'68378695AJ ',
      b'68378696AJ ',
      b'68378696AK ',
      b'68378701AI ',
      b'68378702AI ',
      b'68378710AL ',
      b'68378742AI ',
      b'68378742AK ',
      b'68378748AL ',
      b'68378758AM ',
      b'68448163AJ',
      b'68448163AK',
      b'68448163AL',
      b'68448165AG',
      b'68448165AK',
      b'68455111AC ',
      b'68455119AC ',
      b'68455145AC ',
      b'68455145AE ',
      b'68455146AC ',
      b'68467915AC ',
      b'68467916AC ',
      b'68467936AC ',
      b'68500630AD',
      b'68500630AE',
      b'68500631AE',
      b'68502719AC ',
      b'68502722AC ',
      b'68502733AC ',
      b'68502734AF ',
      b'68502740AF ',
      b'68502741AF ',
      b'68502742AC ',
      b'68502742AF ',
      b'68539650AD',
      b'68539650AF',
      b'68539651AD',
      b'68586101AA ',
      b'68586105AB ',
      b'68629919AC ',
      b'68629922AC ',
      b'68629925AC ',
      b'68629926AC ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'05035706AD',
      b'05035842AB',
      b'05036069AA',
      b'05036181AA',
      b'05149536AC',
      b'05149537AC',
      b'05149543AC',
      b'68360078AL',
      b'68360080AL',
      b'68360080AM',
      b'68360081AM',
      b'68360085AJ',
      b'68360085AL',
      b'68360086AH',
      b'68360086AK',
      b'68384328AD',
      b'68384332AD',
      b'68445531AC',
      b'68445533AB',
      b'68445536AB',
      b'68445537AB',
      b'68466081AB',
      b'68466087AB',
      b'68484466AC',
      b'68484467AC',
      b'68484471AC',
      b'68502994AD',
      b'68502996AD',
      b'68520867AE',
      b'68520867AF',
      b'68520870AC',
      b'68540431AB',
      b'68540433AB',
      b'68551676AA',
      b'68629935AB',
      b'68629936AC',
    ],
  },
  CAR.RAM_HD_5TH_GEN: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68361606AH',
      b'68437735AC',
      b'68492693AD',
      b'68525485AB',
      b'68525487AB',
      b'68525498AB',
      b'68528791AF',
      b'68628474AB',
    ],
    (Ecu.srs, 0x744, None): [
      b'68399794AC',
      b'68428503AA',
      b'68428505AA',
      b'68428507AA',
    ],
    (Ecu.abs, 0x747, None): [
      b'68334977AH',
      b'68455481AC',
      b'68504022AA',
      b'68504022AB',
      b'68504022AC',
      b'68530686AB',
      b'68530686AC',
      b'68544596AC',
      b'68641704AA',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672895AB',
      b'04672934AB',
      b'56029827AG',
      b'56029827AH',
      b'68462657AE',
      b'68484694AD',
      b'68484694AE',
      b'68615489AB',
    ],
    (Ecu.eps, 0x761, None): [
      b'68421036AC',
      b'68507906AB',
      b'68534023AC',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'52370131AF',
      b'52370231AF',
      b'52370231AG',
      b'52370491AA',
      b'52370931CT',
      b'52401032AE',
      b'52421132AF',
      b'52421332AF',
      b'68527616AD ',
      b'M2370131MB',
      b'M2421132MB',
    ],
  },
  CAR.DODGE_DURANGO: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68454261AD',
      b'68471535AE',
    ],
    (Ecu.srs, 0x744, None): [
      b'68355362AB',
      b'68492238AD',
    ],
    (Ecu.abs, 0x747, None): [
      b'68408639AD',
      b'68499978AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'68440581AE',
      b'68456722AC',
    ],
    (Ecu.eps, 0x75a, None): [
      b'68453435AA',
      b'68498477AA',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'05035786AE ',
      b'68449476AE ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'05035826AC',
      b'68449265AC',
    ],
  },
}
