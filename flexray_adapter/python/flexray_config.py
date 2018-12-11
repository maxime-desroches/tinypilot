# -*- coding: utf-8 -*-
# FlexRay configuration parameters definition, loading, saving, and conversion from python dict to c struct
import struct
from math import ceil, floor

from constants import *

default_fr_config = {
    # Network topology parameters, use naming convention in FLexRay spec
    'nStar': 0,
    'LineLength': 24,
    'pdBDTx': 0.1,
    'pdBDRx': 0.1,
    'pdStarDelay': 0.25,
    # Cluster parameters, use naming convention in FLexRay spec
    'gdMinPropagationDelay': 0.0,
    'gdMaxInitializationError': 0.5,
    'gOffsetCorrectionMax': 12.0,
    'gdMacrotick': 2,
    'gPayloadLengthStatic': 8,
    'gNumberOfStaticSlots': 60,
    'gdStaticSlot': 26,
    'gdActionPointOffset': 6,
    'gNumberOfMinislots': 163,
    'gdMinislot': 6,
    'gdMiniSlotActionPointOffset': 3,
    'gdSymbolWindow': 16,
    'gdNIT': 103,
    'gOffsetCorrectionStart': 2558,
    'gdWakeupRxWindow': 301,
    'gColdStartAttempts': 10,
    'gListenNoise': 2,
    'gMaxWithoutClockCorrectionFatal': 14,
    'gMaxWithoutClockCorrectionPassive': 10,
    'gNetworkManagementVectorLength': 2,
    'gSyncFrameIDCountMax': 5,
    'gdCasRxLowMax': 91,
    'gdDynamicSlotIdlePhase': 1,
    'gdTSSTransmitter': 11,
    'gdWakeupSymbolRxIdle': 59,
    'gdWakeupSymbolRxLow': 55,
    'gdWakeupSymbolTxActive': 60,
    'gdWakeupSymbolTxIdle': 180,
    # Node parameters, use naming convention in FLexRay spec
    'pChannels': 0,
    'pWakeupChannel': 0,
    'pWakeupPattern': 63,
    'pPayloadLengthDynMax': 8,
    'pMicroPerCycle': 212800,
    'pdListenTimeout': 426880,
    'pRateCorrectionOut': 640,
    'pKeySlotId': 1,
    'pKeySlotOnlyEnabled': 0,
    'pKeySlotUsedForStartup': 1,
    'pKeySlotUsedForSync': 1,
    'pLatestTx': 157,
    'pOffsetCorrectionOut': 640,
    'pdAcceptedStartupRange': 110,
    'pAllowPassiveToActive': 0,
    'pClusterDriftDamping': 1,
    'pDecodingCorrection': 56,
    'pDelayCompensationA': 0,
    'pDelayCompensationB': 0,
    'pMacroInitialOffsetA': 7,
    'pMacroInitialOffsetB': 7,
    'pMicroInitialOffsetA': 24,
    'pMicroInitialOffsetB': 24,
    'pAllowHaltDueToClock': 1,
    'CLOCK_SRC': 0,         # Protocol engine clock source
    'BIT_RATE': 0,          # Bit rate
    'SCM_EN': 0,            # Single channel mode enabled
    'LOG_STATUS_DATA': 1,   # Log protocol status data defined in FlexRay spec 9.3.1.3
    'FIFOA_EN': 0,          # Receive FIFO for channel A enabled
    'FIFOA_Depth': 0,
    'FIFOA_MIAFV': 0,
    'FIFOA_MIAFM': 0,
    'FIFOA_F0_EN': 0,
    'FIFOA_F0_MODE': 0,
    'FIFOA_F0_SID_LOWER': 0,
    'FIFOA_F0_SID_UPPER': 0,
    'FIFOA_F1_EN': 0,
    'FIFOA_F1_MODE': 0,
    'FIFOA_F1_SID_LOWER': 0,
    'FIFOA_F1_SID_UPPER': 0,
    'FIFOA_F2_EN': 0,
    'FIFOA_F2_MODE': 0,
    'FIFOA_F2_SID_LOWER': 0,
    'FIFOA_F2_SID_UPPER': 0,
    'FIFOA_F3_EN': 0,
    'FIFOA_F3_MODE': 0,
    'FIFOA_F3_SID_LOWER': 0,
    'FIFOA_F3_SID_UPPER': 0,
    'FIFOB_EN': 0,          # Receive FIFO for channel B enabled
    'FIFOB_Depth': 0,
    'FIFOB_MIAFV': 0,
    'FIFOB_MIAFM': 0,
    'FIFOB_F0_EN': 0,
    'FIFOB_F0_MODE': 0,
    'FIFOB_F0_SID_LOWER': 0,
    'FIFOB_F0_SID_UPPER': 0,
    'FIFOB_F1_EN': 0,
    'FIFOB_F1_MODE': 0,
    'FIFOB_F1_SID_LOWER': 0,
    'FIFOB_F1_SID_UPPER': 0,
    'FIFOB_F2_EN': 0,
    'FIFOB_F2_MODE': 0,
    'FIFOB_F2_SID_LOWER': 0,
    'FIFOB_F2_SID_UPPER': 0,
    'FIFOB_F3_EN': 0,
    'FIFOB_F3_MODE': 0,
    'FIFOB_F3_SID_LOWER': 0,
    'FIFOB_F3_SID_UPPER': 0,
    'RxMsgBufs': [                  # Receive frames configuration
        {
            'FrameId': 2,           # The slot id we are listening on
            'Channels': 1,          # The channel we are listening on
            'CCF_EN': 0,            # Cycle counter filter enabled
            'CCF_VAL': 0,           # Cycle counter filter value
            'CCF_MASK': 0,          # Cycle counter filter mask
        },
    ],
    'TxMsgBufs': [                  # Transmit frames configuration
        {
            'FrameId': 1,           # The id of the slot which frame will be send on
            'PayloadLenMax': 8,     # Max payload length in words, for dynamic slot and DynPayloadLen=1 only
            'Channels': 1,          # The channel which frame will be send to
            'CCF_EN': 0,            # Cycle counter filter enabled
            'CCF_VAL': 0,           # Cycle counter filter value
            'CCF_MASK': 0,          # Cycle counter filter mask
            'DynPayloadLen': 0,     # Dynamic payload length enabled, for dynamic slot only
            'PPI': 0,               # Payload preempt flag
        },
    ]
}


def config_to_c_struct(config):
    r = b''
    field_names = ['gdMacrotick', 'gPayloadLengthStatic', 'gNumberOfStaticSlots', 'gdStaticSlot', 'gdActionPointOffset',
                   'gNumberOfMinislots', 'gdMinislot', 'gdMiniSlotActionPointOffset', 'gdSymbolWindow', 'gdNIT',
                   'gOffsetCorrectionStart', 'gdWakeupRxWindow', 'gColdStartAttempts', 'gListenNoise',
                   'gMaxWithoutClockCorrectionFatal', 'gMaxWithoutClockCorrectionPassive', 'gNetworkManagementVectorLength',
                   'gSyncFrameIDCountMax', 'gdCasRxLowMax', 'gdDynamicSlotIdlePhase', 'gdTSSTransmitter', 'gdWakeupSymbolRxIdle',
                   'gdWakeupSymbolRxLow', 'gdWakeupSymbolTxActive', 'gdWakeupSymbolTxIdle', 'pChannels', 'pWakeupChannel', 'pWakeupPattern',
                   'pPayloadLengthDynMax', 'pMicroPerCycle', 'pdListenTimeout', 'pRateCorrectionOut', 'pKeySlotId', 'pKeySlotOnlyEnabled',
                   'pKeySlotUsedForStartup', 'pKeySlotUsedForSync', 'pLatestTx', 'pOffsetCorrectionOut', 'pdAcceptedStartupRange',
                   'pAllowPassiveToActive', 'pClusterDriftDamping', 'pDecodingCorrection', 'pDelayCompensationA', 'pDelayCompensationB',
                   'pMacroInitialOffsetA', 'pMacroInitialOffsetB', 'pMicroInitialOffsetA', 'pMicroInitialOffsetB', 'pAllowHaltDueToClock']
    for f in field_names:
        r += struct.pack('>I', config[f])
    r += struct.pack('>H', config['SCM_EN'] | (config['CLOCK_SRC'] << 1) | (config['FIFOA_EN'] << 2) | (config['FIFOB_EN'] << 3) | (config['LOG_STATUS_DATA'] << 4))
    r += struct.pack('>H', config['BIT_RATE'])
    r += struct.pack('>H', config['FIFOA_Depth'])
    r += struct.pack('>H', config['FIFOA_MIAFV'])
    r += struct.pack('>H', config['FIFOA_MIAFM'])
    r += struct.pack('>H',
        config['FIFOA_F0_EN'] | (config['FIFOA_F0_MODE'] << 1) |
        config['FIFOA_F1_EN'] | (config['FIFOA_F1_MODE'] << 1) |
        config['FIFOA_F2_EN'] | (config['FIFOA_F2_MODE'] << 1) |
        config['FIFOA_F3_EN'] | (config['FIFOA_F3_MODE'] << 1))
    r += struct.pack('>H', config['FIFOA_F0_SID_LOWER'])
    r += struct.pack('>H', config['FIFOA_F0_SID_UPPER'])
    r += struct.pack('>H', config['FIFOA_F1_SID_LOWER'])
    r += struct.pack('>H', config['FIFOA_F1_SID_UPPER'])
    r += struct.pack('>H', config['FIFOA_F2_SID_LOWER'])
    r += struct.pack('>H', config['FIFOA_F2_SID_UPPER'])
    r += struct.pack('>H', config['FIFOA_F3_SID_LOWER'])
    r += struct.pack('>H', config['FIFOA_F3_SID_UPPER'])
    r += struct.pack('>H', config['FIFOB_Depth'])
    r += struct.pack('>H', config['FIFOB_MIAFV'])
    r += struct.pack('>H', config['FIFOB_MIAFM'])
    r += struct.pack('>H',
        config['FIFOB_F0_EN'] | (config['FIFOB_F0_MODE'] << 1) |
        config['FIFOB_F1_EN'] | (config['FIFOB_F1_MODE'] << 1) |
        config['FIFOB_F2_EN'] | (config['FIFOB_F2_MODE'] << 1) |
        config['FIFOB_F3_EN'] | (config['FIFOB_F3_MODE'] << 1))
    r += struct.pack('>H', config['FIFOB_F0_SID_LOWER'])
    r += struct.pack('>H', config['FIFOB_F0_SID_UPPER'])
    r += struct.pack('>H', config['FIFOB_F1_SID_LOWER'])
    r += struct.pack('>H', config['FIFOB_F1_SID_UPPER'])
    r += struct.pack('>H', config['FIFOB_F2_SID_LOWER'])
    r += struct.pack('>H', config['FIFOB_F2_SID_UPPER'])
    r += struct.pack('>H', config['FIFOB_F3_SID_LOWER'])
    r += struct.pack('>H', config['FIFOB_F3_SID_UPPER'])
    r += struct.pack('>H', len(config['RxMsgBufs']))
    r += struct.pack('>H', len(config['TxMsgBufs']))
    for rmb in config['RxMsgBufs']:
        r += struct.pack('>I', 0 | (rmb['Channels'] << 1) |
                         (rmb['CCF_EN'] << 3) |
                         (rmb['CCF_VAL'] << 16) |
                         (rmb['CCF_MASK'] << 24))
        r += struct.pack('>H', rmb['FrameId'])
        r += struct.pack('>H', 0)
    for tmb in config['TxMsgBufs']:
        r += struct.pack('>I', 1 | (tmb['Channels'] << 1) |
                         (tmb['CCF_EN'] << 3) |
                         (tmb['PPI'] << 4) |
                         (tmb['DynPayloadLen'] << 5) |
                         (tmb['CCF_VAL'] << 16) |
                         (tmb['CCF_MASK'] << 24))
        r += struct.pack('>H', tmb['FrameId'])
        r += struct.pack('>H', tmb['PayloadLenMax'])
    return r


def map_frame_id_to_tx_msg_buf_idx(config):
    return {tmb['FrameId']: idx for (idx, tmb) in enumerate(config['TxMsgBufs'])}


def verify_config_format(config):
    missed_keys = set(default_fr_config.keys()) - set(config.keys())
    if len(config['RxMsgBufs']) == 0:
        missed_keys.add('RxMsgBufs')
    if len(config['TxMsgBufs']) == 0:
        missed_keys.add('TxMsgBufs')
    if len(missed_keys) > 0:
        return False, missed_keys
    return True, ()


'''
Verify config parameters:
    1) Parameters constraints, as defined in FlexRay spec 2.1 B.4
    2) Tx/Rx message buffers.
'''
def verify_config(config):
    result = []
    # MPC5748G Reference Manual Table 46-55
    # pdMicrotick (ns) depends on bit rate
    # config['bit_rate']: 0: 10Mbps, 1: 5Mbps, 2: 2.5Mbps, 3: 8Mbps
    pdMicrotick_values = [25.0, 25.0, 50.0, 25.0]
    pSamplesPerMicrotick_values = [2, 1, 1, 2]
    gdSampleClockPeriod_values = [12.5, 25.0, 50.0, 12.5]
    gdBit = cSamplesPerBit * (gdSampleClockPeriod_values[config['BIT_RATE']] / 1000.0)
    result.append('gdBit {} µs'.format(gdBit))
    gdBitMax = gdBit * (1 + cClockDeviationMax)
    pdMicrotick = pdMicrotick_values[config['BIT_RATE']] / 1000.0
    result.append('pdMicrotick {} µs'.format(pdMicrotick))
    # Constraint 3
    T0 = 0.01
    gdMaxPropagationDelay = config['pdBDTx'] + config['LineLength'] * T0 + config['pdStarDelay'] * config['nStar'] + config['pdBDRx']
    result.append('gdMaxPropagationDelay {} µs'.format(gdMaxPropagationDelay))
    if config['gdMinPropagationDelay'] > gdMaxPropagationDelay:
        return False, 'gdMinPropagationDelay should be equal or less than {}'.format(gdMaxPropagationDelay)
    # Constraint 4
    gdMaxMicrotick = pdMicrotick
    gAssumedPrecision = (34 + 20 * config['pClusterDriftDamping']) * gdMaxMicrotick + 2 * gdMaxPropagationDelay
    result.append('gAssumedPrecision {} µs'.format(gAssumedPrecision))

    # Constraint 6
    gdMacrotick_min = cMicroPerMacroNomMin * pdMicrotick
    if config['gdMacrotick'] < cMicroPerMacroNomMin * pdMicrotick:
        return False, 'gdMacrotick should be equal or greater than {}'.format(gdMacrotick_min)

    # Constraint 8
    if config['gdMaxInitializationError'] > gAssumedPrecision:
        return False, 'gdMinPropagationDelay should be equal or less than {}'.format(gAssumedPrecision)
    min_val = 2 * (gdMaxMicrotick * (1 + cClockDeviationMax)) + gdMaxPropagationDelay
    if config['gdMaxInitializationError'] < min_val:
        return False, 'gdMaxInitializationError should be equal or greater than {}'.format(min_val)

    # Constraint 9
    min_val = ceil((gAssumedPrecision + config['gdMaxInitializationError']) / (pdMicrotick * (1 - cClockDeviationMax)))
    if config['pdAcceptedStartupRange'] < min_val:
        return False, 'pdAcceptedStartupRange should be equal or greater than {}'.format(min_val)

    # Constraint 12
    min_val = ceil((gAssumedPrecision - config['gdMinPropagationDelay'] + 2 * config['gdMaxInitializationError']) / (config['gdMacrotick'] * (1 - cClockDeviationMax)))
    if config['gdActionPointOffset'] < min_val:
        return False, 'gdActionPointOffset should be equal or greater than {}'.format(min_val)

    # Constraint 13
    min_val = ceil( (gAssumedPrecision - config['gdMinPropagationDelay']) / (config['gdMacrotick'] * (1 - cClockDeviationMax)) )
    if config['gdMiniSlotActionPointOffset'] < min_val:
        return False, 'gdMinislotActionPointOffset should be equal or greater than {}'.format(min_val)

    # Constraint 14:
    min_val = config['gdMiniSlotActionPointOffset'] + ceil( (gdMaxPropagationDelay + gAssumedPrecision) / (config['gdMacrotick'] * (1 - cClockDeviationMax)) )
    if config['gdMinislot'] < min_val:
        return False, 'gdMinislot should be equal or greater than {}'.format(min_val)

    # Constraint 15
    aFrameLengthStatic = config['gdTSSTransmitter'] + cdFSS + 80 + config['gPayloadLengthStatic'] * 20 + cdFES
    gdStaticSlot = 2 * config['gdActionPointOffset'] + ceil(((aFrameLengthStatic + cChannelIdleDelimiter) * gdBitMax +
        config['gdMinPropagationDelay'] + gdMaxPropagationDelay) / (config['gdMacrotick'] * (1 - cClockDeviationMax)))
    if config['gdStaticSlot'] != gdStaticSlot:
        return False, ('gdStaticSlot', gdStaticSlot)

    # Constraint 16
    gdSymbolWindow_f = lambda dBDRxai: 2 * config['gdActionPointOffset'] + ceil( ((config['gdTSSTransmitter'] + cdCAS +
        cChannelIdleDelimiter) * gdBitMax + dBDRxai + config['gdMinPropagationDelay'] + gdMaxPropagationDelay) / (config['gdMacrotick'] * (1 - cClockDeviationMax)) )
    dBDRxai_min = 0.0
    if config['gdSymbolWindow'] < gdSymbolWindow_f(dBDRxai_min):
        return False, 'gdSymbolWindow should be equal or greater than {}'.format(gdSymbolWindow_f(dBDRxai_min))
    dBDRxai_max = 0.4
    if config['gdSymbolWindow'] > gdSymbolWindow_f(dBDRxai_max):
        return False, 'gdSymbolWindow should be equal or less than {}'.format(gdSymbolWindow_f(dBDRxai_max))

    if (config['gdActionPointOffset'] <= config['gdMiniSlotActionPointOffset'] or config['gNumberOfMinislots'] == 0):
        adActionPointDifference = 0
    else:
        adActionPointDifference = config['gdActionPointOffset'] - config['gdMiniSlotActionPointOffset']
    # Constraint 18:
    gMacroPerCycle = config['gdStaticSlot'] * config['gNumberOfStaticSlots'] + adActionPointDifference + config['gdMinislot'] * config['gNumberOfMinislots'] + config['gdSymbolWindow'] + config['gdNIT']
    result.append('gMacroPerCycle {} MT'.format(gMacroPerCycle))
    # Constraint 17:
    gdCycle = gMacroPerCycle * config['gdMacrotick']
    result.append('gdCycle {} µs'.format(gdCycle))
    if gdCycle > cdCycleMax:
        return False, 'gdCycle should not be greater than {}, but we got {}'.format(cdCycleMax, gdCycle)
    result.append('gdMacrotick {} µs'.format(config['gdMacrotick']))
    # Constraint 19
    pMicroPerCycle = round(gdCycle / pdMicrotick)
    if config['pMicroPerCycle'] != pMicroPerCycle:
        return False, ('pMicroPerCycle', pMicroPerCycle)

    # Constraint 20
    gdDynamicSlotIdlePhase_min = ceil((ceil((cChannelIdleDelimiter * gdBitMax +
        gAssumedPrecision + gdMaxPropagationDelay) / (config['gdMacrotick'] * (1 - cClockDeviationMax))) - (
        config['gdMinislot'] - config['gdMiniSlotActionPointOffset'])) / config['gdMinislot'])
    if config['gdDynamicSlotIdlePhase'] < gdDynamicSlotIdlePhase_min:
        return False, 'gdDynamicSlotIdlePhase should be equal or greater than {}'.format(gdDynamicSlotIdlePhase_min)

    # Constraint 21
    gNumberOfMinislots = \
        (gMacroPerCycle - config['gdNIT'] - adActionPointDifference - config['gNumberOfStaticSlots'] * config['gdStaticSlot'] - config['gdSymbolWindow']) * 1.0 / config['gdMinislot']
    if not gNumberOfMinislots.is_integer():
        return False, 'gNumberOfMinislots should be integer'
    if config['gNumberOfMinislots'] != int(gNumberOfMinislots):
        return False, ('gNumberOfMinislots', gNumberOfMinislots)

    # Constraint 22
    pRateCorrectionOut = ceil(config['pMicroPerCycle'] * 2 * cClockDeviationMax / (1 - cClockDeviationMax))
    if config['pRateCorrectionOut'] != pRateCorrectionOut:
        return False, ('pRateCorrectionOut', pRateCorrectionOut)

    # Constraint 23
    min_val = gAssumedPrecision + gdMaxPropagationDelay - config['gdMinPropagationDelay']
    if config['gOffsetCorrectionMax'] < min_val:
        return False, 'gOffsetCorrectionMax should be equal or greater than {}'.format(min_val)

    # Constraint 24
    max_val = config['gdActionPointOffset'] * config['gdMacrotick'] * (1 + cClockDeviationMax) + config['gdMinPropagationDelay'] + gdMaxPropagationDelay
    if config['gOffsetCorrectionMax'] > max_val:
        return False, 'gOffsetCorrectionMax should be equal or less than {}'.format(max_val)

    # Constraint 25
    pOffsetCorrectionOut = ceil(config['gOffsetCorrectionMax'] / (pdMicrotick * (1 - cClockDeviationMax)))
    if config['pOffsetCorrectionOut'] != pRateCorrectionOut:
        return False, ('pOffsetCorrectionOut', pRateCorrectionOut)

    adOffsetCorrection_min = \
        ceil(config['gOffsetCorrectionMax'] /
             (config['gdMacrotick'] * (1 - cClockDeviationMax) - gdMaxMicrotick * cMicroPerMacroMin * (1 + cClockDeviationMax)))
    result.append('adOffsetCorrection min {} MT'.format(adOffsetCorrection_min))
    # Constraint 26
    max_val = gMacroPerCycle - adOffsetCorrection_min
    if config['gOffsetCorrectionStart'] > max_val:
        return False, 'gOffsetCorrectionStart should be equal or less than {}'.format(max_val)
    if config['gOffsetCorrectionStart'] + config['gdNIT'] <= gMacroPerCycle:
        return False, 'gOffsetCorrectionStart should be greater than {}'.format(gMacroPerCycle - config['gdNIT'])

    # Constraint 29
    pdMaxDrift = ceil(config['pMicroPerCycle'] * 2 * cClockDeviationMax / (1 - cClockDeviationMax))
    result.append('pdMaxDrift {} µT'.format(pdMaxDrift))

    # Constraint 31
    pdListenTimeout = 2 * (config['pMicroPerCycle'] + pdMaxDrift)
    if config['pdListenTimeout'] != pdListenTimeout:
        return False, ('pdListenTimeout', pdListenTimeout)

    # Constraint 32
    pDecodingCorrection = \
        round(((config['gdTSSTransmitter'] + cdFSS + 0.5 * cdBSS) * cSamplesPerBit + cStrobeOffset + cVotingDelay) / pSamplesPerMicrotick_values[config['BIT_RATE']])
    if config['pDecodingCorrection'] != pDecodingCorrection:
        return False, ('pDecodingCorrection', pDecodingCorrection)

    pMicroPerMacroNom = config['pMicroPerCycle'] / gMacroPerCycle
    result.append('pMicroPerMacroNom {} µT/MT'.format(pMicroPerMacroNom))
    # Constraint 33
    pMacroInitialOffsetA = config['gdActionPointOffset'] + ceil((config['pDecodingCorrection'] + config['pDelayCompensationA']) / pMicroPerMacroNom)
    if config['pMacroInitialOffsetA'] != pMacroInitialOffsetA:
        return False, ('pMacroInitialOffsetA', pMacroInitialOffsetA)
    pMacroInitialOffsetB = config['gdActionPointOffset'] + ceil((config['pDecodingCorrection'] + config['pDelayCompensationB']) / pMicroPerMacroNom)
    if config['pMacroInitialOffsetB'] != pMacroInitialOffsetB:
        return False, ('pMacroInitialOffsetB', pMacroInitialOffsetB)
    # Constraint 34
    pMicroInitialOffsetA = round((ceil(pMicroPerMacroNom) - ((config['pDecodingCorrection'] + config['pDelayCompensationA']) % pMicroPerMacroNom)) % pMicroPerMacroNom)
    if config['pMicroInitialOffsetA'] != pMicroInitialOffsetA:
        return False, ('pMicroInitialOffsetA', pMicroInitialOffsetA)
    pMicroInitialOffsetB = round((ceil(pMicroPerMacroNom) - ((config['pDecodingCorrection'] + config['pDelayCompensationB']) % pMicroPerMacroNom)) % pMicroPerMacroNom)
    if config['pMicroInitialOffsetB'] != pMicroInitialOffsetB:
        return False, ('pMicroInitialOffsetB', pMicroInitialOffsetB)
    # Constraint 36
    vDTSLowMin = 1
    aFrameLengthDynamic = config['gdTSSTransmitter'] + cdFSS + 80 + config['pPayloadLengthDynMax'] * 20 + cdFES
    pLatestTx_max = \
        floor(config['gNumberOfMinislots'] - (((aFrameLengthDynamic + vDTSLowMin) * gdBitMax) / (config['gdMacrotick'] * (1 - cClockDeviationMax) * config['gdMinislot'])) - config['gdDynamicSlotIdlePhase'])
    result.append('pLatestTx max {} Minislot'.format(pLatestTx_max))
    if config['pLatestTx'] > pLatestTx_max:
        return False, 'pLatestTx should be equal or less than {}'.format(pLatestTx_max)
    # Constraint 39
    gdWakeupSymbolTxIdle = ceil(cdWakeupSymbolTxIdle / gdBit)
    if config['gdWakeupSymbolTxIdle'] != gdWakeupSymbolTxIdle:
        return False, ('gdWakeupSymbolTxIdle', gdWakeupSymbolTxIdle)
    # Constraint 40
    gdWakeupSymbolTxLow = ceil(cdWakeupSymbolTxLow / gdBit)
    if config['gdWakeupSymbolTxActive'] != gdWakeupSymbolTxLow:
        return False, ('gdWakeupSymbolTxActive', gdWakeupSymbolTxLow)

    rx_slot_ids = [x['FrameId'] for x in config['RxMsgBufs']]
    if len(rx_slot_ids) == 0:
        return False, 'There should be at least one Rx message buffer.'
    if len(set(rx_slot_ids)) != len(rx_slot_ids):
        return False, 'Duplicated frame id found in Rx message buffers.'

    tx_slot_ids = [x['FrameId'] for x in config['TxMsgBufs']]
    if len(set(tx_slot_ids)) != len(tx_slot_ids):
        return False, 'Duplicated frame id found in Tx message buffers.'

    same_fids = set(rx_slot_ids) & set(tx_slot_ids)
    if len(same_fids) > 0:
        return False, 'Frame ids {} found both in Tx and Rx message buffers.'.format(same_fids)

    if config['pKeySlotId'] == 0:
        return False, 'pKeySlotId should not be zero.'
    if config['pKeySlotId'] not in tx_slot_ids:
        return False, 'pKeySlotId {} not found in Tx message buffers.'.format(config['pKeySlotId'])
    if config['pKeySlotId'] in rx_slot_ids:
        return False, 'pKeySlotId {} should not be in Rx message buffers.'.format(config['pKeySlotId'])
    return True, result
