import struct
import numpy as np
import pandas as pd

# from .qc import fix_adc_offset

def decode_gps_packet(mp):
    result = dict()
    result['start_byte'] = struct.unpack('B', mp[0:1])[0]
    result['millis'] = struct.unpack('I', mp[1:5])[0]
    result['latitude'] = struct.unpack('i', mp[5:9])[0] / 10000
    result['longitude'] = struct.unpack('i', mp[9:13])[0] / 10000
    result['altitude'] = struct.unpack('H', mp[13:15])[0]
    result['gps_time'] = struct.unpack('I', mp[15:19])[0]
    result['end_byte'] = struct.unpack('B', mp[19:])[0]
    return result

def decode_data_packet(mp):
    result = dict()
    result['start_byte'] = struct.unpack('B', mp[0:1])[0]
    result['adc_ready_millis'] = struct.unpack('I', mp[1:5])[0]
    result['adc_reading'] = struct.unpack('i', mp[5:9])[0]
    result['magnetometer_x'] = struct.unpack('h', mp[9:11])[0]
    result['magnetometer_y'] = struct.unpack('h', mp[11:13])[0]
    result['magnetometer_z'] = struct.unpack('h', mp[13:15])[0]
    result['gyroscope_x'] = struct.unpack('h', mp[15:17])[0]
    result['gyroscope_y'] = struct.unpack('h', mp[17:19])[0]
    result['gyroscope_z'] = struct.unpack('h', mp[19:21])[0]
    result['acceleration_x'] = struct.unpack('h', mp[21:23])[0]
    result['acceleration_y'] = struct.unpack('h', mp[23:25])[0]
    result['acceleration_z'] = struct.unpack('h', mp[25:27])[0]
    result['temperature'] = struct.unpack('h', mp[27:29])[0] / 10
    result['relative_humidity'] = struct.unpack('H', mp[29:31])[0]
    result['pressure'] = struct.unpack('H', mp[31:33])[0] / 10
    result['end_byte'] = struct.unpack('B', mp[33:])[0]
    return result

def decode_data_packet_old(mp):
    result = dict()
    result['start_byte'] = struct.unpack('B', mp[0:1])[0]
    result['adc_ready_millis'] = struct.unpack('I', mp[1:5])[0]
    result['adc_reading'] = struct.unpack('i', mp[5:9])[0]
    result['acceleration_x'] = struct.unpack('f', mp[9:13])[0]
    result['acceleration_y'] = struct.unpack('f', mp[13:17])[0]
    result['acceleration_z'] = struct.unpack('f', mp[17:21])[0]
    result['magnetometer_x'] = struct.unpack('f', mp[21:25])[0]
    result['magnetometer_y'] = struct.unpack('f', mp[25:29])[0]
    result['magnetometer_z'] = struct.unpack('f', mp[29:33])[0]
    result['gyroscope_x'] = struct.unpack('f', mp[33:37])[0]
    result['gyroscope_y'] = struct.unpack('f', mp[37:41])[0]
    result['gyroscope_z'] = struct.unpack('f', mp[41:45])[0]
    result['temperature'] = struct.unpack('h', mp[45:47])[0] / 10
    result['relative_humidity'] = struct.unpack('H', mp[47:49])[0]
    result['pressure'] = struct.unpack('H', mp[49:51])[0] / 10
    result['end_byte'] = struct.unpack('B', mp[51:])[0]
    return result

def packets_convert(ba, return_all=False,
                    gps_packet_length=20, gps_start_bytes=[],
                    data_packet_length=34, data_start_bytes=[],
                    ):
    # Find Starts
    for i in range(len(ba) - gps_packet_length):
        if (ba[i] == 254) and (ba[i+gps_packet_length-1] == 237):
            gps_start_bytes.append(i)

    # Process GPS
    gps_raw_packets = []
    bytes_to_remove = []
    for sb in gps_start_bytes:
        gps_raw_packets.append(ba[sb:sb+gps_packet_length])
        bytes_to_remove.append(range(sb, sb+gps_packet_length))
    if len(gps_raw_packets) == 0:
        gps_raw_packets.append(bytearray(gps_packet_length))

    bytes_to_remove = np.array(bytes_to_remove)
    bytes_to_remove = bytes_to_remove.ravel()
    #ba = np.delete(ba, bytes_to_remove)

    gps_packets = []
    for p in gps_raw_packets:
        gps_packets.append(decode_gps_packet(p))


    series = {}
    for field in gps_packets[0].keys():
        vals = []
        for p in gps_packets:
            vals.append(p[field])
        series[field] = vals
    df_gps = pd.DataFrame(series)
    df_gps['gps_time'] =  pd.to_datetime(df_gps['gps_time'], format='%H%M%S00', errors='coerce')
    df_gps.replace([0], np.nan, inplace=True)

    # Find Starts
    for i in range(len(ba) - data_packet_length):
        if (ba[i] == 190) and (ba[i+data_packet_length-1] == 239):
            data_start_bytes.append(i)

    # Process
    data_raw_packets = []
    for sb in data_start_bytes:
        data_raw_packets.append(ba[sb:sb+data_packet_length])

    data_packets = []
    for p in data_raw_packets:
        data_packets.append(decode_data_packet(p))

    series = {}
    for field in data_packets[0].keys():
        vals = []
        for p in data_packets:
            vals.append(p[field])
        series[field] = vals
    df_fiber = pd.DataFrame(series)

    adc_cal = (2 * 2.048) / 2**24
    adc_cal *= 2 # For voltage divider that is in-place
    df_fiber['adc_volts'] = adc_cal * df_fiber['adc_reading']

    if return_all==True:
        return df_gps, df_fiber, data_packets, gps_packets
    else:
        return df_gps, df_fiber

def read_efm_raw(filenames):
    """ Read raw EFM files from SD card given a list of filenames.
    """
    ba = bytearray()
    for filename in filenames:
        with open(filename, 'rb') as f:
            print (f'{f}.TXT')
            ba = ba + f.read()
    df_gps, df_fiber = packets_convert(ba)

    return df_gps, df_fiber