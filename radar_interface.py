import serial
import time
import struct
import numpy as np
import threading
from collections import deque

class RadarInterface:
    """Interface for IWR6843ISK mmWave radar"""
    
    # Radar configuration parameters based on IWR6843ISK
    MAGIC_WORD = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Message types
    MSG_HEADER_LEN = 36
    TLV_HEADER_LEN = 8
    POINT_CLOUD_TLV_TYPE = 1
    TARGET_LIST_TLV_TYPE = 2
    
    def __init__(self, cli_port, data_port):
        """Initialize radar interface with CLI and data ports"""
        self.cli_port = cli_port
        self.data_port = data_port
        self.cli_serial = None
        self.data_serial = None
        self.is_sensor_running = False
        self.data_buffer = bytearray()
        
        # Open serial ports
        self._open_ports()
        
    def _open_ports(self):
        """Open CLI and data serial ports"""
        try:
            # Open CLI port for sending configuration
            self.cli_serial = serial.Serial(
                port=self.cli_port,
                baudrate=115200,
                timeout=0.3
            )
            
            # Open data port for receiving radar data
            self.data_serial = serial.Serial(
                port=self.data_port,
                baudrate=921600,
                timeout=0.1
            )
            
            # Clear any existing data
            self.cli_serial.reset_input_buffer()
            self.data_serial.reset_input_buffer()
            
        except Exception as e:
            self.close()
            raise Exception(f"Failed to open serial ports: {str(e)}")
            
    def is_connected(self):
        """Check if both serial ports are connected"""
        return (self.cli_serial is not None and self.cli_serial.is_open and
                self.data_serial is not None and self.data_serial.is_open)
                
    def send_config(self, config_file='xwr68xx_config.cfg'):
        """Send radar configuration commands from file"""
        if not self.cli_serial or not self.cli_serial.is_open:
            return False, "CLI serial port is not open"
            
        responses = []
        try:
            # Read configuration from file if provided
            config_commands = []
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('%'):
                            config_commands.append(line)
            except FileNotFoundError:
                # Use default configuration for IWR6843ISK
                config_commands = [
                    "sensorStop",
                    "flushCfg", 
                    "dfeDataOutputMode 1",
                    "channelCfg 15 7 0",
                    "adcCfg 2 1",
                    "adcbufCfg -1 0 1 1 1",
                    "profileCfg 0 60 47 7 57.14 0 0 70 1 256 5209 0 0 48",
                    "chirpCfg 0 0 0 0 0 0 0 1",
                    "chirpCfg 1 1 0 0 0 0 0 4",
                    "chirpCfg 2 2 0 0 0 0 0 2",
                    "frameCfg 0 2 16 0 100 1 0",
                    "lowPower 0 0",
                    "guiMonitor -1 1 1 0 0 0 1",
                    "cfarCfg -1 0 2 8 4 3 0 15 1",
                    "cfarCfg -1 1 0 4 2 3 1 15 1",
                    "multiObjBeamForming -1 1 0.5",
                    "clutterRemoval -1 0",
                    "calibDcRangeSig -1 0 -5 8 256",
                    "extendedMaxVelocity -1 0",
                    "lvdsStreamCfg -1 0 0 0",
                    "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
                    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
                    "CQRxSatMonitor 0 3 5 121 0",
                    "CQSigImgMonitor 0 127 4",
                    "analogMonitor 0 0",
                    "aoaFovCfg -1 -90 90 -90 90",
                    "cfarFovCfg -1 0 0 8.92",
                    "cfarFovCfg -1 1 -1 1.00",
                    "calibData 0 0 0"
                ]
            
            # Send each command
            for cmd in config_commands:
                # Skip sensorStart command if present (we'll start manually)
                if cmd.lower() == 'sensorstart':
                    continue
                    
                self.cli_serial.write((cmd + '\n').encode())
                time.sleep(0.01)
                
                # Read response
                response = self.cli_serial.read(256).decode('utf-8', errors='ignore').strip()
                responses.append({'command': cmd, 'response': response})
                
                if 'Error' in response:
                    print(f"Error in command: {cmd}")
                    return False, responses
                    
            return True, responses
            
        except Exception as e:
            print(f"Error sending configuration: {str(e)}")
            return False, f"Exception: {str(e)}"
            
    def start_sensor(self):
        """Start the sensor"""
        if self.cli_serial and self.cli_serial.is_open:
            self.cli_serial.write(b'sensorStart\n')
            time.sleep(0.1)
            response = self.cli_serial.read(256).decode('utf-8', errors='ignore').strip()
            self.is_sensor_running = True
            return True, response
        return False, "CLI serial port is not open"
            
    def stop_sensor(self):
        """Stop the sensor"""
        if self.cli_serial and self.cli_serial.is_open:
            self.cli_serial.write(b'sensorStop\n')
            time.sleep(0.1)
            response = self.cli_serial.read(256).decode('utf-8', errors='ignore').strip()
            self.is_sensor_running = False
            return True, response
        return False, "CLI serial port is not open"
            
    def read_data(self):
        """Read and parse radar data"""
        if not self.data_serial or not self.data_serial.is_open:
            return None
            
        try:
            # Read available data
            bytes_available = self.data_serial.in_waiting
            if bytes_available > 0:
                new_data = self.data_serial.read(bytes_available)
                self.data_buffer.extend(new_data)
                
            # Try to parse frame from buffer
            frame_data = self._parse_frame()
            if frame_data:
                return self._process_frame(frame_data)
                
        except Exception as e:
            print(f"Error reading data: {str(e)}")
            
        return None
        
    def _parse_frame(self):
        """Parse a complete frame from the data buffer"""
        # Look for magic word
        while len(self.data_buffer) >= 8:
            # Check if we have magic word at the beginning
            if self._check_magic_word():
                # Try to parse header
                if len(self.data_buffer) >= self.MSG_HEADER_LEN:
                    header = self._parse_header()
                    if header:
                        total_packet_len = header['totalPacketLen']
                        
                        # Check if we have complete frame
                        if len(self.data_buffer) >= total_packet_len:
                            # Extract frame data
                            frame_data = bytes(self.data_buffer[:total_packet_len])
                            self.data_buffer = self.data_buffer[total_packet_len:]
                            return frame_data
                        else:
                            # Wait for more data
                            return None
                    else:
                        # Invalid header, skip magic word
                        self.data_buffer = self.data_buffer[8:]
                else:
                    # Wait for more data
                    return None
            else:
                # No magic word found, skip one byte
                self.data_buffer = self.data_buffer[1:]
                
        return None
        
    def _check_magic_word(self):
        """Check if buffer starts with magic word"""
        if len(self.data_buffer) < 8:
            return False
            
        for i in range(8):
            if self.data_buffer[i] != self.MAGIC_WORD[i]:
                return False
        return True
        
    def _parse_header(self):
        """Parse message header"""
        try:
            header = {}
            idx = 0
            
            # Magic word (8 bytes) - already checked
            idx += 8
            
            # Version (4 bytes)
            header['version'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            idx += 4
            
            # Total packet length (4 bytes)
            header['totalPacketLen'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            idx += 4
            
            # Platform (4 bytes)
            header['platform'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            idx += 4
            
            # Frame number (4 bytes)
            header['frameNumber'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            idx += 4
            
            # Time CPU cycles (4 bytes)
            header['timeCpuCycles'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            idx += 4
            
            # Number of detected objects (4 bytes)
            header['numDetectedObj'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            idx += 4
            
            # Number of TLVs (4 bytes)
            header['numTLVs'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            idx += 4
            
            # Subframe number (4 bytes)
            header['subFrameNumber'] = struct.unpack('<I', self.data_buffer[idx:idx+4])[0]
            
            return header
            
        except Exception as e:
            print(f"Error parsing header: {str(e)}")
            return None
            
    def _process_frame(self, frame_data):
        """Process a complete frame and extract TLVs"""
        result = {}
        
        # Parse header again from frame data
        header = self._parse_header_from_frame(frame_data)
        if not header:
            return None
            
        result['header'] = header
        result['pointCloud'] = None
        result['targets'] = None
        
        # Parse TLVs
        idx = self.MSG_HEADER_LEN
        
        for i in range(header['numTLVs']):
            if idx + self.TLV_HEADER_LEN > len(frame_data):
                break
                
            # Parse TLV header
            tlv_type = struct.unpack('<I', frame_data[idx:idx+4])[0]
            tlv_length = struct.unpack('<I', frame_data[idx+4:idx+8])[0]
            idx += self.TLV_HEADER_LEN
            
            if idx + tlv_length > len(frame_data):
                break
                
            # Parse TLV data based on type
            if tlv_type == self.POINT_CLOUD_TLV_TYPE:
                result['pointCloud'] = self._parse_point_cloud(
                    frame_data[idx:idx+tlv_length],
                    header['numDetectedObj']
                )
            elif tlv_type == self.TARGET_LIST_TLV_TYPE:
                result['targets'] = self._parse_target_list(
                    frame_data[idx:idx+tlv_length]
                )
                
            idx += tlv_length
            
        return result
        
    def _parse_header_from_frame(self, frame_data):
        """Parse header from frame data"""
        if len(frame_data) < self.MSG_HEADER_LEN:
            return None
            
        try:
            header = {}
            idx = 8  # Skip magic word
            
            header['version'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            idx += 4
            header['totalPacketLen'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            idx += 4
            header['platform'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            idx += 4
            header['frameNumber'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            idx += 4
            header['timeCpuCycles'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            idx += 4
            header['numDetectedObj'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            idx += 4
            header['numTLVs'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            idx += 4
            header['subFrameNumber'] = struct.unpack('<I', frame_data[idx:idx+4])[0]
            
            return header
            
        except:
            return None
            
    def _parse_point_cloud(self, data, num_points):
        """Parse point cloud data"""
        point_size = 16  # 4 floats (x, y, z, doppler) * 4 bytes
        points = []
        
        for i in range(num_points):
            if i * point_size + point_size > len(data):
                break
                
            idx = i * point_size
            x = struct.unpack('<f', data[idx:idx+4])[0]
            y = struct.unpack('<f', data[idx+4:idx+8])[0]
            z = struct.unpack('<f', data[idx+8:idx+12])[0]
            doppler = struct.unpack('<f', data[idx+12:idx+16])[0]
            
            points.append([x, y, z])
            
        return np.array(points) if points else np.array([]).reshape(0, 3)
        
    def _parse_target_list(self, data):
        """Parse target list data"""
        # Implementation depends on specific radar configuration
        # This is a placeholder
        return []
        
    def close(self):
        """Close serial connections"""
        if self.cli_serial and self.cli_serial.is_open:
            if self.is_sensor_running:
                self.stop_sensor()
            self.cli_serial.close()
            
        if self.data_serial and self.data_serial.is_open:
            self.data_serial.close() 