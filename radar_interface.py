import serial
import time
import struct
import numpy as np
import threading
from collections import deque

class RadarInterface:
    """Interface for IWR6843ISK mmWave radar"""
    
    # Radar configuration parameters based on IWR6843ISK
    # Magic word as per TI's specification
    MAGIC_WORD = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Message types - Updated based on IWR6843ISK SDK
    MSG_HEADER_LEN = 40  # 40 bytes for SDK 3.x (8 magic + 8*4 fields)
    TLV_HEADER_LEN = 8
    
    # TLV Types for SDK 3.x
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
    MMWDEMO_OUTPUT_MSG_RANGE_PROFILE = 2
    MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
    MMWDEMO_OUTPUT_MSG_STATS = 6
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO = 7
    MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8
    MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS = 9
    
    # For detected points
    MMWDEMO_OUTPUT_MSG_POINT_CLOUD_INDICES = 10
    
    def __init__(self, cli_port, data_port):
        """Initialize radar interface with CLI and data ports"""
        self.cli_port = cli_port
        self.data_port = data_port
        self.cli_serial = None
        self.data_serial = None
        self.is_sensor_running = False
        self.data_buffer = bytearray()
        self.frame_count = 0
        
        # Open serial ports
        self._open_ports()
        
    def _open_ports(self):
        """Open CLI and data serial ports"""
        try:
            # Open CLI port for sending configuration
            self.cli_serial = serial.Serial(
                port=self.cli_port,
                baudrate=115200,
                timeout=0.3,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Open data port for receiving radar data
            self.data_serial = serial.Serial(
                port=self.data_port,
                baudrate=921600,
                timeout=0.025,  # Reduced timeout for faster reading
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Clear any existing data
            time.sleep(0.1)
            self.cli_serial.reset_input_buffer()
            self.cli_serial.reset_output_buffer()
            self.data_serial.reset_input_buffer()
            self.data_serial.reset_output_buffer()
            
            print(f"DEBUG: Ports opened successfully - CLI: {self.cli_port}, Data: {self.data_port}")
            
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
                # Skip empty lines
                if not cmd or cmd.isspace():
                    continue
                    
                # Handle sensorStart specially
                if cmd.lower().strip() == 'sensorstart':
                    print("DEBUG: Found sensorStart in config, skipping (will start manually)")
                    continue
                    
                # Clear input buffer before sending command
                self.cli_serial.reset_input_buffer()
                
                # Send command
                self.cli_serial.write((cmd + '\n').encode())
                time.sleep(0.01)  # Small delay between commands
                
                # Read response with multiple attempts
                response = ""
                max_attempts = 3
                start_time = time.time()
                timeout = 0.1  # 100ms timeout per command
                
                while (time.time() - start_time) < timeout:
                    if self.cli_serial.in_waiting > 0:
                        try:
                            raw_data = self.cli_serial.read(self.cli_serial.in_waiting)
                            # Try to decode as ASCII/UTF-8
                            decoded = raw_data.decode('ascii', errors='ignore').strip()
                            # Filter out non-printable characters except newlines, tabs
                            filtered = ''.join(char for char in decoded if char.isprintable() or char in '\n\r\t')
                            response += filtered
                            
                            # Check if we got a complete response (ends with prompt or Done)
                            if response and ('>' in response or 'Done' in response or '\n' in response):
                                break
                        except Exception as e:
                            response = f"Decode error: {str(e)}"
                            break
                    time.sleep(0.01)  # Small delay to avoid busy waiting
                
                # Clean up the response
                response = response.strip()
                
                responses.append({'command': cmd, 'response': response})
                
                # Check for error in response
                if 'Error' in response or 'error' in response.lower():
                    print(f"Error in command: {cmd}")
                    return False, responses
                    
            print(f"DEBUG: Configuration sent, {len(responses)} commands processed")
            return True, responses
            
        except Exception as e:
            print(f"Error sending configuration: {str(e)}")
            return False, f"Exception: {str(e)}"
            
    def start_sensor(self):
        """Start the sensor"""
        if self.cli_serial and self.cli_serial.is_open:
            # Clear input buffer before sending command
            self.cli_serial.reset_input_buffer()
            
            # Clear data buffer before starting
            self.data_buffer = bytearray()
            self.frame_count = 0
            
            # Also clear data serial buffer
            if self.data_serial and self.data_serial.is_open:
                self.data_serial.reset_input_buffer()
                
            print("DEBUG: Sending sensorStart command...")
            self.cli_serial.write(b'sensorStart\n')
            time.sleep(0.1)
            
            # Read response with multiple attempts
            response = ""
            start_time = time.time()
            timeout = 0.5  # 500ms timeout for start command
            
            while (time.time() - start_time) < timeout:
                if self.cli_serial.in_waiting > 0:
                    try:
                        raw_data = self.cli_serial.read(self.cli_serial.in_waiting)
                        decoded = raw_data.decode('ascii', errors='ignore').strip()
                        filtered = ''.join(char for char in decoded if char.isprintable() or char in '\n\r\t')
                        response += filtered
                        if response and not response.isspace():
                            break
                    except Exception as e:
                        response = f"Decode error: {str(e)}"
                time.sleep(0.01)  # Small delay to avoid busy waiting
            
            if not response or response.isspace():
                response = "No response (timeout)"
                
            print(f"DEBUG: sensorStart response: {response}")
            self.is_sensor_running = True
            
            # Give sensor time to start sending data
            time.sleep(0.1)
            
            # Check if data is coming in
            if self.data_serial and self.data_serial.is_open:
                time.sleep(0.1)
                bytes_waiting = self.data_serial.in_waiting
                print(f"DEBUG: After starting sensor, {bytes_waiting} bytes waiting on data port")
                
            return True, response
        return False, "CLI serial port is not open"
            
    def stop_sensor(self):
        """Stop the sensor"""
        if self.cli_serial and self.cli_serial.is_open:
            # Clear input buffer before sending command
            self.cli_serial.reset_input_buffer()
            
            self.cli_serial.write(b'sensorStop\n')
            time.sleep(0.1)
            
            # Read response with multiple attempts
            response = ""
            max_attempts = 3
            start_time = time.time()
            timeout = 0.2  # 200ms timeout
            
            while (time.time() - start_time) < timeout:
                if self.cli_serial.in_waiting > 0:
                    try:
                        raw_data = self.cli_serial.read(self.cli_serial.in_waiting)
                        decoded = raw_data.decode('ascii', errors='ignore').strip()
                        filtered = ''.join(char for char in decoded if char.isprintable() or char in '\n\r\t')
                        response += filtered
                        if response and not response.isspace():
                            break
                    except Exception as e:
                        response = f"Decode error: {str(e)}"
                time.sleep(0.01)  # Small delay to avoid busy waiting
            
            if not response or response.isspace():
                response = "No response (timeout)"
                
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
                
                # Debug: print buffer size periodically
                self.frame_count += 1
                
                # First, just check if we're getting ANY data
                if self.frame_count == 10:
                    print(f"DEBUG: After 10 reads, total bytes received: {len(self.data_buffer)}")
                    if len(self.data_buffer) == 0:
                        print("DEBUG: No data received at all! Check:")
                        print("  1. Is the radar powered on?")
                        print("  2. Are the COM ports correct?")
                        print("  3. Is the sensor actually started?")
                
                # Minimal buffer status - only print every 500 frames
                if self.frame_count % 500 == 0:
                    print(f"DEBUG: Buffer size: {len(self.data_buffer)} bytes")
                
            # Try to parse frame from buffer
            frame_data = self._parse_frame()
            if frame_data:
                return self._process_frame(frame_data)
                
        except Exception as e:
            print(f"Error reading data: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return None
        
    def _find_magic_word_pattern(self):
        """Debug function to find potential magic word patterns"""
        # Look for common magic word patterns
        patterns = [
            [0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07],  # Standard
            [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],  # Sequential
            [0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07],  # TI Standard
        ]
        
        for pattern in patterns:
            pattern_str = ' '.join([f'{b:02X}' for b in pattern])
            # Search for pattern in buffer
            for i in range(min(len(self.data_buffer) - 8, 100)):
                if all(self.data_buffer[i+j] == pattern[j] for j in range(8)):
                    print(f"DEBUG: Found pattern {pattern_str} at offset {i}")
                    break
        
    def _check_magic_word(self):
        """Check if buffer starts with magic word"""
        if len(self.data_buffer) < 8:
            return False
            
        for i in range(8):
            if self.data_buffer[i] != self.MAGIC_WORD[i]:
                return False
        return True
        
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
                        # Invalid header, skip one byte instead of magic word
                        self.data_buffer = self.data_buffer[1:]
                else:
                    # Wait for more data
                    return None
            else:
                # No magic word found, skip one byte
                self.data_buffer = self.data_buffer[1:]
                
        return None
        
    def _parse_header(self):
        """Parse message header"""
        try:
            header = {}
            idx = 0
            
            # Check if we have enough data for complete header
            if len(self.data_buffer) < self.MSG_HEADER_LEN:
                return None
            
            # Magic word (8 bytes) - already checked
            idx += 8
            
            # Version (4 bytes) - Don't reject based on version
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
        
        # Debug: when objects are detected, show TLV info
        tlv_debug_info = []
        
        for i in range(header['numTLVs']):
            if idx + self.TLV_HEADER_LEN > len(frame_data):
                break
                
            # Parse TLV header
            tlv_type = struct.unpack('<I', frame_data[idx:idx+4])[0]
            tlv_length = struct.unpack('<I', frame_data[idx+4:idx+8])[0]
            idx += self.TLV_HEADER_LEN
            
            # Collect TLV info for debug
            tlv_debug_info.append(f"Type:{tlv_type}(len:{tlv_length})")
                
            # Validate TLV length
            if tlv_length > len(frame_data) or idx + tlv_length > len(frame_data):
                break
                
            # Parse TLV data based on type
            if tlv_type == self.MMWDEMO_OUTPUT_MSG_DETECTED_POINTS or tlv_type == 1:
                print(f"DEBUG: Found detected points TLV (type {tlv_type}) with {header['numDetectedObj']} objects")
                result['pointCloud'] = self._parse_point_cloud(
                    frame_data[idx:idx+tlv_length],
                    header['numDetectedObj']
                )
            elif tlv_type == 0 and header['numDetectedObj'] > 0:
                # Sometimes point cloud data comes as type 0
                # Check if the length makes sense for point cloud data
                expected_size = header['numDetectedObj'] * 16  # 4 floats per point
                if tlv_length >= expected_size or tlv_length == header['numDetectedObj'] * 12:
                    print(f"DEBUG: Type 0 TLV might be point cloud data (len={tlv_length}, expected={expected_size})")
                    result['pointCloud'] = self._parse_point_cloud(
                        frame_data[idx:idx+tlv_length],
                        header['numDetectedObj']
                    )
                
            idx += tlv_length
        
        # Debug: show TLV types when objects are detected    
        if header['numDetectedObj'] > 0:
            print(f"DEBUG: Frame {header['frameNumber']} TLVs: {', '.join(tlv_debug_info)}")
            
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
            
            # Only print header info when there are detected objects
            if header['numDetectedObj'] > 0:
                print(f"DEBUG: Frame {header['frameNumber']}: {header['numDetectedObj']} objects, {header['numTLVs']} TLVs")
            
            return header
            
        except Exception as e:
            print(f"DEBUG: Error in _parse_header_from_frame: {str(e)}")
            return None
            
    def _parse_point_cloud(self, data, num_points):
        """Parse point cloud data - SDK 3.x format"""
        points = []
        
        # Check if we have the expected data structure
        if num_points == 0:
            return np.array([]).reshape(0, 4)
            
        # Calculate expected sizes for different formats
        # Format 1: x, y, z, doppler (4 floats = 16 bytes per point)
        size_format1 = num_points * 16
        # Format 2: range, azimuth, elevation, doppler (4 floats = 16 bytes per point) 
        size_format2 = num_points * 16
        # Format 3: x, y, z, doppler, snr (5 floats = 20 bytes per point)
        size_format3 = num_points * 20
        
        # Try to determine format based on data length
        if len(data) >= size_format3:
            # Format with SNR
            point_size = 20
            for i in range(num_points):
                if i * point_size + point_size > len(data):
                    break
                    
                idx = i * point_size
                x = struct.unpack('<f', data[idx:idx+4])[0]
                y = struct.unpack('<f', data[idx+4:idx+8])[0]
                z = struct.unpack('<f', data[idx+8:idx+12])[0]
                doppler = struct.unpack('<f', data[idx+12:idx+16])[0]
                snr = struct.unpack('<f', data[idx+16:idx+20])[0]
                
                points.append([x, y, z, doppler])
                
        elif len(data) >= size_format1:
            # Standard format
            point_size = 16
            for i in range(num_points):
                if i * point_size + point_size > len(data):
                    break
                    
                idx = i * point_size
                x = struct.unpack('<f', data[idx:idx+4])[0]
                y = struct.unpack('<f', data[idx+4:idx+8])[0]
                z = struct.unpack('<f', data[idx+8:idx+12])[0]
                doppler = struct.unpack('<f', data[idx+12:idx+16])[0]
                
                points.append([x, y, z, doppler])
        else:
            # Try a more compact format (range, azimuth, doppler)
            point_size = 12
            for i in range(num_points):
                if i * point_size + point_size > len(data):
                    break
                    
                idx = i * point_size
                range_val = struct.unpack('<f', data[idx:idx+4])[0]
                azimuth = struct.unpack('<f', data[idx+4:idx+8])[0]
                doppler = struct.unpack('<f', data[idx+8:idx+12])[0]
                
                # Convert spherical to cartesian
                x = range_val * np.sin(azimuth)
                y = range_val * np.cos(azimuth)
                z = 0.0  # No elevation data
                
                points.append([x, y, z, doppler])
        
        if len(points) > 0:
            print(f"DEBUG: Successfully parsed {len(points)} points from {num_points} expected")
            
        return np.array(points) if points else np.array([]).reshape(0, 4)
        
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