"""
Modbus TCP client for ADAM-6117 weight sensor integration.
Handles non-blocking async polling of weight registers.
"""

import logging
from typing import Dict, Optional
import asyncio
from pymodbus.client import AsyncModbusTcpClient

logger = logging.getLogger(__name__)


class ModbusWeightClient:
    """Async Modbus TCP client for ADAM-6117 weight modules."""
    
    def __init__(self, config: Dict):
        """
        Initialize Modbus TCP client.
        
        Args:
            config: Modbus settings from settings.yaml
        """
        self.config = config
        self.client = None
        self.connected = False
        self.weight_cache = {}
    
    async def connect(self) -> bool:
        """Connect to Modbus TCP server."""
        try:
            self.client = AsyncModbusTcpClient(
                host=self.config['ip_address'],
                port=self.config['port'],
            )
            await self.client.connect()
            self.connected = True
            logger.info(
                f"Connected to Modbus TCP: {self.config['ip_address']}:{self.config['port']}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Modbus: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Modbus TCP server."""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("Disconnected from Modbus TCP")
    
    async def read_bin_weight(self, bin_id: str) -> Optional[float]:
        """
        Read weight for a specific bin.
        
        Args:
            bin_id: Bin identifier (e.g., 'bin_0_0')
            
        Returns:
            Weight in grams, or None if read fails
        """
        if not self.connected:
            logger.warning("Modbus not connected")
            return None
        
        register_key = f"{bin_id}_weight"
        register_addr = self.config['registers'].get(register_key)
        
        if register_addr is None:
            logger.error(f"Register not found for {bin_id}")
            return None
        
        try:
            response = await self.client.read_holding_registers(
                address=register_addr,
                count=2,  # Read 2 registers for 32-bit value
                slave=self.config['slave_id'],
            )
            
            if response.isError():
                logger.error(f"Modbus error reading {bin_id}: {response}")
                return None
            
            # Convert 2x16-bit registers to 32-bit float
            weight = self._decode_weight(response.registers)
            self.weight_cache[bin_id] = weight
            
            return weight
            
        except Exception as e:
            logger.error(f"Exception reading weight for {bin_id}: {e}")
            return None
    
    async def read_all_weights(self) -> Dict[str, float]:
        """
        Read weights for all configured bins.
        
        Returns:
            Dictionary mapping bin_id to weight (grams)
        """
        weights = {}
        
        for bin_id in self.config['registers'].keys():
            if '_weight' in bin_id:
                bin_name = bin_id.replace('_weight', '')
                weight = await self.read_bin_weight(bin_name)
                if weight is not None:
                    weights[bin_name] = weight
        
        return weights
    
    def get_weight_delta(self, bin_id: str, previous_weight: float) -> float:
        """
        Calculate weight change for a bin.
        
        Args:
            bin_id: Bin identifier
            previous_weight: Previous weight reading
            
        Returns:
            Weight delta in grams
        """
        current_weight = self.weight_cache.get(bin_id, 0.0)
        delta = current_weight - previous_weight
        return abs(delta)
    
    @staticmethod
    def _decode_weight(registers: list) -> float:
        """
        Decode 32-bit weight from two 16-bit Modbus registers.
        
        Args:
            registers: List of register values
            
        Returns:
            Weight as float (grams)
        """
        if len(registers) < 2:
            return 0.0
        
        # Combine two 16-bit registers into 32-bit value
        value = (registers[0] << 16) | registers[1]
        
        # Convert to signed integer if necessary
        if value & 0x80000000:
            value = value - 0x100000000
        
        # Convert to float (assuming 1/100 gram resolution)
        return value / 100.0


class ModbusPoller:
    """Async poller for continuous Modbus weight monitoring."""
    
    def __init__(self, modbus_client: ModbusWeightClient, poll_interval: float = 0.1):
        """
        Initialize poller.
        
        Args:
            modbus_client: ModbusWeightClient instance
            poll_interval: Time between polls (seconds)
        """
        self.client = modbus_client
        self.poll_interval = poll_interval
        self.running = False
        self.latest_weights = {}
    
    async def start_polling(self) -> None:
        """Start continuous weight polling."""
        self.running = True
        logger.info("Starting Modbus weight polling...")
        
        while self.running:
            try:
                self.latest_weights = await self.client.read_all_weights()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error during polling: {e}")
                await asyncio.sleep(self.poll_interval)
    
    def stop_polling(self) -> None:
        """Stop continuous polling."""
        self.running = False
        logger.info("Stopped Modbus weight polling")
    
    def get_latest_weight(self, bin_id: str) -> Optional[float]:
        """Get most recent weight reading for a bin."""
        return self.latest_weights.get(bin_id)
