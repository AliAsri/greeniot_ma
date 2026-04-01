"""
GreenIoT-MA — Générateur de capteurs IoT
==========================================
Simule 4 types de capteurs produisant des séries temporelles réalistes :
- SolarSensor  : panneaux solaires (production kWh, irradiance)
- ServerSensor : serveurs (conso CPU/RAM/kWh)
- CoolingSensor: refroidissement (température entrée/sortie air, PUE)
- BatterySensor: batterie (SOC, charge/décharge)

Les valeurs suivent des patterns circadiens réalistes avec bruit gaussien.
"""

import random
import time
import json
import math
from datetime import datetime
import numpy as np


class SolarSensor:
    """Simule un panneau solaire — production suit une courbe sinusoïdale diurne."""

    def __init__(self, sensor_id: str, peak_kw: float = 500.0):
        self.sensor_id = sensor_id
        self.peak_kw = peak_kw

    def read(self) -> dict:
        hour = datetime.now().hour + datetime.now().minute / 60

        # Courbe solaire : pic à 13h, nulle la nuit
        irradiance = max(0, math.sin(math.pi * (hour - 6) / 12))
        production = self.peak_kw * irradiance * random.gauss(1.0, 0.03)

        return {
            "sensor_id": self.sensor_id,
            "type": "solar",
            "timestamp": datetime.utcnow().isoformat(),
            "production_kw": round(max(0, production), 2),
            "irradiance_wm2": round(max(0, irradiance * 1000 * random.gauss(1.0, 0.02)), 1),
            "panel_temp_c": round(25 + irradiance * 20 + random.gauss(0, 1.5), 1),
        }


class ServerSensor:
    """Simule un rack de serveurs — charge variable selon le moment de la journée."""

    def __init__(self, sensor_id: str, max_kw: float = 80.0):
        self.sensor_id = sensor_id
        self.max_kw = max_kw
        self._base_load = random.uniform(0.3, 0.6)

    def read(self) -> dict:
        hour = datetime.now().hour

        # Charge plus élevée aux heures de bureau
        if 8 <= hour <= 20:
            day_factor = 0.5 + 0.4 * math.sin(math.pi * max(0, hour - 8) / 10)
        else:
            day_factor = 0.3

        cpu = self._base_load * day_factor * 100 + random.gauss(0, 3)
        ram = 40 + day_factor * 40 + random.gauss(0, 2)
        kw = self.max_kw * (cpu / 100) * random.gauss(1.0, 0.02)

        return {
            "sensor_id": self.sensor_id,
            "type": "server",
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_pct": round(min(100, max(0, cpu)), 1),
            "ram_pct": round(min(100, max(0, ram)), 1),
            "power_kw": round(max(0, kw), 2),
            "temp_c": round(35 + (cpu / 100) * 25 + random.gauss(0, 1), 1),
        }


class CoolingSensor:
    """Simule le système de refroidissement — PUE entre 1.2 et 2.0."""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id

    def read(self) -> dict:
        it_load = random.uniform(200, 600)  # kW charge IT
        pue = random.gauss(1.45, 0.08)      # PUE réaliste pour un DC marocain

        return {
            "sensor_id": self.sensor_id,
            "type": "cooling",
            "timestamp": datetime.utcnow().isoformat(),
            "inlet_temp_c": round(random.gauss(22, 1.5), 1),
            "outlet_temp_c": round(random.gauss(35, 2), 1),
            "it_load_kw": round(it_load, 1),
            "total_power_kw": round(it_load * pue, 1),
            "pue": round(pue, 3),
        }


class BatterySensor:
    """Simule une batterie de stockage — SOC, charge/décharge."""

    def __init__(self, sensor_id: str, capacity_kwh: float = 1000.0):
        self.sensor_id = sensor_id
        self.capacity_kwh = capacity_kwh
        self._soc = random.uniform(0.4, 0.8)

    def read(self) -> dict:
        hour = datetime.now().hour

        # Charge pendant les heures solaires, décharge la nuit
        if 10 <= hour <= 16:
            charge_rate = random.uniform(20, 80)  # kW charge
            self._soc = min(1.0, self._soc + charge_rate / self.capacity_kwh * 0.1)
        else:
            charge_rate = -random.uniform(10, 50)  # kW décharge
            self._soc = max(0.1, self._soc + charge_rate / self.capacity_kwh * 0.1)

        return {
            "sensor_id": self.sensor_id,
            "type": "battery",
            "timestamp": datetime.utcnow().isoformat(),
            "soc_pct": round(self._soc * 100, 1),
            "charge_rate_kw": round(charge_rate, 2),
            "voltage_v": round(48 * (0.9 + 0.1 * self._soc) + random.gauss(0, 0.5), 1),
            "temp_c": round(25 + abs(charge_rate) * 0.05 + random.gauss(0, 1), 1),
        }


def generate_stream(interval_sec: float = 5.0):
    """Générateur infini de mesures IoT. Utiliser dans le producer Kafka."""
    sensors = [
        SolarSensor("solar_dakhla_01", peak_kw=500),
        SolarSensor("solar_dakhla_02", peak_kw=480),
        ServerSensor("rack_A1", max_kw=80),
        ServerSensor("rack_A2", max_kw=75),
        ServerSensor("rack_B1", max_kw=90),
        CoolingSensor("cooling_unit_01"),
        BatterySensor("battery_01", capacity_kwh=1000),
    ]

    while True:
        for s in sensors:
            yield s.read()
        time.sleep(interval_sec)


if __name__ == "__main__":
    # Test : affiche 10 mesures
    print("=== GreenIoT-MA — Test Sensor Simulator ===\n")
    sensors = [
        SolarSensor("solar_dakhla_01"),
        ServerSensor("rack_A1"),
        CoolingSensor("cooling_unit_01"),
        BatterySensor("battery_01"),
    ]
    for sensor in sensors:
        reading = sensor.read()
        print(f"[{reading['type'].upper():>8}] {reading['sensor_id']}")
        print(f"           {json.dumps(reading, indent=2)}\n")
