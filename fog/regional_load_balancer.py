class RegionalLoadBalancer:

    def __init__(self, safe_zones, zone_capacity):
        self.safe_zones = safe_zones
        self.zone_capacity = zone_capacity

    def balance(self, drone_packets):

        zone_load = {z: 0 for z in self.safe_zones}

        total_people = sum(p["people_count"] for p in drone_packets)

        remaining_people = total_people

        # Sort zones by capacity (largest first)
        sorted_zones = sorted(self.zone_capacity.items(), key=lambda x: -x[1])

        for zone_id, capacity in sorted_zones:
            if remaining_people <= 0:
                break

            allocated = min(capacity, remaining_people)
            zone_load[zone_id] += allocated
            remaining_people -= allocated

        # Assign primary zone as most loaded one
        primary_zone = max(zone_load, key=zone_load.get)

        for packet in drone_packets:
            packet["safe_zone_id"] = primary_zone

        self.latest_zone_load = zone_load

        return drone_packets

    def compute_zone_load(self, packets):
        return getattr(self, "latest_zone_load", {})