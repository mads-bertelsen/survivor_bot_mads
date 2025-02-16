# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from tilthenightends import Levelup, LevelupOptions, Vector, Team, Towards


RNG = np.random.default_rng(seed=12)


def get_pickup_positions(pickups, pickup="chicken"):
    if pickup in pickups:
        x_array = pickups[pickup].x
        y_array = pickups[pickup].y

        coordinate_list = []
        for x, y in zip(x_array, y_array):
            coordinate_list.append( (x,y) )

    else:
        coordinate_list = None

    return coordinate_list

def get_chicken_positions(pickups):
    return get_pickup_positions(pickups, pickup="chicken")

def get_treasure_positions(pickups):
    return get_pickup_positions(pickups, pickup="treasure")


def dist_coords(a, b):
    return np.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )


def find_closest_coord_list(coord_list, me):

    if len(coord_list) == 1:
        return coord_list[0]

    dists = []
    for coord in coord_list:
        dists.append(dist_coords(a=coord, b=(me.x, me.y)) )

    index = np.argmin(dists)

    return coord_list[index]


def get_all_monsters(monsters):
    monster_x_position_arrays = []
    monster_y_position_arrays = []
    for monster_type, monster_info in monsters.items():
        monster_x_position_arrays.append(monster_info.x)
        monster_y_position_arrays.append(monster_info.y)

    if len(monster_x_position_arrays) > 0:
        monster_x_array = np.concatenate(monster_x_position_arrays)
        monster_y_array = np.concatenate(monster_y_position_arrays)

        return monster_x_array, monster_y_array

    else:
        return np.empty((0,)), np.empty((0,))

def get_all_monsters_type(monsters, given_type=None):
    monster_x_position_arrays = []
    monster_y_position_arrays = []
    for monster_type, monster_info in monsters.items():
        if given_type is not None:
            if not given_type == monster_type:
                continue

        monster_x_position_arrays.append(monster_info.x)
        monster_y_position_arrays.append(monster_info.y)

    if len(monster_x_position_arrays) > 0:
        monster_x_array = np.concatenate(monster_x_position_arrays)
        monster_y_array = np.concatenate(monster_y_position_arrays)

        return monster_x_array, monster_y_array

    else:
        return np.empty((0,)), np.empty((0,))


def pick_those_in_circle(x0, y0, x_array, y_array, radius=100):
    distances = np.sqrt((x_array - x0) ** 2 + (y_array - y0) ** 2)

    # Select points within distance d
    mask = distances <= radius
    filtered_x = x_array[mask]
    filtered_y = y_array[mask]

    return filtered_x, filtered_y


def monsters_in_directions(x0, y0, x_array, y_array, n_bins):
    # Step 1: Compute angles relative to the given point
    angles = np.arctan2(y_array - y0, x_array - x0)  # Returns angles in radians in range [-π, π]

    # Step 2: Define bins
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)  # Evenly spaced angle bins
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers

    # Step 3: Convert bin centers to unit vectors
    unit_vectors = np.column_stack((np.cos(bin_centers), np.sin(bin_centers)))

    # Step 3: Count points in each bin
    counts, _ = np.histogram(angles, bins=bins)

    """
    for i in range(n_bins):
        print(f"Bin {i + 1}: Angle [{180/np.pi*bins[i]:.2f}, {180/np.pi*bins[i + 1]:.2f}], Count = {counts[i]}, "
              f"Unit Vector = ({unit_vectors[i, 0]:.2f}, {unit_vectors[i, 1]:.2f})")
    """

    return counts, unit_vectors


def find_closest(x0, y0, x_positions, y_positions):

    if len(x_positions) == 0:
        return None, None

    # Compute squared Euclidean distances (avoiding sqrt for efficiency)
    distances_sq = (x_positions - x0) ** 2 + (y_positions - y0) ** 2

    # Find the index of the closest point
    closest_index = np.argmin(distances_sq)

    # Extract closest point
    closest_x = x_positions[closest_index]
    closest_y = y_positions[closest_index]

    return closest_x, closest_y


def check_emergency(me, players, monster_x, monster_y, radius=100):
    hero_names = players.keys()

    monsters_near_x, monsters_near_y = pick_those_in_circle(me.x, me.y, monster_x, monster_y, radius=radius)
    closest_x, closest_y = find_closest(me.x, me.y, monsters_near_x, monsters_near_y)

    if closest_x is not None:
        dist_closest = dist_coords((me.x, me.y), (closest_x, closest_y))
    else:
        dist_closest = None

    # Emergency response
    if dist_closest is not None:
        if dist_closest < radius:
            #print("Emergency, me: ", (me.x, me.y), " enemy: ", (closest_x, closest_y), "vector:",  (me.x - closest_x, me.y - closest_y))
            return Vector(me.x - closest_x, me.y - closest_y)
        else:
            return None

    else:
        return None



class Hero:
    def __init__(self, name: str, leader_order: list[str]):
        self.hero = name
        self.next_turn = 5.0
        self.vector = Vector(1, 1)
        self.leader_order = leader_order

        self.movement = Vector(1, 1)

    def current_leader(self, players):
        for name in self.leader_order:
            if players[name].alive:
                return name

    def calculate_movement(self, me, pickups, monster_x, monster_y):
        chicken_list = get_chicken_positions(pickups)
        treasure_list = get_treasure_positions(pickups)

        dist_string = str(300)
        monsters_near_x, monsters_near_y = pick_those_in_circle(me.x, me.y, monster_x, monster_y, radius=250)

        if len(monsters_near_x) < 10:
            dist_string = str(400)
            monsters_near_x, monsters_near_y = pick_those_in_circle(me.x, me.y, monster_x, monster_y, radius=400)

        slice_counts, slice_vectors = monsters_in_directions(me.x, me.y, monsters_near_x, monsters_near_y, n_bins=12)

        most_enemy_direction = slice_vectors[np.argmax(slice_counts)]
        fewest_enemy_direction = slice_vectors[np.argmax(slice_counts)]

        if len(monsters_near_x) > 12:
            enemy_direction = fewest_enemy_direction
            direction_string = "fewest " + dist_string
        else:
            enemy_direction = most_enemy_direction
            direction_string = "most " + dist_string

        # treasure_list = None
        if treasure_list is None:
            self.movement = Vector(enemy_direction[0], enemy_direction[1])
            #print("enemy: " + direction_string, enemy_direction)
        else:
            closest = find_closest_coord_list(treasure_list, me)
            #print("towards tressure at (", closest, ") with direction", me.x - closest[0], me.y - closest[1])
            self.movement = Vector(closest[0] - me.x, closest[1] - me.y)

    def run(self, t, dt, monsters, players, pickups) -> Vector | Towards | None:
        leader_name = self.current_leader(players)
        # Base movement on whether the Hero is the current leader or not.
        if self.hero == leader_name:
            # Insert leader logic here.

            # Emergency movement
            monster_x, monster_y = get_all_monsters(monsters)
            output = check_emergency(players[leader_name], players, monster_x, monster_y, radius=90)
            if output is not None:
                return output

            if "volcano" in monsters.keys():
                #print("oh no, volcano!")
                monster_x_volcano, monster_y_volcano = get_all_monsters_type(monsters, given_type="volcano")

                output = check_emergency(players[leader_name], players, monster_x_volcano, monster_y_volcano, radius=250)
                if output is not None:
                    return output

            if t > self.next_turn:
                self.calculate_movement(players[leader_name], pickups, monster_x, monster_y)
                #print(monsters.keys())
                self.next_turn += 1.3

            return self.movement
        else:
            # Follower logic.
            return Towards(players[leader_name].x, players[leader_name].y)


class Brain:
    def __init__(self):
        self.level = 0
        self.order = ["alaric", "evelyn", "cedric", "seraphina", "kaelen"]

    def levelup(self, t: float, info: dict, players: dict) -> Levelup:
        hero_to_level = self.order[self.level % len(self.order)]
        self.level += 1

        if self.level < 40:
            return Levelup("seraphina", LevelupOptions.weapon_cooldown)

        if self.level < 60:
            return Levelup("seraphina", LevelupOptions.weapon_size)

        if self.level < 90:
            return Levelup("seraphina", LevelupOptions.weapon_longevity)

        wpn_cd_lvl = players[hero_to_level].levels["weapon_cooldown"]

        # Attack speed up to 20
        # Then dump in damage
        if wpn_cd_lvl < 20:
            return Levelup(hero_to_level, LevelupOptions.weapon_cooldown)
        else:
            dmg_lvl = players[hero_to_level].levels["weapon_damage"]
            radius_lvl = players[hero_to_level].levels["weapon_radius"]
            longevity_lvl = players[hero_to_level].levels["weapon_longevity"]

            category_names = [LevelupOptions.weapon_damage, LevelupOptions.weapon_size, LevelupOptions.weapon_longevity]
            category_levels = [dmg_lvl, radius_lvl, longevity_lvl]
            min_level = min(category_levels)

            for level, index in enumerate(category_levels):
                if level == min_level:
                    return Levelup(hero_to_level, category_names[index])

            # This return value should not be reached, but is simply here for good measure.
            return Levelup(hero_to_level, LevelupOptions.weapon_damage)

order = ["alaric", "evelyn", "cedric", "seraphina", "kaelen"]

team = Team(
    players=[
        Hero(name="alaric", leader_order=order),
        Hero(name="evelyn", leader_order=order),
        Hero(name="cedric", leader_order=order),
        Hero(name="seraphina", leader_order=order),
        Hero(name="kaelen", leader_order=order),
    ],
    strategist=Brain(),
)
