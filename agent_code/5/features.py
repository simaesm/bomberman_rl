import torch
import numpy as np
from collections import deque
import heapq
import random
import bisect

STEP = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

DIRECTION = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}

MOVE = ["rechts", "links", "unten", "oben"]

ACTIONS = ['RIGHT', 'LEFT', 'DOWN', 'UP', 'WAIT', 'BOMB']
ACTIONS_INVERSE = {"RIGHT": 0, "LEFT": 1, "DOWN": 2, "UP": 3, "WAIT": 4, "BOMB": 5}

MAX_SEARCHING_DISTANCE = 20

MAX_FOUND_CRATE_POSITIONS = 3

MAX_FOUND_DEAD_ENDS = 2

MAX_CRATES = MAX_FOUND_DEAD_ENDS + MAX_FOUND_CRATE_POSITIONS


def state_to_features(self, game_state: dict) -> torch.tensor:

    max_len_coins = 9  # only the coins
    max_len_crates = 240

    # at the beginning and the end:
    if game_state is None:
        return None

    def possible_neighbors(pos):

        result = []
        for new_pos in (pos + STEP):
            if field[new_pos[0], new_pos[1]] == 0:
                result.append(new_pos.tolist())
        return result

    def bomb_effect(pos):

        destroyed_crates = 0
        for direction in STEP:
            for length in range(1, 4):
                beam = direction * length + pos
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break
                if (obj == 1) and future_explosion_map[beam[0], beam[1]] == 1:  # we will ge the crate destroyed
                    destroyed_crates += 1
        return destroyed_crates

    def fill_explosion_map(explosions, bombs, field):

        future_explosion_map = (np.copy(explosions) * -4) + 1  # -3 now exploding, 1 no bomb in reach
        for bomb in bombs:
            pos = np.array(bomb[0])
            timer = bomb[1] - 3  # the smaller, the more dangerous
            field[pos[0], pos[1]] = -2  # put the bombs in the field array as n obstackles

            for direction in STEP:
                for length in range(0, 4):
                    beam = direction * length + pos
                    obj = field[beam[0], beam[1]]
                    if obj == -1:
                        break
                    if future_explosion_map[beam[0], beam[1]] > timer:
                        future_explosion_map[beam[0], beam[1]] = timer

        return future_explosion_map

    def create_new_future_explosion_map(future_explosion_map, pos):
        new_future_explosion_map = np.copy(future_explosion_map)
        # nex turn -> each counter - 1
        new_future_explosion_map[new_future_explosion_map < 1] -= 1
        new_future_explosion_map[new_future_explosion_map < -3] = 1
        timer = 0

        for direction in STEP:
            for length in range(0, 4):
                beam = direction * length + pos
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break
                if new_future_explosion_map[beam[0], beam[1]] > timer:
                    new_future_explosion_map[beam[0], beam[1]] = timer

        return new_future_explosion_map

    def certain_death(pos, future_explosion_map, turns=0, forbidden_fields=None):
        q = deque()

        visited = []

        if forbidden_fields is not None:
            for forbidden_pos in forbidden_fields:
                visited.append(forbidden_pos)

        q.append((pos.tolist(), turns))
        while len(q):
            pos, turns = q.popleft()

            # The bomb did explode
            if turns > 4:
                break

            if pos in visited:
                continue

            # field = -3 exploding the coming turn, += 1 for each additional turn

            # YOU DIED
            if turns - 1 - future_explosion_map[pos[0], pos[1]] == 3:
                continue

            # We found a way out
            if future_explosion_map[pos[0], pos[1]] == 1:
                return False

            visited.append(pos)
            for neighbor in possible_neighbors(pos):
                q.append((neighbor, turns + 1))

        return True

    # save the position of maverick as ndarray
    player_pos = np.array(game_state["self"][3])

    # save the known positions of the coins
    coins = np.array(game_state["coins"])
    coins_list = coins.tolist()
    number_of_coins = len(coins)

    # save the positions of the crates
    field = np.array(game_state["field"])
    explosions = np.array(game_state["explosion_map"])
    bombs = game_state["bombs"]

    crates = np.argwhere(field == 1)
    number_of_crates = len(crates)
    future_explosion_map = fill_explosion_map(explosions, bombs, field)

    # save the posisition of the other agents
    others = []
    for agent in game_state["others"]:
        pos = agent[3]
        others.append(pos)

        field[pos[0], pos[1]] == 2  # append the opponents as new obstacle to the field

    others = np.array(others)
    others_list = others.tolist()

    # in the last gamestate during training we have no more coins
    # thus the algorithm would crash if we did not create a fake coin entry somewhere. Append would not wor othervise
    # 0,0 is always in the border and thus not reachable
    if number_of_coins == 0:
        coins = np.zeros((max_len_coins, 2))

    possible_next_pos = possible_neighbors(player_pos)

    inv_coins = [[] for _ in range(4)]
    inv_crate_distances = [[] for _ in range(4)]
    crate_points = [[] for _ in range(4)]
    inv_opponents = [[] for _ in range(4)]

    coin_distances_after_step = np.empty((4, max_len_coins))
    crate_distances_after_step = np.empty((4, MAX_CRATES))
    opponent_distances_after_step = np.empty((4, 4))  # MAX_OPPONENTS = 4

    expected_destructions_after_step = np.zeros((4, MAX_CRATES))

    coin_distances_after_step.fill(np.inf)
    crate_distances_after_step.fill(np.inf)
    opponent_distances_after_step.fill(np.inf)

    visited = [player_pos.tolist()]

    q = []
    for pos in (player_pos + STEP):
        pos = pos.tolist()  # needed for the "in" command to work
        # initialization of the search algorithm
        x = pos[0] - player_pos[0]
        y = pos[1] - player_pos[1]
        heapq.heappush(q, (1, pos, DIRECTION[(x, y)]))

    number_of_found_crate_positions = np.zeros(4)
    number_of_found_dead_ends = np.zeros(4)
    number_of_found_coins = np.zeros(4)
    number_of_found_opponents = np.zeros(4)

    found_one = False
    skipped = [False, False, False, False]

    while len(q) != 0:

        distance, pos, direction = heapq.heappop(q)

        if (distance > MAX_SEARCHING_DISTANCE) and (found_one == True):
            break

        if pos in visited:
            continue

        visited.append(pos)

        if distance == 1:
            # Safely blown up
            if future_explosion_map[pos[0], pos[1]] == -2:
                crate_points[direction] = np.zeros(MAX_CRATES)
                placebo1 = np.zeros(MAX_CRATES)
                placebo1.fill(-2)
                placebo2 = np.zeros(max_len_coins)
                placebo2.fill(-2)
                placebo3 = np.zeros(4)
                placebo3.fill(-2)
                inv_crate_distances[direction] = np.copy(placebo1)
                inv_coins[direction] = np.copy(placebo2)
                inv_opponents[direction] = np.copy(placebo3)

                skipped[direction] = True
                continue

            if pos not in possible_next_pos:
                # we are walking against a wall or a crate
                crate_points[direction] = np.zeros(MAX_CRATES)
                placebo1 = np.zeros(MAX_CRATES)
                placebo1.fill(-1)
                placebo2 = np.zeros(max_len_coins)
                placebo2.fill(-1)
                placebo3 = np.zeros(4)
                placebo3.fill(-1)
                inv_crate_distances[direction] = np.copy(placebo1)
                inv_coins[direction] = np.copy(placebo2)
                inv_opponents[direction] = np.copy(placebo3)

                skipped[direction] = True
                continue

        # coins
        is_coin = pos in coins_list  # check if pos is in coins -> we reached a coin
        if is_coin:
            coin_distances_after_step[direction][int(number_of_found_coins[direction])] = distance
            number_of_found_coins[direction] += 1
        if is_coin and not found_one:
            found_one = True

        # opponents
        if (number_of_found_opponents[direction] < 4):
            for possible_opponent in (pos + STEP):
                if (possible_opponent.tolist() in others_list):
                    # one of the neighboring fields is a free crate
                    index_opponents = int(number_of_found_opponents[direction])
                    opponent_distances_after_step[direction][index_opponents] = distance

                    number_of_found_opponents[direction] += 1
                    # Here found one should not be set to True!
                    break

        neighbors = possible_neighbors(pos)

        # visit all neighbors
        ways_out = 0
        for node in neighbors:
            ways_out += 1
            if (distance + 1) <= 3 and (future_explosion_map[node[0], node[1]] != 1):
                heapq.heappush(q, (distance + 1, node, direction))
            heapq.heappush(q, (distance + 1, node, direction))

        # crates
        if future_explosion_map[pos[0], pos[1]] != 1:  # this position is already used -> dont drop a bomb
            continue

        dead_end = False
        if (ways_out == 1) and (number_of_found_dead_ends[direction] < MAX_FOUND_DEAD_ENDS):
            # we found a unused dead end, this might be a good bomb position
            index_crates = int(number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
            crate_distances_after_step[direction][index_crates] = distance
            expected_destructions_after_step[direction][index_crates] = bomb_effect(pos)

            dead_end = True
            number_of_found_dead_ends[direction] += 1
            found_one = True

        if (number_of_found_crate_positions[direction] < MAX_FOUND_CRATE_POSITIONS) and not dead_end:
            for possible_crate in (pos + STEP):
                if field[possible_crate[0], possible_crate[1]] == 1 and (
                        future_explosion_map[possible_crate[0], possible_crate[1]] == 1):
                    index_crates = int(
                        number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
                    crate_distances_after_step[direction][index_crates] = distance
                    expected_destructions_after_step[direction][index_crates] = bomb_effect(pos)

                    number_of_found_crate_positions[direction] += 1
                    found_one = True
                    break

    for direction in range(4):
        if skipped[direction]:
            continue
        inv_coins[direction] = 1 / np.array(coin_distances_after_step[direction])

        inv_opponents[direction] = 1 / np.array(opponent_distances_after_step[direction])

        inv_crate_distances[direction] = 1 / np.array(crate_distances_after_step[direction])

        crate_points[direction] = np.array(expected_destructions_after_step[direction])

    inv_crate_distances = np.array(inv_crate_distances)

    crate_points = np.array(crate_points)

    inv_coins = np.array(inv_coins)

    inv_opponents = np.array(inv_opponents)


    features = []
    # append the coins feature
    features = np.append(features, np.max(inv_coins, axis=1))

    # append the crates features
    features = np.append(features, np.max(inv_crate_distances * crate_points, axis=1))

    # is it senceful to drop a bomb here?
    neighboring_chest = False
    neighboring_opponent = False
    if future_explosion_map[player_pos[0], player_pos[1]] == 1:
        for pos in player_pos + STEP:
            if (field[pos[0], pos[1]] == 1) and (future_explosion_map[pos[0], pos[1]] == 1):  # free crate
                neighboring_chest = True
            if pos.tolist() in others_list:  # oppontent
                neighboring_opponent = True

    # Points for opponent and crates
    if neighboring_opponent:
        bomb_here = 5 + bomb_effect(player_pos)

    # Points only for the crates
    elif neighboring_chest:
        bomb_here = bomb_effect(player_pos)

    # We do not get anything if we drop a bomb here
    if not neighboring_chest and not neighboring_opponent:
        bomb_here = -1

    # We do not have our bomb
    if not game_state["self"][2]:
        bomb_here = -1

    new_future_explosion_map = create_new_future_explosion_map(future_explosion_map, player_pos)

    # We would kill ourself if we drop a bomb here
    if certain_death(player_pos, new_future_explosion_map):
        # print("Sicherer Tod")
        bomb_here = -1

    features = np.append(features, bomb_here)

    features = np.append(features, -(future_explosion_map[player_pos[0], player_pos[1]] - 1))

    running = np.zeros(4)
    # no bomb at our position
    if future_explosion_map[player_pos[0], player_pos[1]] == 1:
        features = np.append(features, running)
    else:
        direction = -1
        for pos in player_pos + STEP:
            direction += 1
            if certain_death(pos, future_explosion_map, turns=1, forbidden_fields=[player_pos.tolist()]):
                running[direction] = -1
            elif ((direction == np.argmax(np.max(inv_opponents, axis=1)))
                  and ((np.max(inv_opponents, axis=1) != 0).all())
                  and ((1 / np.max(inv_opponents, axis=1)) < 5).any()):
                running[direction] = -0.5
            if field[pos[0], pos[1]] != 0:
                running[direction] = -1

        features = np.append(features, running)

    danger = np.zeros(4)
    if future_explosion_map[player_pos[0], player_pos[1]] == 1:  # current position is save
        dim = 0
        for pos in player_pos + STEP:
            if future_explosion_map[pos[0], pos[1]] == -3:
                danger[dim] = -1
            dim += 1
    features = np.append(features, danger)

    # append the opponents feature
    features = np.append(features, np.max(inv_opponents, axis=1))

    # append the remaining positive total reward
    features = np.append(features, number_of_coins + 1 / 3 * number_of_crates)

    # needed for the rulebased version
    self.features = features

    # needed for the crate features
    self.destroyed_crates = self.bomb_buffer
    self.bomb_buffer = bomb_effect(player_pos)

    # crate a torch tensor that can be returned from the features
    features = torch.from_numpy(features).float()

    return features.unsqueeze(0)