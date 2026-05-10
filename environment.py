import numpy as np
import random

# --- State Encoding ---
# Soil moisture levels
MOISTURE_LEVELS = ["Very Dry", "Dry", "Optimal", "Moist", "Saturated"]
# Weather conditions
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy"]
# Time slots
TIME_SLOTS = ["Morning", "Afternoon", "Evening"]

# Action definitions
ACTIONS = ["No Water", "Water Low", "Water High"]
ACTION_NO_WATER   = 0
ACTION_WATER_LOW  = 1
ACTION_WATER_HIGH = 2

# Water added per action (in moisture units)
WATER_AMOUNT = {ACTION_NO_WATER: 0, ACTION_WATER_LOW: 1, ACTION_WATER_HIGH: 2}

# Natural evaporation per time slot per weather
EVAPORATION = {
    "Sunny":  {"Morning": 1, "Afternoon": 2, "Evening": 1},
    "Cloudy": {"Morning": 0, "Afternoon": 1, "Evening": 0},
    "Rainy":  {"Morning": 0, "Afternoon": 0, "Evening": 0},
}

# Rain adds moisture
RAIN_BOOST = {"Sunny": 0, "Cloudy": 0, "Rainy": 2}


class IrrigationEnv:
    """
    Simulates a single crop field over time.
    State  : (moisture_index, weather_index, time_index)  -> int 0..44
    Action : 0=no_water, 1=water_low, 2=water_high
    Reward : based on crop health, water waste, drought, overwatering
    """

    def __init__(self):
        self.n_states  = len(MOISTURE_LEVELS) * len(WEATHER_CONDITIONS) * len(TIME_SLOTS)
        self.n_actions = len(ACTIONS)
        self.reset()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _encode_state(self):
        return (
            self.moisture_idx * len(WEATHER_CONDITIONS) * len(TIME_SLOTS)
            + self.weather_idx * len(TIME_SLOTS)
            + self.time_idx
        )

    def _next_weather(self):
        # Simple Markov-style transition: rain more likely to follow rain, etc.
        transitions = {
            0: [0.6, 0.3, 0.1],   # Sunny -> mostly sunny
            1: [0.3, 0.4, 0.3],   # Cloudy -> balanced
            2: [0.2, 0.3, 0.5],   # Rainy  -> likely stays rainy
        }
        return np.random.choice(3, p=transitions[self.weather_idx])

    def _compute_reward(self, action, old_moisture):
        reward = 0.0
        m = self.moisture_idx

        # Crop health is best when moisture is Optimal (2) or Moist (3)
        if m == 2:    reward += 10.0   # Optimal
        elif m == 3:  reward += 6.0    # Moist — acceptable
        elif m == 1:  reward += 1.0    # Dry — stressed
        elif m == 0:  reward -= 10.0   # Very Dry — drought penalty
        elif m == 4:  reward -= 8.0    # Saturated — root rot risk

        # Penalise watering when it was unnecessary
        weather = WEATHER_CONDITIONS[self.weather_idx]
        if action == ACTION_WATER_HIGH and weather == "Rainy":
            reward -= 6.0   # heavy water during rain = waste
        if action == ACTION_WATER_HIGH and old_moisture >= 3:
            reward -= 5.0   # already moist, no need
        if action == ACTION_WATER_LOW and old_moisture == 4:
            reward -= 3.0   # already saturated

        # Small cost for any watering (encourages efficiency)
        reward -= WATER_AMOUNT[action] * 0.5

        return reward

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

    def reset(self):
        self.moisture_idx = random.randint(1, 3)           # start Dry..Moist
        self.weather_idx  = random.randint(0, 2)
        self.time_idx     = 0                              # always start Morning
        self.step_count   = 0
        self.total_water  = 0
        self.crop_health  = 100.0
        return self._encode_state()

    def step(self, action):
        assert action in range(self.n_actions)
        old_moisture = self.moisture_idx

        weather = WEATHER_CONDITIONS[self.weather_idx]
        time    = TIME_SLOTS[self.time_idx]

        # Apply watering
        self.moisture_idx += WATER_AMOUNT[action]
        self.total_water  += WATER_AMOUNT[action]

        # Apply rain
        self.moisture_idx += RAIN_BOOST[weather]

        # Apply evaporation
        self.moisture_idx -= EVAPORATION[weather][time]

        # Clamp moisture
        self.moisture_idx = max(0, min(4, self.moisture_idx))

        # Advance time
        self.time_idx = (self.time_idx + 1) % len(TIME_SLOTS)
        if self.time_idx == 0:
            self.weather_idx = self._next_weather()

        reward = self._compute_reward(action, old_moisture)
        self.crop_health = max(0.0, min(100.0, self.crop_health + reward * 0.1))

        self.step_count += 1
        done = self.step_count >= 365   # one simulated year

        next_state = self._encode_state()
        return next_state, reward, done

    def get_state_label(self):
        return (
            f"Moisture={MOISTURE_LEVELS[self.moisture_idx]}, "
            f"Weather={WEATHER_CONDITIONS[self.weather_idx]}, "
            f"Time={TIME_SLOTS[self.time_idx]}"
        )
