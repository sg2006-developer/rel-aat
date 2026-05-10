import numpy as np
import pickle


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Q-table shape: (n_states, n_actions)
    Update rule  : Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',.) - Q(s,a)]
    Policy       : epsilon-greedy — explore early, exploit later
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,       # learning rate
        gamma=0.95,      # discount factor
        epsilon=1.0,     # initial exploration rate
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.n_states     = n_states
        self.n_actions    = n_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialise Q-table to small random values to break symmetry
        self.q_table = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))

    # ------------------------------------------------------------------ #
    #  Policy                                                              #
    # ------------------------------------------------------------------ #

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        return int(np.argmax(self.q_table[state]))     # exploit

    def choose_action_greedy(self, state):
        """Pure greedy — used during evaluation (no exploration)."""
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------ #
    #  Learning                                                            #
    # ------------------------------------------------------------------ #

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"  Q-table saved to {path}")

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        self.epsilon = self.epsilon_min   # pure exploit after loading
        print(f"  Q-table loaded from {path}")

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def stats(self):
        return {
            "epsilon":    round(self.epsilon, 4),
            "q_min":      round(float(self.q_table.min()), 3),
            "q_max":      round(float(self.q_table.max()), 3),
            "q_mean":     round(float(self.q_table.mean()), 3),
        }
