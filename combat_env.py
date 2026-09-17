import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CombatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 300

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.enemy_pos = np.random.uniform(-5, 5)
        self.player_pos = np.random.uniform(-5, 5)
        self.enemy_hp = 100
        self.player_hp = 100
        self.steps_taken = 0
        self.player_attack_cooldown = 0

        observation = self._get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _get_observation(self):
        distance = abs(self.enemy_pos - self.player_pos)
        player_attacking = 1.0 if distance <= 1.5 else 0.0

        return np.array([
            self.enemy_pos / 10.0,
            self.player_pos / 10.0,
            distance / 20.0,
            self.enemy_hp / 100.0,
            self.player_hp / 100.0,
            player_attacking
        ], dtype=np.float32)

    def step(self, action):
        self.steps_taken += 1
        reward = -0.1
        dodged = False

        if abs(self.enemy_pos - self.player_pos) > 1.5:
            if self.player_pos < self.enemy_pos:
                self.player_pos += 1
            else:
                self.player_pos -= 1
        self.player_pos = np.clip(self.player_pos, -10, 10)

        distance_before = abs(self.enemy_pos - self.player_pos)

        if action == 0:
            if self.enemy_pos < self.player_pos:
                self.enemy_pos += 1
            else:
                self.enemy_pos -= 1
        elif action == 1:
            if self.enemy_pos < self.player_pos:
                self.enemy_pos -= 1
            else:
                self.enemy_pos += 1
        elif action == 3:
            dodged = True

        self.enemy_pos = np.clip(self.enemy_pos, -10, 10)

        distance = abs(self.enemy_pos - self.player_pos)

        if distance_before > 1.5:
            reward += 0.1 * (distance_before - distance)

        if action == 2 and distance <= 1.5:
            self.player_hp -= 10
            reward += 10

        player_in_range = distance <= 1.5
        if player_in_range and self.player_attack_cooldown <= 0:
            if not dodged:
                self.enemy_hp -= 10
                reward -= 15
            self.player_attack_cooldown = 3
        else:
            self.player_attack_cooldown = max(0, self.player_attack_cooldown - 1)

        terminated = self.enemy_hp <= 0 or self.player_hp <= 0
        truncated = self.steps_taken >= 200

        if terminated and self.player_hp <= 0 and self.enemy_hp > 0:
            reward += 50
        elif terminated and self.enemy_hp <= 0:
            reward -= 30

        if truncated and not terminated:
            reward -= 20

        observation = self._get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        import pygame

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Combat Arena")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Convert world positions (-10 to 10) to screen x-coordinates
        def world_to_screen_x(pos):
            return int((pos + 10) / 20 * self.screen_width)

        ground_y = 200

        # Draw the ground line
        pygame.draw.line(self.screen, (0, 0, 0), (0, ground_y), (self.screen_width, ground_y), 2)

        # Draw enemy (our agent) as a red circle
        enemy_x = world_to_screen_x(self.enemy_pos)
        pygame.draw.circle(self.screen, (200, 50, 50), (enemy_x, ground_y - 20), 20)

        # Draw player (scripted opponent) as a blue circle
        player_x = world_to_screen_x(self.player_pos)
        pygame.draw.circle(self.screen, (50, 50, 200), (player_x, ground_y - 20), 20)

        # Draw health bars above each fighter
        bar_width = 60
        pygame.draw.rect(self.screen, (0, 0, 0), (enemy_x - bar_width // 2, ground_y - 60, bar_width, 8), 1)
        pygame.draw.rect(self.screen, (200, 50, 50), (enemy_x - bar_width // 2, ground_y - 60, bar_width * (self.enemy_hp / 100), 8))

        pygame.draw.rect(self.screen, (0, 0, 0), (player_x - bar_width // 2, ground_y - 60, bar_width, 8), 1)
        pygame.draw.rect(self.screen, (50, 50, 200), (player_x - bar_width // 2, ground_y - 60, bar_width * (self.player_hp / 100), 8))

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()