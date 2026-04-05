# ═══════════════════════════════════════════════════════════════════════════════
# Fruit Ninja Clone — OpenCV + MediaPipe + pygame
# ═══════════════════════════════════════════════════════════════════════════════
# Install dependencies:
#   pip install opencv-python mediapipe numpy
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOMISE YOUR GAME HERE
# ═══════════════════════════════════════════════════════════════════════════════
SWIPE_THRESHOLD = 2       # higher = less sensitive, lower = triggers easier
GRAVITY = 0.8             # higher = fruits arc more steeply
SPAWN_RATE = 55             # frames between fruit spawns, lower = harder
TRAIL_LENGTH = 6            # number of trail points behind fingertip
TRAIL_COLOR = (255, 255, 255)  # RGB color of the swipe trail (OpenCV: BGR stored, converted on draw)
LIVES = 3                   # starting lives
BOMB_EVERY_N = 5            # every Nth fruit is a bomb

# Window / display
WIN_W, WIN_H = 640, 360
FPS_TARGET = 30

# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Particle:
    """A single juice/flesh particle spawned on slice."""
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        # Random velocity in all directions
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 9)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color           # (B, G, R)
        self.lifespan = 18           # total frames alive
        self.age = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3               # slight gravity on particles
        self.age += 1

    def is_dead(self):
        return self.age >= self.lifespan

    def draw(self, frame):
        # Radius shrinks as particle ages
        progress = self.age / self.lifespan
        radius = max(1, int(5 * (1 - progress)))
        alpha = 1.0 - progress
        # Blend particle color with background using alpha simulation via overlay
        cx, cy = int(self.x), int(self.y)
        if 0 <= cx < WIN_W and 0 <= cy < WIN_H:
            # Draw a filled circle; we overlay with lower intensity as it fades
            faded_color = tuple(int(c * alpha) for c in self.color)
            cv2.circle(frame, (cx, cy), radius, faded_color, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# HALF-CIRCLE SLICE HALVES
# ═══════════════════════════════════════════════════════════════════════════════

class SliceHalf:
    """One half of a sliced fruit that flies apart."""
    def __init__(self, x, y, radius, color, outline_color, direction):
        self.x = float(x)
        self.y = float(y)
        self.radius = radius
        self.color = color
        self.outline_color = outline_color
        self.direction = direction   # -1 = left, +1 = right
        self.vx = direction * random.uniform(3, 6)
        self.vy = -random.uniform(4, 7)
        self.lifespan = 20
        self.age = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.4               # gravity on halves
        self.age += 1

    def is_dead(self):
        return self.age >= self.lifespan

    def draw(self, frame):
        progress = self.age / self.lifespan
        alpha = 1.0 - progress
        cx, cy = int(self.x), int(self.y)
        if cx < -100 or cx > WIN_W + 100 or cy < -100 or cy > WIN_H + 100:
            return
        # Draw half-circle using ellipse with specific start/end angles
        # Left half: 90° to 270°; Right half: 270° to 90°
        start_angle = 90 if self.direction == -1 else 270
        end_angle = 270 if self.direction == -1 else 450  # 450 = 90 wrapped
        faded_fill = tuple(int(c * alpha) for c in self.color)
        faded_outline = tuple(int(c * alpha) for c in self.outline_color)
        cv2.ellipse(frame, (cx, cy), (self.radius, self.radius),
                    0, start_angle, end_angle, faded_fill, -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (self.radius, self.radius),
                    0, start_angle, end_angle, faded_outline, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# FRUIT / BOMB CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Fruit:
    """Represents a fruit or bomb flying across the screen."""

    TYPES = ['watermelon', 'orange', 'blueberry']

    def __init__(self, fruit_type=None, is_bomb=False):
        self.is_bomb = is_bomb
        self.fruit_type = fruit_type if fruit_type else random.choice(self.TYPES)
        self.is_sliced = False
        self.slice_timer = 0         # countdown after slicing before removal
        self.spawn_side = random.choice(['left', 'right'])
        self.spark_angle = 0.0       # for animated bomb fuse spark

        # Set radius and colors by type
        if self.is_bomb:
            self.radius = 38
            self.color = (50, 50, 60)          # dark grey fill (BGR)
            self.outline_color = (30, 30, 40)
            self.inner_color = (50, 50, 60)
        elif self.fruit_type == 'watermelon':
            self.radius = 45
            self.color = (30, 120, 30)         # dark green outline (BGR)
            self.outline_color = (20, 80, 20)
            self.inner_color = (60, 60, 200)   # bright red flesh (BGR)
        elif self.fruit_type == 'orange':
            self.radius = 35
            self.color = (0, 140, 255)         # orange (BGR)
            self.outline_color = (0, 100, 200)
            self.inner_color = (0, 165, 255)
        else:  # blueberry
            self.radius = 25
            self.color = (150, 50, 30)         # deep blue (BGR)
            self.outline_color = (120, 30, 20)
            self.inner_color = (170, 80, 50)

        # Spawn position — random Y in playable band
        y = random.randint(int(WIN_H * 0.3), int(WIN_H * 0.8))
        if self.spawn_side == 'left':
            self.x = float(-self.radius)
            self.vx = random.uniform(6, 12)
        else:
            self.x = float(WIN_W + self.radius)
            self.vx = -random.uniform(6, 12)

        self.y = float(y)
        # Negative vy so fruit launches upward initially
        self.vy = -random.uniform(10, 16)

    def update(self):
        """Apply physics each frame."""
        if not self.is_sliced:
            self.vy += GRAVITY       # gravity pulls fruit downward
            self.x += self.vx
            self.y += self.vy
        # Animate bomb spark
        self.spark_angle += 0.3

    def is_off_screen(self):
        """Check if fruit has left the visible area."""
        return (self.y > WIN_H + self.radius + 20 or
                self.x < -self.radius - 50 or
                self.x > WIN_W + self.radius + 50)

    def fell_off_bottom(self):
        """Check specifically if it dropped below without being sliced."""
        return not self.is_sliced and self.y > WIN_H + self.radius + 20

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        r = self.radius

        if self.is_bomb:
            self._draw_bomb(frame, cx, cy, r)
        elif self.fruit_type == 'watermelon':
            self._draw_watermelon(frame, cx, cy, r)
        elif self.fruit_type == 'orange':
            self._draw_orange(frame, cx, cy, r)
        else:
            self._draw_blueberry(frame, cx, cy, r)

    def _draw_watermelon(self, frame, cx, cy, r):
        # Green rind
        cv2.circle(frame, (cx, cy), r, self.color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, self.outline_color, 3, cv2.LINE_AA)
        # Red flesh — slightly smaller inner circle
        inner_r = int(r * 0.78)
        cv2.circle(frame, (cx, cy), inner_r, self.inner_color, -1, cv2.LINE_AA)
        # Seeds — small black dots arranged in a fan pattern
        for i in range(6):
            angle = math.radians(i * 50 - 120)
            sx = int(cx + inner_r * 0.55 * math.cos(angle))
            sy = int(cy + inner_r * 0.55 * math.sin(angle))
            cv2.circle(frame, (sx, sy), 3, (10, 10, 10), -1, cv2.LINE_AA)

    def _draw_orange(self, frame, cx, cy, r):
        # Orange body
        cv2.circle(frame, (cx, cy), r, self.color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, self.outline_color, 3, cv2.LINE_AA)
        # White segment lines radiating from center
        for i in range(8):
            angle = math.radians(i * 45)
            ex = int(cx + r * 0.85 * math.cos(angle))
            ey = int(cy + r * 0.85 * math.sin(angle))
            cv2.line(frame, (cx, cy), (ex, ey), (220, 220, 220), 1, cv2.LINE_AA)
        # Small white center dot
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_blueberry(self, frame, cx, cy, r):
        # Deep blue body
        cv2.circle(frame, (cx, cy), r, self.color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, self.outline_color, 2, cv2.LINE_AA)
        # White shine dot — offset upper-left
        shine_x = int(cx - r * 0.3)
        shine_y = int(cy - r * 0.3)
        cv2.circle(frame, (shine_x, shine_y), max(3, r // 5), (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_bomb(self, frame, cx, cy, r):
        # Dark grey body
        cv2.circle(frame, (cx, cy), r, self.color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, (80, 80, 90), 3, cv2.LINE_AA)
        # Highlight ring
        cv2.circle(frame, (cx - 8, cy - 8), r // 3, (80, 80, 90), 2, cv2.LINE_AA)
        # Fuse — short orange line at the top
        fuse_start = (cx, cy - r)
        fuse_end = (cx + 8, cy - r - 18)
        cv2.line(frame, fuse_start, fuse_end, (0, 140, 255), 3, cv2.LINE_AA)
        # Animated spark at fuse tip — small pulsing dot
        spark_offset = int(3 * math.sin(self.spark_angle))
        spark_x = fuse_end[0] + spark_offset
        spark_y = fuse_end[1] - 2
        cv2.circle(frame, (spark_x, spark_y), 5, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (spark_x, spark_y), 3, (0, 255, 255), -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# HAND TRACKER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class HandTracker:
    MP_W, MP_H = 480, 270  # better accuracy than 320x180

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.tip_history = []
        self.last_tip = None
        self._frame_skip = 0
        self._cached_tip = (None, None, 0.0)

        
        self.smooth_tip = None
        self.alpha = 0.7  # higher = smoother but slightly laggy

    def process(self, frame_rgb):
        self._frame_skip += 1

        if self._frame_skip % 2 == 0:
            small = cv2.resize(frame_rgb, (self.MP_W, self.MP_H),
                               interpolation=cv2.INTER_LINEAR)

            results = self.hands.process(small)

            tip_x, tip_y, velocity = None, None, 0.0

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark[8]

                raw_x = int(lm.x * WIN_W)
                raw_y = int(lm.y * WIN_H)

                # 🔥 SMOOTHING (main upgrade)
                if self.smooth_tip is None:
                    self.smooth_tip = (raw_x, raw_y)
                else:
                    sx = int(self.alpha * self.smooth_tip[0] + (1 - self.alpha) * raw_x)
                    sy = int(self.alpha * self.smooth_tip[1] + (1 - self.alpha) * raw_y)
                    self.smooth_tip = (sx, sy)

                tip_x, tip_y = self.smooth_tip

                # velocity based on smoothed motion (more stable)
                if self.last_tip is not None:
                    dx = tip_x - self.last_tip[0]
                    dy = tip_y - self.last_tip[1]
                    velocity = math.sqrt(dx * dx + dy * dy)

                self.last_tip = (tip_x, tip_y)

                # trail (use smoothed points)
                self.tip_history.append((tip_x, tip_y))
                if len(self.tip_history) > TRAIL_LENGTH:
                    self.tip_history.pop(0)

            else:
                self.tip_history = []
                self.last_tip = None
                self.smooth_tip = None  # reset smoothing when hand disappears

            self._cached_tip = (tip_x, tip_y, velocity)

        return self._cached_tip

    def draw_trail(self, frame):
        n = len(self.tip_history)
        for i, (tx, ty) in enumerate(self.tip_history):
            alpha = (i + 1) / n
            radius = max(2, int(8 * alpha))
            tr, tg, tb = TRAIL_COLOR
            color = (int(tb * alpha), int(tg * alpha), int(tr * alpha))
            cv2.circle(frame, (tx, ty), radius, color, -1, cv2.LINE_AA)
        if n >= 2:
            for i in range(1, n):
                alpha = (i + 1) / n
                tr, tg, tb = TRAIL_COLOR
                color = (int(tb * alpha), int(tg * alpha), int(tr * alpha))
                cv2.line(frame, self.tip_history[i-1], self.tip_history[i],
                         color, max(1, int(3 * alpha)), cv2.LINE_AA)
# ═══════════════════════════════════════════════════════════════════════════════
# HEART DRAWING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def draw_heart(frame, cx, cy, size, color=(60, 60, 220)):
    """Draw a filled heart shape using two circles and a triangle polygon."""
    # Two circles for the top bumps
    r = size // 2
    cv2.circle(frame, (cx - r // 2, cy - r // 4), r // 2, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + r // 2, cy - r // 4), r // 2, color, -1, cv2.LINE_AA)
    # Triangle for the bottom point — four-point polygon
    pts = np.array([
        [cx - r, cy - r // 4],
        [cx + r, cy - r // 4],
        [cx,     cy + r]
    ], np.int32)
    cv2.fillPoly(frame, [pts], color)


# ═══════════════════════════════════════════════════════════════════════════════
# GAME STATE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class GameState:
    """Holds all mutable game state and handles state transitions."""

    def __init__(self):
        self.reset()
        self.state = 'start'         # 'start' | 'playing' | 'gameover'

    def reset(self):
        self.score = 0
        self.lives = LIVES
        self.fruits = []             # active Fruit objects
        self.particles = []          # active Particle objects
        self.slice_halves = []       # active SliceHalf objects
        self.frame_count = 0
        self.spawn_counter = 0       # counts frames since last spawn
        self.fruit_total = 0         # total fruits spawned (for bomb logic)
        self.flash_timer = 0         # red flash duration on bomb hit (frames)
        self.flash_color = (0, 0, 0)

    def spawn_fruit(self):
        """Spawn a new fruit or bomb depending on count."""
        if len(self.fruits) >= 8:
            return
        self.fruit_total += 1
        # Every Nth fruit is a bomb
        is_bomb = (self.fruit_total % BOMB_EVERY_N == 0)
        self.fruits.append(Fruit(is_bomb=is_bomb))

    def slice_fruit(self, fruit):
        """Handle a fruit being sliced — score, particles, halves."""
        if fruit.is_bomb:
            self.lives -= 1
            self.flash_timer = 4             # flash red for 4 frames
            self.flash_color = (0, 0, 180)   # BGR red-ish
        else:
            self.score += 1

        # Spawn 12 particles in fruit's flesh color
        for _ in range(12):
            p = Particle(fruit.x, fruit.y, fruit.inner_color)
            self.particles.append(p)

        # Spawn two flying halves
        for direction in (-1, 1):
            h = SliceHalf(fruit.x, fruit.y, fruit.radius,
                          fruit.inner_color, fruit.outline_color, direction)
            self.slice_halves.append(h)

        fruit.is_sliced = True
        self.fruits.remove(fruit)

    def update(self, tip_x, tip_y, velocity):
        """One frame of physics, collision, and cleanup."""
        self.frame_count += 1
        self.spawn_counter += 1

        # Spawn new fruit on schedule
        if self.spawn_counter >= SPAWN_RATE:
            self.spawn_fruit()
            self.spawn_counter = 0

        # Detect swipe and check collisions
        swiping = (tip_x is not None and velocity > SWIPE_THRESHOLD)
        sliced_this_frame = []
        for fruit in list(self.fruits):
            fruit.update()

            if swiping and not fruit.is_sliced:
                dx = tip_x - fruit.x
                dy = tip_y - fruit.y
                dist = math.sqrt(dx * dx + dy * dy)
                # Slice if fingertip is within fruit radius + 10 px
                if dist < fruit.radius + 10:
                    sliced_this_frame.append(fruit)

            # Remove fruits that have fully left the screen
            if fruit.is_off_screen():
                # Penalise if non-sliced fruit fell off the bottom
                if fruit.fell_off_bottom() and not fruit.is_sliced:
                    if not fruit.is_bomb:    # bombs falling off is fine
                        self.lives -= 1
                if fruit in self.fruits:
                    self.fruits.remove(fruit)

        # Apply slices after iterating to avoid mutation during loop
        for fruit in sliced_this_frame:
            if fruit in self.fruits:  # double-check still active
                self.slice_fruit(fruit)

        # Update particles
        for p in list(self.particles):
            p.update()
            if p.is_dead():
                self.particles.remove(p)

        # Update slice halves
        for h in list(self.slice_halves):
            h.update()
            if h.is_dead():
                self.slice_halves.remove(h)

        # Count down flash
        if self.flash_timer > 0:
            self.flash_timer -= 1

        # Check game over
        if self.lives <= 0:
            self.lives = 0
            return 'gameover'

        return 'playing'

    def draw_hud(self, frame):
        """Draw score and lives overlay."""
        # Score — top center, clean white text
        score_text = str(self.score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(score_text, font, 2.5, 4)[0]
        tx = (WIN_W - text_size[0]) // 2
        cv2.putText(frame, score_text, (tx, 70), font, 2.5,
                    (0, 0, 0), 7, cv2.LINE_AA)          # shadow
        cv2.putText(frame, score_text, (tx, 70), font, 2.5,
                    (255, 255, 255), 4, cv2.LINE_AA)

        # Lives — top left as hearts
        for i in range(self.lives):
            draw_heart(frame, 45 + i * 50, 45, 30)

    def apply_flash(self, frame):
        """Overlay a red flash when a bomb is hit."""
        if self.flash_timer > 0:
            # Create a red overlay and blend it onto the frame
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (WIN_W, WIN_H), (0, 0, 200), -1)
            alpha = self.flash_timer / 4 * 0.4          # fade as timer decreases
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_start_screen(frame):
    """Dark overlay with title and instructions."""
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title — large white text
    title = "FRUIT NINJA"
    ts = cv2.getTextSize(title, font, 3.5, 6)[0]
    tx = (WIN_W - ts[0]) // 2
    cv2.putText(frame, title, (tx, WIN_H // 2 - 60), font, 3.5,
                (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, WIN_H // 2 - 60), font, 3.5,
                (255, 255, 255), 6, cv2.LINE_AA)

    # Subtitle
    sub = "raise your index finger to begin"
    ss = cv2.getTextSize(sub, font, 1.0, 2)[0]
    sx = (WIN_W - ss[0]) // 2
    cv2.putText(frame, sub, (sx, WIN_H // 2 + 30), font, 1.0,
                (200, 200, 200), 2, cv2.LINE_AA)


def draw_gameover_screen(frame, score):
    """Game over overlay."""
    overlay = np.zeros_like(frame)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # GAME OVER — red
    go_text = "GAME OVER"
    ts = cv2.getTextSize(go_text, font, 3.0, 6)[0]
    tx = (WIN_W - ts[0]) // 2
    cv2.putText(frame, go_text, (tx, WIN_H // 2 - 80), font, 3.0,
                (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(frame, go_text, (tx, WIN_H // 2 - 80), font, 3.0,
                (0, 60, 220), 6, cv2.LINE_AA)

    # Score
    sc_text = f"Score: {score}"
    ss = cv2.getTextSize(sc_text, font, 1.8, 3)[0]
    sx = (WIN_W - ss[0]) // 2
    cv2.putText(frame, sc_text, (sx, WIN_H // 2), font, 1.8,
                (255, 255, 255), 3, cv2.LINE_AA)

    # Restart instructions
    inst = "press R to restart   or   Q to quit"
    ist = cv2.getTextSize(inst, font, 0.9, 2)[0]
    ix = (WIN_W - ist[0]) // 2
    cv2.putText(frame, inst, (ix, WIN_H // 2 + 60), font, 0.9,
                (180, 180, 180), 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIN_W)  # width
    cap.set(4, WIN_H)  # height

    tracker = HandTracker()
    game = GameState()

    window_name = "Fruit Ninja"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIN_W, WIN_H)

    prev_time = time.time()

    while True:
        # ── Capture & flip ────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            break
        # Flip horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)
        # Resize to exact target if camera resolution differs
        frame = cv2.resize(frame, (WIN_W, WIN_H))

        # ── Run MediaPipe on RGB copy ─────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tip_x, tip_y, velocity = tracker.process(frame_rgb)

        # ── State machine ─────────────────────────────────────────────────────
        if game.state == 'start':
            draw_start_screen(frame)
            # Start game as soon as a hand is detected
            if tip_x is not None:
                game.state = 'playing'
                game.reset()

        elif game.state == 'playing':
            # Physics & collision update
            result = game.update(tip_x, tip_y, velocity)
            if result == 'gameover':
                game.state = 'gameover'

            # ── Draw layer: fruits ────────────────────────────────────────────
            for fruit in game.fruits:
                fruit.draw(frame)

            # ── Draw layer: slice halves ──────────────────────────────────────
            for h in game.slice_halves:
                h.draw(frame)

            # ── Draw layer: particles ─────────────────────────────────────────
            for p in game.particles:
                p.draw(frame)

            # ── Draw layer: fingertip trail ───────────────────────────────────
            tracker.draw_trail(frame)

            # ── Draw layer: current fingertip indicator ───────────────────────
            if tip_x is not None:
                # Small green dot at live fingertip position
                cv2.circle(frame, (tip_x, tip_y), 8, (0, 220, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (tip_x, tip_y), 8, (0, 255, 0), 2, cv2.LINE_AA)

            # ── Draw layer: HUD ───────────────────────────────────────────────
            game.draw_hud(frame)

            # ── Red flash overlay on bomb hit ─────────────────────────────────
            game.apply_flash(frame)

        elif game.state == 'gameover':
            draw_gameover_screen(frame, game.score)

        # ── FPS cap ───────────────────────────────────────────────────────────
        now = time.time()
        elapsed = now - prev_time
        wait_ms = max(1, int((1.0 / FPS_TARGET - elapsed) * 1000))
        prev_time = now

        # ── Show frame ────────────────────────────────────────────────────────
        cv2.imshow(window_name, frame)

        # ── Key input ─────────────────────────────────────────────────────────
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q') or key == 27:    # Q or Escape to quit
            break
        if key == ord('r') and game.state == 'gameover':
            game.state = 'playing'
            game.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()