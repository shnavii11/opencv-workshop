# Concept: Particle Systems
# This project demonstrates a basic particle system using OpenCV.
# A particle system is a technique in computer graphics where many small
# entities (particles) are simulated to create effects like explosions,
# fire, smoke, or rain. Each particle has properties like position,
# velocity, and lifespan, and is updated over time.

import cv2
import numpy as np
import random
import math

# ---------------- PARTICLE CLASS ---------------- #
# The Particle class represents individual particles in the system.
# Each particle has position, velocity, and a limited lifespan.
class Particle:
    def __init__(self, x, y):
        # Initialize particle at given position (x, y)
        self.x = x
        self.y = y

        # Random direction and speed for realistic explosion spread
        angle = random.uniform(0, 2 * math.pi)  # Random angle in radians
        speed = random.uniform(2, 8)  # Random speed between 2-8 pixels/frame

        # Calculate velocity components using trigonometry
        self.vx = math.cos(angle) * speed  # Horizontal velocity
        self.vy = math.sin(angle) * speed  # Vertical velocity

        # Lifespan: how many frames the particle will live
        self.life = random.randint(20, 50)

    def update(self):
        # Apply gravity to make particles fall realistically
        self.vy += 0.2  # Gravity acceleration

        # Update position based on velocity
        self.x += self.vx
        self.y += self.vy

        # Decrease lifespan (particle dies when life reaches 0)
        self.life -= 1

    def draw(self, frame):
        # Only draw if particle is still alive
        if self.life > 0:
            # Color scheme: yellow → orange → red for explosion effect
            # Blue channel = 0, Green = random 150-255, Red = random 200-255
            color = (
                0,  # Blue
                random.randint(150, 255),  # Green (yellow to orange)
                random.randint(200, 255)   # Red (orange to red)
            )
            # Draw particle as a filled circle (radius 3, thickness -1)
            cv2.circle(frame, (int(self.x), int(self.y)), 3, color, -1)

    def alive(self):
        # Check if particle is still active (has life remaining)
        return self.life > 0


# ---------------- MOUSE INTERACTION ---------------- #
# Global list to store all active particles
particles = []

def mouse_callback(event, x, y, flags, param):
    global particles

    # Trigger explosion when left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Create 120 particles at the click position for a big explosion
        for _ in range(120):  # number of particles in explosion
            particles.append(Particle(x, y))


# ---------------- MAIN PROGRAM LOOP ---------------- #
# Capture video from webcam (device 0)
cap = cv2.VideoCapture(0)

# Create window and set up mouse callback for interaction
cv2.namedWindow('Explosion System')
cv2.setMouseCallback('Explosion System', mouse_callback)

# Main loop: process frames continuously
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit if can't read frame (webcam disconnected)

    # Flip frame horizontally for mirror effect (selfie view)
    frame = cv2.flip(frame, 1)

    # Update and draw all particles
    for p in particles:
        p.update()  # Update position and physics
        p.draw(frame)  # Draw particle on the frame

    # Remove dead particles from the list (garbage collection)
    # This keeps the list size manageable and improves performance
    particles = [p for p in particles if p.alive()]

    # Display the frame with particles overlaid
    cv2.imshow('Explosion System', frame)

    # Exit loop if 'q' key is pressed (wait 1ms for key input)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: release webcam and close windows
cap.release()
cv2.destroyAllWindows()