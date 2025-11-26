import pygame
import numpy as np
import threading
import time
import math
from kaira_core import KAIRACore 

class KAIRAUI:
    def __init__(self, core: KAIRACore):
        self.core = core
        pygame.init()
        
        # Screen setup
        self.screen = pygame.display.set_mode((1280, 720))
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("KAIRA (Waiting for 'hey kaira')")
        
        # --- Fonts ---
        self.caption_font = pygame.font.SysFont("Arial", 36, bold=True)
        # NEW: Font for KAIRA's response
        self.kaira_response_font = pygame.font.SysFont("Arial", 34, italic=True) 

        # --- Colors ---
        self.bg_color = (0, 0, 0)
        self.accent_color = (0, 208, 255)      # Bright blue (listening)
        self.accent_color_dim = (0, 150, 200)  # Dim blue (waiting)
        self.caption_color = (255, 255, 255)   # User's final text
        self.realtime_color = (180, 180, 180)  # User's realtime text
        self.kaira_response_color = (200, 200, 255) # KAIRA's text

        # --- UI State (independent of core state) ---
        self.current_mouth_scale = 1.0
        self.is_blinking = False
        self.blink_scale = 1.0
        self.last_blink_time = time.time()
        self.animation_time = 0.0
        self.mic_particles = []
        self.clock = pygame.time.Clock()
        
        # UI's local copy of core state
        self.caption_display_duration = 3.0 # Fade-out time
        self.current_display_text = ""      # User's text
        self.kaira_response_text = ""       # KAIRA's text
        self.is_final_sentence = False
        self.is_kaira_speaking = False
        self.last_sentence_time = 0
        self.normalized_amplitude = 0.0
        self.listening_state = 'WAITING'

        # --- Add a microphone button ---
        self.mic_button_rect = pygame.Rect(self.screen_width - 150, self.screen_height - 150, 100, 50)
        self.mic_button_color = (0, 208, 255)  # Bright blue
        self.mic_button_text = pygame.font.SysFont("Arial", 20).render("Mic", True, (255, 255, 255))

        print("KAIRA UI initialized.")

    def update_animations(self, dt):
        """Update smooth animations"""
        self.animation_time += dt
        scale_speed = 5.0
        
        # Decay visual amplitude
        self.normalized_amplitude *= (1.0 - 4.0 * dt)

        # Get latest state from core
        core_state = self.core.get_state()
        if core_state['normalized_amplitude'] > self.normalized_amplitude:
             self.normalized_amplitude = core_state['normalized_amplitude']
             
        # --- Update UI state from core ---
        self.listening_state = core_state['listening_state']
        self.current_display_text = core_state['display_text']
        self.kaira_response_text = core_state['kaira_response_text']
        self.is_final_sentence = core_state['is_final_sentence']
        self.is_kaira_speaking = core_state['is_kaira_speaking']
        self.last_sentence_time = core_state['last_sentence_time']
        
        if self.listening_state == 'WAITING':
             pygame.display.set_caption("KAIRA (Press Spacebar to Talk)")
        else:
             pygame.display.set_caption("KAIRA (Listening...)")

        # Mouth scale (always reacts to sound)
        target = 1.0 + self.normalized_amplitude * 0.5
        self.current_mouth_scale += (target - self.current_mouth_scale) * scale_speed * dt
        
        # Blinking (Unchanged)
        current_time = time.time()
        if current_time - self.last_blink_time > 3.0 + np.random.rand() * 2.0:
            self.trigger_blink()
            self.last_blink_time = current_time
        if self.is_blinking:
            self.blink_scale = max(0.05, self.blink_scale - 10.0 * dt)
            if self.blink_scale <= 0.05:
                self.blink_scale = 0.05
                threading.Timer(0.15, self.end_blink).start()
        else:
            self.blink_scale = min(1.0, self.blink_scale + 10.0 * dt)
        
        # Mic animation particles (Unchanged)
        if len(self.mic_particles) < 3 and np.random.rand() < 0.1:
            self.mic_particles.append({'radius': 20, 'alpha': 255, 'growth_rate': 60})
        for particle in self.mic_particles[:]:
            particle['radius'] += particle['growth_rate'] * dt
            particle['alpha'] = max(0, particle['alpha'] - 200 * dt)
            if particle['alpha'] <= 0:
                self.mic_particles.remove(particle)
            
        # --- NEW FADE-OUT LOGIC ---
        # This one timer now handles fading for BOTH user text and AI text
        time_since_last_sentence = time.time() - self.last_sentence_time
        
        # Check if we should fade the user's final sentence
        if self.is_final_sentence and not self.is_kaira_speaking and (time_since_last_sentence > self.caption_display_duration):
            # Fade user's text if KAIRA doesn't respond
            self.core.state['display_text'] = "" 
            self.core.state['is_final_sentence'] = False

        # Check if we should fade KAIRA's final response
        if not self.is_kaira_speaking and self.kaira_response_text and (time_since_last_sentence > self.caption_display_duration):
             # Fade KAIRA's text after she finishes
             self.core.state['kaira_response_text'] = ""


    def trigger_blink(self):
        self.is_blinking = True

    def end_blink(self):
        self.is_blinking = False

    def draw_3d_mic_animation(self):
        """Draw 3D mic animation (Unchanged)"""
        mic_x = self.screen_width - 120
        mic_y = self.screen_height - 120
        
        if self.listening_state == 'LISTENING':
            if self.normalized_amplitude > 0.3: 
                base_color = (255, 50, 50) # Red
                pulse = 0.8 + 0.2 * self.normalized_amplitude
            else: 
                base_color = self.accent_color # Bright Blue
                pulse = 0.8 + 0.1 * math.sin(self.animation_time * 4) 
        else: # 'WAITING' state
            base_color = self.accent_color_dim # Dim Blue
            pulse = 0.7
        
        particle_color = self.accent_color if self.listening_state == 'LISTENING' else self.accent_color_dim
        for particle in self.mic_particles:
            alpha = int(particle['alpha'])
            color = (*particle_color, alpha)
            size = int(particle['radius'] * 2 + 10)
            if size > 0:
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size // 2, size // 2), int(particle['radius']), 3)
                self.screen.blit(surf, (mic_x - size // 2, mic_y - size // 2))
        
        mic_size = int(35 * pulse)
        shadow_offset = 4
        pygame.draw.circle(self.screen, (20, 20, 20), 
                          (mic_x + shadow_offset, mic_y + shadow_offset), mic_size + 5)
        
        for i in range(3, 0, -1):
            alpha = int(100 / i)
            glow_color = (*base_color, alpha)
            glow_surf = pygame.Surface((mic_size * 3, mic_size * 3), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (mic_size * 3 // 2, mic_size * 3 // 2), mic_size + i * 8)
            self.screen.blit(glow_surf, (mic_x - mic_size * 3 // 2, mic_y - mic_size * 3 // 2))
        
        # ... (rest of mic drawing unchanged) ...
        for i in range(5):
            shade = tuple(max(0, c - i * 20) for c in base_color)
            pygame.draw.circle(self.screen, shade, (mic_x, mic_y - i), mic_size - i * 2)
        highlight_offset = int(mic_size * 0.3); pygame.draw.circle(self.screen, (255, 255, 255), (mic_x - highlight_offset, mic_y - highlight_offset), mic_size // 4)
        stem_width, stem_height = 12, 20; stem_rect = pygame.Rect(mic_x - stem_width // 2, mic_y + mic_size - 5, stem_width, stem_height); pygame.draw.rect(self.screen, base_color, stem_rect, border_radius=6)
        base_width, base_height = 30, 8; base_rect = pygame.Rect(mic_x - base_width // 2, mic_y + mic_size + stem_height - 8, base_width, base_height); pygame.draw.rect(self.screen, base_color, base_rect, border_radius=4)


    def draw_face(self):
        """Draw the minimalist robot face (Unchanged)"""
        self.screen.fill(self.bg_color)
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        scale = min(self.screen_width, self.screen_height) / 100
        
        eye_color = self.accent_color if self.listening_state == 'LISTENING' else self.accent_color_dim
        eye_width, eye_height = int(20 * scale), int(30 * scale * self.blink_scale)
        eye_y = center_y - int(20 * scale)
        
        left_eye_rect = pygame.Rect(center_x - int(30 * scale) - eye_width // 2, eye_y - eye_height // 2, eye_width, eye_height)
        pygame.draw.rect(self.screen, eye_color, left_eye_rect, border_radius=int(5 * scale))
        right_eye_rect = pygame.Rect(center_x + int(30 * scale) - eye_width // 2, eye_y - eye_height // 2, eye_width, eye_height)
        pygame.draw.rect(self.screen, eye_color, right_eye_rect, border_radius=int(5 * scale))
        
        mouth_y, mouth_width = center_y + int(30 * scale), int(30 * scale)
        pygame.draw.line(self.screen, eye_color, (center_x - mouth_width, mouth_y), (center_x + mouth_width, mouth_y), int(5 * scale * self.current_mouth_scale))

        self.draw_3d_mic_animation()
        
        # --- NEW: Prioritized Caption Drawing ---
        
        # KAIRA's text takes priority. If it exists, draw it.
        if self.kaira_response_text:
            self.draw_wrapped_text(
                self.kaira_response_text,
                self.kaira_response_font,
                self.kaira_response_color,
                self.screen_width // 2, 
                self.screen_height - 80, # At the bottom
                max_width=self.screen_width - 100
            )
        # Otherwise, draw the user's text.
        elif self.current_display_text:
            # Choose color based on whether text is final or realtime
            color = self.caption_color if self.is_final_sentence else self.realtime_color
            self.draw_wrapped_text(
                self.current_display_text,
                self.caption_font,
                color,
                self.screen_width // 2, 
                self.screen_height - 80, # At the bottom
                max_width=self.screen_width - 100
            )

        pygame.display.flip()

    def draw_wrapped_text(self, text, font, color, center_x, bottom_y, max_width):
        """Helper function to draw word-wrapped text (Unchanged)"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        
        line_height = font.get_linesize()
        for i, line in enumerate(reversed(lines)):
            line_surf = font.render(line, True, color)
            line_rect = line_surf.get_rect(
                center=(center_x, bottom_y - i * line_height - line_height // 2)
            )
            self.screen.blit(line_surf, line_rect)

    def draw_mic_button(self):
        """Draw the microphone button on the screen."""
        pygame.draw.rect(self.screen, self.mic_button_color, self.mic_button_rect, border_radius=10)
        text_rect = self.mic_button_text.get_rect(center=self.mic_button_rect.center)
        self.screen.blit(self.mic_button_text, text_rect)

    def handle_mic_button_click(self, event):
        """Handle click events for the microphone button."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.mic_button_rect.collidepoint(event.pos):
            print("Microphone button clicked!")
            self.core.start_recording()

    def run(self):
        """Main UI application loop (MODIFIED)"""
        
        # --- NEW: Updated print instructions ---
        print("=" * 70)
        print("🤖 KAIRA - STT & Animation Demo (Tap-to-Talk)")
        print("=" * 70)
        print("\n🎮 CONTROLS:")
        print("  PRESS Spacebar to talk.")
        print("  ESC    → Exit")
        print("=" * 70 + "\n")
        # --- END NEW ---
        
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            
            # --- Key Event Handling (Wake Word Mode) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        # Only call start_recording.
                        # We will no longer check if it's already recording,
                        # the core will handle that.
                        self.core.start_recording()
                self.handle_mic_button_click(event)  # Handle mic button clicks

            # --- END MODIFIED ---
            
            self.update_animations(dt)
            self.draw_face()
            self.draw_mic_button()  # Draw the mic button

        self.cleanup()

    def cleanup(self):
        """Clean up UI resources (Unchanged)"""
        try:
            pygame.quit()
        except: pass
        print("KAIRA UI shut down.")