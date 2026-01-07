from common.path_config import PROJECT_ROOT

import pygame
from pygame.locals import *
from enum import IntEnum, unique

@unique
class JoystickButton(IntEnum):
    # Standard PlayStation/Xbox Layout
    A = 0      # PS: Cross(×), Xbox: A
    B = 1      # PS: Circle(○), Xbox: B
    X = 2      # PS: Square(□), Xbox: X
    Y = 3      # PS: Triangle(△), Xbox: Y
    L1 = 4     # Left Bumper (L1 on PS)
    R1 = 5     # Right Bumper (R1 on PS)
    SELECT = 6   # Select/Share button
    START = 7  # Start/Options button
    L3 = 8     # Left Stick Press
    R3 = 9     # Right Stick Press
    HOME = 10  # PS: PS FSMCommand, Xbox: Xbox FSMCommand
    UP = 11    # D-pad Up (if mapped as separate button)
    DOWN = 12  # D-pad Down
    LEFT = 13  # D-pad Left
    RIGHT = 14 # D-pad Right

class JoyStick:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise RuntimeError("No joystick connected!")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        self.button_count = self.joystick.get_numbuttons()
        self.button_states = [False] * self.button_count  
        self.button_pressed = [False] * self.button_count  
        self.button_released = [False] * self.button_count 

        self.axis_count = self.joystick.get_numaxes()
        self.axis_states = [0.0] * self.axis_count
        
        self.hat_count = self.joystick.get_numhats()
        self.hat_states = [(0, 0)] * self.hat_count
        
        
    def update(self):
        """update joystick state"""
        pygame.event.pump()  
        
        self.button_released = [False] * self.button_count
        
        for i in range(self.button_count):
            current_state = self.joystick.get_button(i) == 1
            if self.button_states[i] and not current_state:
                self.button_released[i] = True
            self.button_states[i] = current_state

        for i in range(self.axis_count):
            self.axis_states[i] = self.joystick.get_axis(i)
        
        for i in range(self.hat_count):
            self.hat_states[i] = self.joystick.get_hat(i)

    def is_button_pressed(self, button_id):
        """detect button pressed"""
        if 0 <= button_id < self.button_count:
            return self.button_states[button_id]
        return False

    def is_button_released(self, button_id):
        """detect button released"""
        if 0 <= button_id < self.button_count:
            return self.button_released[button_id]
        return False

    def get_axis_value(self, axis_id):
        """get joystick axis value"""
        if 0 <= axis_id < self.axis_count:
            return self.axis_states[axis_id]
        return 0.0

    def get_hat_direction(self, hat_id=0):
        """get joystick hat direction"""
        if 0 <= hat_id < self.hat_count:
            return self.hat_states[hat_id]
        return (0, 0)