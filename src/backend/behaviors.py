from enum import Enum


class Behaviors(Enum):
    """Enum for behaviors and their index in database"""
    SAFE_DRIVING = 0
    TEXTING = 1
    TALKING_USING_PHONE = 2
    DRINKING = 3
    HEAD_DOWN = 4
    LOOK_BEHIND = 5
