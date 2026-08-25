__author__ = 'justinarmstrong'

"""
This module initializes the display and creates dictionaries of resources.
"""

import os
import pygame as pg
from . import tools
from .import constants as c

ORIGINAL_CAPTION = c.ORIGINAL_CAPTION


# NOTE (modified from upstream): this module originally began with
#     os.environ['SDL_VIDEO_CENTERED'] = '1'
# which has been removed. custom_mario_env.py deliberately sets
# SDL_VIDEO_WINDOW_POS (and deletes SDL_VIDEO_CENTERED) just before importing
# this module, so that 8 parallel training windows land at different spots
# instead of stacking on top of each other. Because this line ran afterwards,
# it silently put SDL_VIDEO_CENTERED back and centering won every time -
# verified directly: with both variables set, a window requested at (137,241)
# was actually created at (360,101). The staggering never worked until this
# line was removed.
pg.init()
pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
pg.display.set_caption(c.ORIGINAL_CAPTION)
SCREEN = pg.display.set_mode(c.SCREEN_SIZE)
SCREEN_RECT = SCREEN.get_rect()


# Resource paths are built from this file's own location rather than the
# process's current working directory. Upstream used relative paths
# ("resources/music/..."), which forced every caller to chdir into
# mario_clone/ first and made the game silently unloadable from anywhere
# else. pg.mixer.music.load() stores these paths and re-opens them at
# runtime, so relative paths were a live dependency on cwd during play, not
# just at import.
_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")

MUSIC = tools.load_all_music(os.path.join(_RESOURCES, "music"))
GFX   = tools.load_all_gfx(os.path.join(_RESOURCES, "graphics"))
SFX   = tools.load_all_sfx(os.path.join(_RESOURCES, "sound"))


