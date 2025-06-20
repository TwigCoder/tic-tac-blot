# using blot-cli at https://github.com/polypixeldev/blot-cli!
# cargo install blot-cli

import os
import time

def draw_board():
    os.system('blot go 0 0')
    os.system('blot go 30 10')

    os.system('blot pen down')
    os.system('blot go 30 70')
    os.system('blot pen up')

    os.system('blot go 50 10')

    os.system('blot pen down')
    os.system('blot go 50 70')
    os.system('blot pen up')

    os.system('blot go 10 30')

    os.system('blot pen down')
    os.system('blot go 70 30')
    os.system('blot pen up')

    os.system('blot go 10 50')

    os.system('blot pen down')
    os.system('blot go 70 50')
    os.system('blot pen up')

    os.system('blot go 0 0')


def draw_x(x, y, padding=5):
    # assuming x and y between 0 and 2
    x = x * 20 + 10 + padding
    y = y * 20 + 10 + padding
    end_x = x + 20 - 2 * padding
    end_y = y + 20 - 2 * padding

    os.system('blot go 0 0')
    os.system(f'blot go {x} {y}')
    
    os.system('blot pen down')
    os.system(f'blot go {end_x} {end_y}')
    os.system('blot pen up')

    os.system(f'blot go {x} {end_y}')
    
    os.system('blot pen down')
    os.system(f'blot go {end_x} {y}')
    os.system('blot pen up')

    os.system('blot go 0 0')


def draw_circ(x, y, padding=5):
    # I know its not a circle
    x = x * 20 + 10 + padding
    y = y * 20 + 10 + padding
    end_x = x + 20 - 2 * padding
    end_y = y + 20 - 2 * padding

    os.system('blot go 0 0')
    os.system(f'blot go {x} {y}')
    
    os.system('blot pen down')
    os.system(f'blot go {end_x} {y}')
    os.system(f'blot go {end_x} {end_y}')
    os.system(f'blot go {x} {end_y}')
    os.system(f'blot go {x} {y}')
    os.system('blot pen up')

    os.system('blot go 0 0')

draw_board()
draw_x(0,2)
draw_circ(1,1)