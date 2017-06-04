import numpy as np
import pandas as pd

def leftShift2D(df, rows, cols, offset=1, fill=0):
    new_index= []
    for i in range(rows):
        new_index += range(offset + cols * i, cols * (i + 1))
        new_index += range(cols * i, cols * i + offset)
    new_index = [0] + [i + 1 for i in new_index]
    new_df = df[:, new_index]
    for i in range(1, rows + 1):
        for j in range(cols * i - offset, cols * i):
            new_df[:, j + 1] = 0
    return new_df

def rightShift2D(df, rows, cols, offset=1, fill=0):
    new_index= []
    for i in range(rows):
        new_index += range(cols * i, cols * i + offset)
        new_index += range(offset + cols * i, cols * (i + 1))
    new_index = [0] + [i + 1 for i in new_index]
    new_df = df[:, new_index]
    for i in range(0, rows):
        for j in range(cols * i, cols * i + offset):
            new_df[:, j + 1] = 0
    return new_df

def downShift2D(df, rows, cols, offset=1, fill=0):
    new_index = range(0, rows * cols)
    breakpoint = (rows - offset) * cols
    new_index = new_index[breakpoint:] + new_index[:breakpoint]
    new_index = [0] + [i + 1 for i in new_index]
    new_df = df[:, new_index]
    for i in range(0, offset * cols):
        new_df[:, i + 1] = 0
    return new_df

def upShift2D(df, rows, cols, offset=1, fill=0):
    new_index = range(0, rows * cols)
    breakpoint = offset * cols
    new_index = new_index[breakpoint:] + new_index[:breakpoint]
    new_index = [0] + [i + 1 for i in new_index]
    new_df = df[:, new_index]
    for i in range((rows - offset) * cols, rows * cols):
        new_df[:, i + 1] = 0
    return new_df


def expand(df, rows, cols, offset=1, fill=0):
    df = np.uint8(df)
    return np.concatenate([leftShift2D(df, rows, cols, offset),
                           rightShift2D(df, rows, cols, offset),
                           upShift2D(df, rows, cols, offset),
                           downShift2D(df, rows, cols, offset)])