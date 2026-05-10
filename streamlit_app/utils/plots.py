PLOT_BG  = "rgba(0, 0, 0, 0)"
PAPER_BG = "rgba(0, 0, 0, 0)"
GRID_CLR = "rgba(255, 255, 255, 0.05)"
FONT_CLR = "rgba(255, 255, 255, 0.5)"
FONT_FAM = "DM Sans"

def base_layout(height = 360):
    return dict(
        paper_bgcolor = PAPER_BG, plot_bgcolor = PLOT_BG,
        font = dict(family = FONT_FAM, color = FONT_CLR, size = 12),
        height = height, margin = dict(t = 24, b = 24, l = 0, r = 0),
        xaxis = dict(gridcolor = GRID_CLR, zeroline = False),
        yaxis = dict(gridcolor = GRID_CLR, zeroline = False),
    )