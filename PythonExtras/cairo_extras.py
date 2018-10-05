import cairo


def cairo_show_text_centered(cr, text: str, x, y):
    xBearing, yBearing, width, height, xAdvance, yAdvance = cr.text_extents(text)
    x -= width / 2 + xBearing
    y -= height / 2 + yBearing

    cr.move_to(x, y)
    cr.show_text(text)